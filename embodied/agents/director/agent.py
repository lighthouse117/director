import sys
import pprint

import embodied
import ruamel.yaml as yaml
import tensorflow as tf
from tensorflow.keras import mixed_precision as prec
from tensorflow_probability import distributions as tfd

from . import behaviors
from . import nets
from . import tfagent
from . import tfutils


class Agent(tfagent.TFAgent):
    configs = yaml.YAML(typ="safe").load(
        (embodied.Path(sys.argv[0]).parent / "configs.yaml").read()
    )

    def __init__(self, obs_space, act_space, step, config):
        print("[Agent.__init__]")
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.step = step
        self.wm = WorldModel(obs_space, config)
        # Hierarchy class
        self.task_behavior = getattr(behaviors, config.task_behavior)(
            self.wm, self.act_space, self.config
        )
        if config.expl_behavior == "None":
            self.expl_behavior = self.task_behavior
        else:
            self.expl_behavior = getattr(behaviors, config.expl_behavior)(
                self.wm, self.act_space, self.config
            )
        self.initial_policy_state = tf.function(
            lambda obs: (
                self.wm.rssm.initial(len(obs["is_first"])),
                self.task_behavior.initial(len(obs["is_first"])),
                self.expl_behavior.initial(len(obs["is_first"])),
                tf.zeros((len(obs["is_first"]),) + self.act_space.shape),
            )
        )
        self.initial_train_state = tf.function(
            lambda obs: (self.wm.rssm.initial(len(obs["is_first"])))
        )

    @tf.function(jit_compile=False)
    def policy(self, obs, state=None, mode="train"):
        # tf.print("[Agent.policy]")
        self.config.tf.jit and print("Tracing policy function.")
        if state is None:
            state = self.initial_policy_state(obs)
        obs = self.preprocess(obs)
        latent, task_state, expl_state, action = state
        embed = self.wm.encoder(obs)
        latent, _ = self.wm.rssm.obs_step(latent, action, embed, obs["is_first"])
        noise = self.config.expl_noise
        if mode == "eval":
            noise = self.config.eval_noise
            outs, task_state = self.task_behavior.policy(latent, task_state)
            outs = {**outs, "action": outs["action"].mode()}
        elif mode == "explore":
            outs, expl_state = self.expl_behavior.policy(latent, expl_state)
            outs = {**outs, "action": outs["action"].sample()}
        elif mode == "train":
            outs, task_state = self.task_behavior.policy(latent, task_state)
            outs = {**outs, "action": outs["action"].sample()}
        outs = {
            **outs,
            "action": tfutils.action_noise(outs["action"], noise, self.act_space),
        }
        state = (latent, task_state, expl_state, outs["action"])
        # tf.print("[/Agent.policy]")
        return outs, state

    @tf.function(jit_compile=False)
    def train(self, data, state=None):
        # tf.print("[Agent.train]", output_stream=sys.stdout)
        self.config.tf.jit and print("Tracing train function.")
        metrics = {}
        if state is None:
            state = self.initial_train_state(data)
        data = self.preprocess(data)

        # 世界モデルの学習
        state, wm_outs, mets = self.wm.train(data, state)

        metrics.update(mets)
        context = {**data, **wm_outs["post"]}
        start = tf.nest.map_structure(
            lambda x: x.reshape([-1] + list(x.shape[2:])), context
        )

        # print("context")
        # pprint.pprint(context)
        # print("start")
        # pprint.pprint(start)

        # Goal Autoencoder + Manager + Worker の学習 (Hierarchy class)
        # tf.print("[Agent.train] task_behavior.train")
        _, mets = self.task_behavior.train(self.wm.imagine, start, context)

        metrics.update(mets)

        # 探索行動の学習（上と全く同じ）
        if self.config.expl_behavior != "None":
            # tf.print("[Agent.train] expl_behavior.train")
            _, mets = self.expl_behavior.train(self.wm.imagine, start, context)
            metrics.update({"expl_" + key: value for key, value in mets.items()})

        outs = {}
        if "key" in data:
            criteria = {**data, **wm_outs}
            outs.update(key=data["key"], priority=criteria[self.config.priority])
        # tf.print("[/Agent.train]")
        return outs, state, metrics

    @tf.function(jit_compile=False)
    def report(self, data):
        self.config.tf.jit and print("Tracing report function.")
        data = self.preprocess(data)
        report = {}
        report.update(self.wm.report(data))
        mets = self.task_behavior.report(data)
        report.update({f"task_{k}": v for k, v in mets.items()})
        if self.expl_behavior is not self.task_behavior:
            mets = self.expl_behavior.report(data)
            report.update({f"expl_{k}": v for k, v in mets.items()})
        return report

    def dataset(self, generator):
        if self.config.data_loader == "tfdata":
            example = next(generator())
            dtypes = {k: v.dtype for k, v in example.items()}
            shapes = {k: v.shape for k, v in example.items()}
            return (
                tf.data.Dataset.range(self.config.batch_size)
                .interleave(
                    lambda _: tf.data.Dataset.from_generator(generator, dtypes, shapes),
                )
                .batch(self.config.batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )
        elif self.config.data_loader == "embodied":
            return embodied.Prefetch(
                sources=[generator] * self.config.batch_size, workers=8, prefetch=4
            )
        else:
            raise NotImplementedError(self.config.data_loader)

    def preprocess(self, obs):
        """
        Preprocesses observations to be compatible with the world model.
        """
        dtype = prec.global_policy().compute_dtype
        obs = {k: tf.tensor(v) for k, v in obs.items()}
        obs = obs.copy()
        for key, value in obs.items():
            if key.startswith("log_") or key in ("key",):
                continue
            if len(value.shape) > 3 and value.dtype == tf.uint8:
                value = value.astype(dtype) / 255.0
            else:
                value = value.astype(tf.float32)
            obs[key] = value
        obs["reward"] = {
            "off": tf.identity,
            "sign": tf.sign,
            "tanh": tf.tanh,
            "symlog": tfutils.symlog,
        }[self.config.transform_rewards](obs["reward"])
        obs["cont"] = 1.0 - obs["is_terminal"].astype(tf.float32)
        return obs


class WorldModel(tfutils.Module):
    def __init__(self, obs_space, config):
        print("[WorldModel.__init__]")
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        shapes = {k: v for k, v in shapes.items() if not k.startswith("log_")}
        self.config = config
        self.rssm = nets.RSSM(**config.rssm)
        self.encoder = nets.MultiEncoder(shapes, **config.encoder)
        self.heads = {}
        self.heads["decoder"] = nets.MultiDecoder(shapes, **config.decoder)
        self.heads["reward"] = nets.MLP((), **config.reward_head)
        self.heads["cont"] = nets.MLP((), **config.cont_head)
        self.model_opt = tfutils.Optimizer("model", **config.model_opt)
        self.wmkl = tfutils.AutoAdapt((), **self.config.wmkl, inverse=False)

    def train(self, data, state=None):
        # tf.print("[WorldModel.train]")

        # 勾配計算のために計算内容を記録しておく
        with tf.GradientTape() as model_tape:
            # 損失関数を計算
            model_loss, state, outputs, metrics = self.loss(data, state, training=True)

        # 世界モデル一式
        # heads = Reward, Discount, Decoder
        modules = [self.encoder, self.rssm, *self.heads.values()]

        # Optimizer.__call__()で勾配を計算し、パラメータを更新
        metrics.update(self.model_opt(model_tape, model_loss, modules))

        # tf.print("[/WorldModel.train]")
        return state, outputs, metrics

    def loss(self, data, state=None, training=False):
        """
        世界モデルの損失関数をEnd-to-Endで計算する
        """
        metrics = {}

        # 観測画像をエンコーダーに入力し、潜在ベクトルを取得
        embed = self.encoder(data)
        # print("Input image into encoder")
        # print("embed.shape = ", embed.shape)

        # エンコードされた画像入力から求めた潜在表現（posterior z）と
        # RSSMのhのみから推論した潜在表現（prior z）を計算
        post, prior = self.rssm.observe(embed, data["action"], data["is_first"], state)
        # print("RSSM.observe()")
        # print("post: ", post)
        # print("prior: ", prior)
        # deter: deterministic
        # stoch: stochastic
        # logit: logit

        dists = {}

        # 勾配計算を止めて定数に
        post_const = tf.nest.map_structure(tf.stop_gradient, post)

        # 勾配計算を止めたpost潜在表現をheadsの各モジュールに入力
        # Reward: 報酬予測 (MLP)
        # Count: 割引率予測（MLP）
        # Decoder: 画像予測（CNN デコーダ）
        for name, head in self.heads.items():
            # 勾配計算を止めたpost潜在表現を入力
            out = head(post if name in self.config.grad_heads else post_const)
            if not isinstance(out, dict):
                out = {name: out}
            # print(f"Output of {name} head: ", out)
            dists.update(out)

        losses = {}

        # posterior z　と prior z^　を比較してKLロスを計算
        # wmkl_balance = 0.8
        kl = self.rssm.kl_loss(post, prior, self.config.wmkl_balance)
        kl, mets = self.wmkl(kl, update=training)
        losses["kl"] = kl
        metrics.update({f"wmkl_{k}": v for k, v in mets.items()})

        # reward, cont, decoderモジュールのロスを計算
        # - Image log loss
        # - Reward log loss
        # - Discount log loss
        for key, dist in dists.items():
            losses[key] = -dist.log_prob(data[key].astype(tf.float32))

        metrics.update({f"{k}_loss_mean": v.mean() for k, v in losses.items()})
        metrics.update({f"{k}_loss_std": v.std() for k, v in losses.items()})

        # 4つのロスを重み付けして合計
        scaled = {}
        for key, loss in losses.items():
            assert loss.shape == embed.shape[:2], (key, loss.shape)
            scaled[key] = loss * self.config.loss_scales.get(key, 1.0)
        model_loss = sum(scaled.values())

        if "prob" in data and self.config.priority_correct:
            weights = (1.0 / data["prob"]) ** self.config.priority_correct
            weights /= weights.max()
            assert weights.shape == model_loss.shape
            model_loss *= weights

        out = {"embed": embed, "post": post, "prior": prior}
        out.update({f"{k}_loss": v for k, v in losses.items()})
        metrics["prior_ent"] = self.rssm.get_dist(prior).entropy().mean()
        metrics["post_ent"] = self.rssm.get_dist(post).entropy().mean()
        metrics["model_loss_mean"] = model_loss.mean()
        metrics["model_loss_std"] = model_loss.std()

        if not self.config.tf.debug_nans:
            if "reward" in dists:
                stats = tfutils.balance_stats(dists["reward"], data["reward"], 0.1)
                metrics.update({f"reward_{k}": v for k, v in stats.items()})
            if "cont" in dists:
                stats = tfutils.balance_stats(dists["cont"], data["cont"], 0.5)
                metrics.update({f"cont_{k}": v for k, v in stats.items()})

        # state = 末尾タイムステップのpost潜在表現
        last_state = {k: v[:, -1] for k, v in post.items()}
        return model_loss.mean(), last_state, out, metrics

    def imagine(self, policy, start, horizon):
        first_cont = (1.0 - start["is_terminal"]).astype(tf.float32)
        keys = list(self.rssm.initial(1).keys())
        start = {k: v for k, v in start.items() if k in keys}
        start["action"] = policy(start)

        def step(prev, _):
            prev = prev.copy()
            action = prev.pop("action")
            state = self.rssm.img_step(prev, action)
            action = policy(state)
            return {**state, "action": action}

        traj = tfutils.scan(step, tf.range(horizon), start, self.config.imag_unroll)
        traj = {k: tf.concat([start[k][None], v], 0) for k, v in traj.items()}
        traj["cont"] = tf.concat(
            [first_cont[None], self.heads["cont"](traj).mean()[1:]], 0
        )
        traj["weight"] = (
            tf.math.cumprod(self.config.discount * traj["cont"]) / self.config.discount
        )
        return traj

    def imagine_carry(self, policy, start, horizon, carry):
        first_cont = (1.0 - start["is_terminal"]).astype(tf.float32)
        keys = list(self.rssm.initial(1).keys())
        start = {k: v for k, v in start.items() if k in keys}
        keys += list(carry.keys()) + ["action"]
        states = [start]
        outs, carry = policy(start, carry)
        action = outs["action"]
        if hasattr(action, "sample"):
            action = action.sample()
        actions = [action]
        carries = [carry]

        # horizon = 16
        # 数タイムステップ先までをループで予測
        for _ in range(horizon):
            # RSSMに潜在状態と行動を入力して次の状態を予測
            states.append(self.rssm.img_step(states[-1], actions[-1]))

            # 行動を決定
            outs, carry = policy(states[-1], carry)
            action = outs["action"]

            if hasattr(action, "sample"):
                action = action.sample()
            actions.append(action)
            carries.append(carry)

        transp = lambda x: {k: [x[t][k] for t in range(len(x))] for k in x[0]}
        traj = {**transp(states), **transp(carries), "action": actions}
        traj = {k: tf.stack(v, 0) for k, v in traj.items()}
        cont = self.heads["cont"](traj).mean()
        cont = tf.concat([first_cont[None], cont[1:]], 0)
        traj["cont"] = cont
        traj["weight"] = (
            tf.math.cumprod(self.config.imag_discount * cont)
            / self.config.imag_discount
        )
        return traj

    def report(self, data):
        report = {}
        report.update(self.loss(data)[-1])
        context, _ = self.rssm.observe(
            self.encoder(data)[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        start = {k: v[:, -1] for k, v in context.items()}
        recon = self.heads["decoder"](context)
        openl = self.heads["decoder"](self.rssm.imagine(data["action"][:6, 5:], start))
        for key in self.heads["decoder"].cnn_shapes.keys():
            truth = data[key][:6].astype(tf.float32)
            model = tf.concat([recon[key].mode()[:, :5], openl[key].mode()], 1)
            error = (model - truth + 1) / 2
            video = tf.concat([truth, model, error], 2)
            report[f"openl_{key}"] = tfutils.video_grid(video)
        return report


# Worker & Manager
class ImagActorCritic(tfutils.Module):
    def __init__(self, critics, scales, act_space, config):
        critics = {k: v for k, v in critics.items() if scales[k]}
        for key, scale in scales.items():
            assert not scale or key in critics, key
        self.critics = {k: v for k, v in critics.items() if scales[k]}
        self.scales = scales
        self.act_space = act_space
        self.config = config
        self.actor = nets.MLP(
            act_space.shape,
            **self.config.actor,
            dist=(
                config.actor_dist_disc if act_space.discrete else config.actor_dist_cont
            ),
        )
        self.grad = (
            config.actor_grad_disc if act_space.discrete else config.actor_grad_cont
        )
        self.advnorm = tfutils.Normalize(**self.config.advnorm)
        self.retnorms = {
            k: tfutils.Normalize(**self.config.retnorm) for k in self.critics
        }
        self.scorenorms = {
            k: tfutils.Normalize(**self.config.scorenorm) for k in self.critics
        }
        if self.config.actent_perdim:
            shape = act_space.shape[:-1] if act_space.discrete else act_space.shape
            self.actent = tfutils.AutoAdapt(shape, **self.config.actent, inverse=True)
        else:
            self.actent = tfutils.AutoAdapt((), **self.config.actent, inverse=True)
        self.opt = tfutils.Optimizer("actor", **self.config.actor_opt)

    def initial(self, batch_size):
        return None

    def policy(self, state, carry):
        return {"action": self.actor(state)}, carry

    def train(self, imagine, start, context):
        policy = lambda s: self.actor(
            tf.nest.map_structure(tf.stop_gradient, s)
        ).sample()
        with tf.GradientTape(persistent=True) as tape:
            traj = imagine(policy, start, self.config.imag_horizon)
        metrics = self.update(traj, tape)
        return traj, metrics

    def update(self, traj, tape=None):
        tape = tape or tf.GradientTape()
        metrics = {}
        for key, critic in self.critics.items():
            mets = critic.train(traj, self.actor)
            metrics.update({f"{key}_{k}": v for k, v in mets.items()})
        with tape:
            scores = []
            for key, critic in self.critics.items():
                ret, baseline = critic.score(traj, self.actor)
                ret = self.retnorms[key](ret)
                baseline = self.retnorms[key](baseline, update=False)
                score = self.scorenorms[key](ret - baseline)
                metrics[f"{key}_score_mean"] = score.mean()
                metrics[f"{key}_score_std"] = score.std()
                metrics[f"{key}_score_mag"] = tf.abs(score).mean()
                metrics[f"{key}_score_max"] = tf.abs(score).max()
                scores.append(score * self.scales[key])
            score = self.advnorm(tf.reduce_sum(scores, 0))
            loss, mets = self.loss(traj, score)
            metrics.update(mets)
            loss = loss.mean()
        metrics.update(self.opt(tape, loss, self.actor))
        return metrics

    def loss(self, traj, score):
        metrics = {}
        policy = self.actor(tf.nest.map_structure(tf.stop_gradient, traj))
        action = tf.stop_gradient(traj["action"])
        if self.grad == "backprop":
            loss = -score
        elif self.grad == "reinforce":
            loss = -policy.log_prob(action)[:-1] * tf.stop_gradient(score)
        else:
            raise NotImplementedError(self.grad)

        shape = (
            self.act_space.shape[:-1]
            if self.act_space.discrete
            else self.act_space.shape
        )
        if self.config.actent_perdim and len(shape) > 0:
            assert isinstance(policy, tfd.Independent), type(policy)
            ent = policy.distribution.entropy()[:-1]
            if self.config.actent_norm:
                lo = policy.minent / ent.shape[-1]
                hi = policy.maxent / ent.shape[-1]
                ent = (ent - lo) / (hi - lo)
            ent_loss, mets = self.actent(ent)
            ent_loss = ent_loss.sum(-1)

        else:
            ent = policy.entropy()[:-1]
            if self.config.actent_norm:
                lo, hi = policy.minent, policy.maxent
                ent = (ent - lo) / (hi - lo)
            ent_loss, mets = self.actent(ent)

        metrics.update({f"actent_{k}": v for k, v in mets.items()})
        loss += ent_loss
        loss *= tf.stop_gradient(traj["weight"])[:-1]
        return loss, metrics


class VFunction(tfutils.Module):
    def __init__(self, rewfn, config):
        assert "action" not in config.critic.inputs, config.critic.inputs
        self.rewfn = rewfn
        self.config = config
        self.net = nets.MLP((), **self.config.critic)
        if self.config.slow_target:
            self.target_net = nets.MLP((), **self.config.critic)
            self.updates = tf.Variable(-1, dtype=tf.int64)
        else:
            self.target_net = self.net
        self.opt = tfutils.Optimizer("critic", **self.config.critic_opt)

    def train(self, traj, actor):
        metrics = {}
        reward = self.rewfn(traj)
        target = tf.stop_gradient(
            self.target(traj, reward, self.config.critic_return)[0]
        )
        with tf.GradientTape() as tape:
            dist = self.net({k: v[:-1] for k, v in traj.items()})
            loss = -(dist.log_prob(target) * traj["weight"][:-1]).mean()
        metrics.update(self.opt(tape, loss, self.net))
        metrics.update(
            {
                "critic_loss": loss,
                "imag_reward_mean": reward.mean(),
                "imag_reward_std": reward.std(),
                "imag_critic_mean": dist.mean().mean(),
                "imag_critic_std": dist.mean().std(),
                "imag_return_mean": target.mean(),
                "imag_return_std": target.std(),
            }
        )
        self.update_slow()
        return metrics

    def score(self, traj, actor):
        return self.target(traj, self.rewfn(traj), self.config.actor_return)

    def target(self, traj, reward, impl):
        if len(reward) != len(traj["action"]) - 1:
            raise AssertionError("Should provide rewards for all but last action.")
        disc = traj["cont"][1:] * self.config.discount
        value = self.target_net(traj).mean()
        if impl == "gae":
            advs = [tf.zeros_like(value[0])]
            deltas = reward + disc * value[1:] - value[:-1]
            for t in reversed(range(len(disc))):
                advs.append(deltas[t] + disc[t] * self.config.return_lambda * advs[-1])
            adv = tf.stack(list(reversed(advs))[:-1])
            return adv + value[:-1], value[:-1]
        elif impl == "gve":
            vals = [value[-1]]
            interm = reward + disc * value[1:] * (1 - self.config.return_lambda)
            for t in reversed(range(len(disc))):
                vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
            ret = tf.stack(list(reversed(vals))[:-1])
            return ret, value[:-1]
        else:
            raise NotImplementedError(impl)

    def update_slow(self):
        if not self.config.slow_target:
            return
        assert self.net.variables
        initialize = self.updates == -1
        if initialize or self.updates >= self.config.slow_target_update:
            self.updates.assign(0)
            mix = 1.0 if initialize else self.config.slow_target_fraction
            for s, d in zip(self.net.variables, self.target_net.variables):
                d.assign(mix * s + (1 - mix) * d)
        self.updates.assign_add(1)


class QFunction(tfutils.Module):
    def __init__(self, rewfn, config):
        assert config.actor_grad_disc == "backprop"
        assert config.actor_grad_cont == "backprop"
        assert "action" in config.actor.inputs
        self.rewfn = rewfn
        self.config = config
        self.net = nets.MLP((), **self.config.critic)
        if self.config.slow_target:
            self.target_net = nets.MLP((), **self.config.critic)
            self.updates = tf.Variable(-1, dtype=tf.int64)
        else:
            self.target_net = self.net
        self.opt = tfutils.Optimizer("critic", **self.config.critic_opt)

    def score(self, traj, actor):
        traj = tf.nest.map_structure(tf.stop_gradient, traj)
        ret = self.net({**traj, "action": actor(traj).sample()}).mode()[:-1]
        baseline = tf.zeros_like(ret)
        return ret, baseline

    def train(self, traj, actor):
        metrics = {}
        reward = self.rewfn(traj)
        target = tf.stop_gradient(self.target(traj, actor, reward))
        with tf.GradientTape() as tape:
            dist = self.net({k: v[:-1] for k, v in traj.items()})
            loss = -(dist.log_prob(target) * traj["weight"][:-1]).mean()
        metrics.update(self.opt(tape, loss, self.net))
        metrics.update(
            {
                "imag_reward_mean": reward.mean(),
                "imag_reward_std": reward.std(),
                "imag_critic_mean": dist.mean().mean(),
                "imag_critic_std": dist.mean().std(),
                "imag_target_mean": target.mean(),
                "imag_target_std": target.std(),
            }
        )
        self.update_slow()
        return metrics

    def target(self, traj, actor, reward):
        if len(reward) != len(traj["action"]) - 1:
            raise AssertionError("Should provide rewards for all but last action.")
        cont = traj["cont"][1:]
        disc = cont * self.config.discount
        action = actor(traj).sample()
        value = self.target_net({**traj, "action": action}).mean()
        if self.config.pengs_qlambda:
            vals = [value[-1]]
            interm = reward + disc * value[1:] * (1 - self.config.return_lambda)
            for t in reversed(range(len(disc))):
                vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
            tar = tf.stack(list(reversed(vals))[:-1])
            return tar
        else:
            return reward + disc * value[1:]

    def update_slow(self):
        if not self.config.slow_target:
            return
        assert self.net.variables
        initialize = self.updates == -1
        if initialize or self.updates >= self.config.slow_target_update:
            self.updates.assign(0)
            mix = 1.0 if initialize else self.config.slow_target_fraction
            for s, d in zip(self.net.variables, self.target_net.variables):
                d.assign(mix * s + (1 - mix) * d)
        self.updates.assign_add(1)


class TwinQFunction(tfutils.Module):
    def __init__(self, rewfn, config):
        assert config.actor_grad_disc == "backprop"
        assert config.actor_grad_cont == "backprop"
        assert "action" in config.actor.inputs
        self.rewfn = rewfn
        self.config = config
        self.net1 = nets.MLP((), **self.config.critic)
        self.net2 = nets.MLP((), **self.config.critic)
        if self.config.slow_target:
            self.target_net1 = nets.MLP((), **self.config.critic)
            self.target_net2 = nets.MLP((), **self.config.critic)
            self.updates = tf.Variable(-1, dtype=tf.int64)
        else:
            self.target_net1 = self.net1
            self.target_net2 = self.net2
        self.opt = tfutils.Optimizer("critic", **self.config.critic_opt)

    def score(self, traj, actor):
        traj = tf.nest.map_structure(tf.stop_gradient, traj)
        inps = {**traj, "action": actor(traj).sample()}
        ret = tf.math.reduce_min([self.net1(inps).mode(), self.net2(inps).mode()], 0)[
            :-1
        ]
        baseline = tf.zeros_like(ret)
        return ret, baseline

    def train(self, traj, actor):
        metrics = {}
        reward = self.rewfn(traj)
        target = tf.stop_gradient(self.target(traj, actor, reward))
        inps = {k: v[:-1] for k, v in traj.items()}
        with tf.GradientTape() as tape:
            dist1 = self.net1(inps)
            dist2 = self.net2(inps)
            loss1 = -(dist1.log_prob(target) * traj["weight"][:-1]).mean()
            loss2 = -(dist2.log_prob(target) * traj["weight"][:-1]).mean()
            loss = loss1 + loss2
        metrics.update(self.opt(tape, loss, [self.net1, self.net2]))
        metrics.update(
            {
                "imag_reward_mean": reward.mean(),
                "imag_reward_std": reward.std(),
                "imag_critic_mean": dist1.mean().mean(),
                "imag_critic_std": dist2.mean().std(),
                "imag_target_mean": target.mean(),
                "imag_target_std": target.std(),
            }
        )
        self.update_slow()
        return metrics

    def target(self, traj, actor, reward):
        if len(reward) != len(traj["action"]) - 1:
            raise AssertionError("Should provide rewards for all but last action.")
        cont = traj["cont"][1:]
        disc = cont * self.config.discount
        value = tf.math.reduce_min(
            [
                self.target_net1({**traj, "action": actor(traj).sample()}).mean(),
                self.target_net2({**traj, "action": actor(traj).sample()}).mean(),
            ],
            0,
        )
        if self.config.pengs_qlambda:
            vals = [value[-1]]
            interm = reward + disc * value[1:] * (1 - self.config.return_lambda)
            for t in reversed(range(len(disc))):
                vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
            tar = tf.stack(list(reversed(vals))[:-1])
            return tar
        else:
            return reward + disc * value[1:]

    def update_slow(self):
        if not self.config.slow_target:
            return
        assert self.net1.variables
        assert self.net2.variables
        initialize = self.updates == -1
        if initialize or self.updates >= self.config.slow_target_update:
            self.updates.assign(0)
            mix = 1.0 if initialize else self.config.slow_target_fraction
            for s, d in zip(self.net1.variables, self.target_net2.variables):
                d.assign(mix * s + (1 - mix) * d)
            for s, d in zip(self.net2.variables, self.target_net2.variables):
                d.assign(mix * s + (1 - mix) * d)
        self.updates.assign_add(1)
