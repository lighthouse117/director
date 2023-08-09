import functools
import inspect
import pprint

import embodied
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from . import agent
from . import expl
from . import nets
from . import tfutils


class Hierarchy(tfutils.Module):
    def __init__(self, wm, act_space, config):
        print("[Hierarchy.__init__]")
        # wm: WorldModel
        self.wm = wm

        # extrinsic reward
        self.extr_reward = lambda traj: self.wm.heads["reward"](traj).mean()[1:]

        self.skill_space = embodied.Space(
            # onehot
            np.int32 if config.goal_encoder.dist == "onehot" else np.float32,
            config.skill_shape,  # skill_shape: [8, 8]
        )

        worker_inputs = config.worker_inputs + (config.worker_goal_input,)
        self.config = config.update({"worker_inputs": worker_inputs})

        wconfig = config.update(
            {
                "actor.inputs": self.config.worker_inputs,
                "critic.inputs": self.config.worker_inputs,
            }
        )

        # Worker
        # - critics
        # - scales
        # - act_space
        # - config
        self.worker = agent.ImagActorCritic(
            {
                "extr": agent.VFunction(lambda s: s["reward_extr"], wconfig),
                "expl": agent.VFunction(lambda s: s["reward_expl"], wconfig),
                "goal": agent.VFunction(lambda s: s["reward_goal"], wconfig),
            },
            config.worker_rews,
            act_space,
            wconfig,
        )

        mconfig = config.update(
            {
                "actor_grad_cont": "reinforce",
                "actent.target": config.manager_actent,
            }
        )

        # Manager
        self.manager = agent.ImagActorCritic(
            {
                "extr": agent.VFunction(lambda s: s["reward_extr"], mconfig),
                "expl": agent.VFunction(lambda s: s["reward_expl"], mconfig),
                "goal": agent.VFunction(lambda s: s["reward_goal"], mconfig),
            },
            config.manager_rews,
            self.skill_space,
            mconfig,
        )

        if self.config.expl_rew == "disag":
            self.expl_reward = expl.Disag(wm, act_space, config)
        elif self.config.expl_rew == "adver":
            # default
            self.expl_reward = self.elbo_reward
        else:
            raise NotImplementedError(self.config.expl_rew)

        # explorer = false
        if config.explorer:
            self.explorer = agent.ImagActorCritic(
                {
                    "expl": agent.VFunction(self.expl_reward, config),
                },
                {"expl": 1.0},
                act_space,
                config,
            )

        shape = self.skill_space.shape
        if self.skill_space.discrete:
            self.prior = tfutils.OneHotDist(tf.zeros(shape))
            self.prior = tfd.Independent(self.prior, len(shape) - 1)
        else:
            self.prior = tfd.Normal(tf.zeros(shape), tf.ones(shape))
            self.prior = tfd.Independent(self.prior, len(shape))

        self.feat = nets.Input(["deter"])
        self.goal_shape = (self.config.rssm.deter,)

        # Goal Encoder
        # skill_shape: [8, 8] (Output)
        self.enc = nets.MLP(config.skill_shape, dims="context", **config.goal_encoder)
        # Goal Decoder
        self.dec = nets.MLP(self.goal_shape, dims="context", **self.config.goal_decoder)

        self.kl = tfutils.AutoAdapt((), **self.config.encdec_kl)
        self.opt = tfutils.Optimizer("goal", **config.encdec_opt)

    def initial(self, batch_size):
        return {
            "step": tf.zeros((batch_size,), tf.int64),
            "skill": tf.zeros((batch_size,) + self.config.skill_shape, tf.float32),
            "goal": tf.zeros((batch_size,) + self.goal_shape, tf.float32),
        }

    def policy(self, latent, carry, imag=False):
        # tf.print("[Hierarchy.policy]")
        # imag = True
        duration = (
            self.config.train_skill_duration
            if imag
            else (self.config.env_skill_duration)
        )

        # for stoppping gradients
        sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)

        update = (carry["step"] % duration) == 0

        switch = lambda x, y: (
            tf.einsum("i,i...->i...", 1 - update.astype(x.dtype), x)
            + tf.einsum("i,i...->i...", update.astype(x.dtype), y)
        )

        # 現在の状態からmanagerがgoalを決定
        # print("Decide skill")
        skill = sg(switch(carry["skill"], self.manager.actor(sg(latent)).sample()))

        # 離散表現のゴールから連続の潜在状態へデコード
        new_goal = self.dec({"skill": skill, "context": self.feat(latent)}).mode()

        # TODO: 現在の観測を離散表現へエンコード
        disc_deter = self.enc(
            {"goal": latent["deter"], "context": self.feat(latent)}
        ).mode()
        # print("Encode current observation")
        # print(disc_deter)

        # manager_delta = false
        # manager_delta -> 現在状態からの差分をゴールとするようにする
        # new_goal = (
        #     self.feat(latent).astype(tf.float32) + new_goal
        #     if self.config.manager_delta
        #     else new_goal
        # )

        goal = sg(switch(carry["goal"], new_goal))

        # 現在状態とゴールとの差分
        delta = goal - self.feat(latent).astype(tf.float32)

        # Workerが行動を決定
        # print("Decide action\n")

        # TODO(done): デコード後のgoalとデコード前のskillの両方をWorkerへ入力
        # TODO: 現在の状態をデコードした離散バージョンのzも入力する

        worker_input = {
            **latent,
            "goal": goal,
            "skill": skill,
            "delta": delta,
            # "disc_deter": disc_deter,
        }

        # for key in worker_input:
        #     print(key, worker_input[key].shape)

        dist = self.worker.actor(sg(worker_input))

        # print("Action (worker actor): ", dist.sample())
        outs = {"action": dist}

        if "image" in self.wm.heads["decoder"].shapes:
            outs["log_goal"] = self.wm.heads["decoder"](
                {
                    "deter": goal,
                    "stoch": self.wm.rssm.get_stoch(goal),
                }
            )["image"].mode()

        # 次のタイムステップに持ち越す情報
        carry = {"step": carry["step"] + 1, "skill": skill, "goal": goal}

        return outs, carry

    def train(self, imagine, start, data):
        # tf.print("[Hierarchy.train]")
        success = lambda rew: (rew[-1] > 0.7).astype(tf.float32).mean()
        metrics = {}

        # No
        if self.config.expl_rew == "disag":
            tf.print("[Hierarchy.train] expl_reward.train")
            metrics.update(self.expl_reward.train(data))

        # Goal Autoencoderの学習
        if self.config.vae_replay:
            # TODO: これだけを最初に単独で回したい
            metrics.update(self.train_vae_replay(data))

        # No
        if self.config.explorer:
            tf.print("[Hierarchy.train] explorer.train")
            traj, mets = self.explorer.train(imagine, start, data)
            metrics.update({f"explorer_{k}": v for k, v in mets.items()})
            metrics.update(self.train_vae_imag(traj))
            if self.config.explorer_repeat:
                goal = self.feat(traj)[-1]
                metrics.update(self.train_worker(imagine, start, goal)[1])

        # "new"
        if self.config.jointly == "new":
            # まとめて学習
            traj, mets = self.train_jointly(imagine, start)
            metrics.update(mets)
            metrics["success_manager"] = success(traj["reward_goal"])
            if self.config.vae_imag:
                metrics.update(self.train_vae_imag(traj))

        elif self.config.jointly == "old":
            traj, mets = self.train_jointly_old(imagine, start)
            metrics.update(mets)
            metrics["success_manager"] = success(traj["reward_goal"])
            # No
            if self.config.vae_imag:
                metrics.update(self.train_vae_imag(traj))
        elif self.config.jointly == "off":
            for impl in self.config.worker_goals:
                goal = self.propose_goal(start, impl)
                traj, mets = self.train_worker(imagine, start, goal)
                metrics.update(mets)
                metrics[f"success_{impl}"] = success(traj["reward_goal"])
                if self.config.vae_imag:
                    metrics.update(self.train_vae_imag(traj))
            traj, mets = self.train_manager(imagine, start)
            metrics.update(mets)
            metrics["success_manager"] = success(traj["reward_goal"])
        else:
            raise NotImplementedError(self.config.jointly)
        # tf.print("[/Hierarchy.train]")

        return None, metrics

    def train_jointly(self, imagine, start):
        """
        RSSMでトラジェクトリを生成して、WorkerとManagerを学習
        """
        # tf.print("[Hierarchy.train_jointly]")
        start = start.copy()
        metrics = {}

        with tf.GradientTape(persistent=True) as tape:
            # imag=True
            policy = functools.partial(self.policy, imag=True)

            # 世界モデルで数タイムステップ先までのトラジェクトリを想像して生成
            traj = self.wm.imagine_carry(
                policy,
                start,
                self.config.imag_horizon,
                self.initial(len(start["is_first"])),
            )

            # print("traj (wm.imagine_carry)", traj.keys())

            # Manager用 環境から得られる報酬をWorld Modelで予測
            traj["reward_extr"] = self.extr_reward(traj)
            # print("traj[reward_extr]", traj["reward_extr"].shape)

            # Manager用 探索報酬を計算 elbo_reward(traj)
            traj["reward_expl"] = self.expl_reward(traj)
            # print("traj[reward_expl]", traj["reward_expl"].shape)

            # Worker用 ゴールとの近さに基づく報酬を計算
            traj["reward_goal"] = self.goal_reward(traj)
            # print("traj[reward_goal]", traj["reward_goal"].shape)

            # trajからdeterministicnな潜在状態を取り出して、ゴール状態との差分を計算
            traj["delta"] = traj["goal"] - self.feat(traj).astype(tf.float32)
            # print("traj[delta]", traj["delta"].shape)

            wtraj = self.split_traj(traj)
            # print("wtraj", wtraj.keys())

            # Kタイムステップとばした抽象的なtrajectory
            mtraj = self.abstract_traj(traj)
            # print("mtraj", mtraj.keys())

        # Workerの学習
        mets = self.worker.update(wtraj, tape)
        metrics.update({f"worker_{k}": v for k, v in mets.items()})

        # Managerの学習
        mets = self.manager.update(mtraj, tape)
        metrics.update({f"manager_{k}": v for k, v in mets.items()})

        return traj, metrics

    def train_jointly_old(self, imagine, start):
        tf.print("[Hierarchy.train_jointly_old]")
        start = start.copy()
        metrics = {}
        sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
        context = self.feat(start)
        with tf.GradientTape(persistent=True) as tape:
            skill = self.manager.actor(sg(start)).sample()
            goal = self.dec({"skill": skill, "context": context}).mode()
            goal = (
                self.feat(start).astype(tf.float32) + goal
                if self.config.manager_delta
                else goal
            )
            worker = lambda s: self.worker.actor(
                sg({**s, "goal": goal, "delta": goal - self.feat(s)})
            ).sample()
            traj = imagine(worker, start, self.config.imag_horizon)
            traj["goal"] = tf.repeat(goal[None], 1 + self.config.imag_horizon, 0)
            traj["skill"] = tf.repeat(skill[None], 1 + self.config.imag_horizon, 0)
            traj["reward_extr"] = self.extr_reward(traj)
            traj["reward_expl"] = self.expl_reward(traj)
            traj["reward_goal"] = self.goal_reward(traj)
            traj["delta"] = traj["goal"] - self.feat(traj).astype(tf.float32)
            wtraj = traj.copy()
            mtraj = self.abstract_traj_old(traj)
        mets = self.worker.update(wtraj, tape)
        metrics.update({f"worker_{k}": v for k, v in mets.items()})
        mets = self.manager.update(mtraj, tape)
        metrics.update({f"manager_{k}": v for k, v in mets.items()})
        return traj, metrics

    def train_manager(self, imagine, start):
        tf.print("[Hierarchy.train_manager]")
        start = start.copy()
        with tf.GradientTape(persistent=True) as tape:
            policy = functools.partial(self.policy, imag=True)
            traj = self.wm.imagine_carry(
                policy,
                start,
                self.config.imag_horizon,
                self.initial(len(start["is_first"])),
            )
            traj["reward_extr"] = self.extr_reward(traj)
            traj["reward_expl"] = self.expl_reward(traj)
            traj["reward_goal"] = self.goal_reward(traj)
            traj["delta"] = traj["goal"] - self.feat(traj).astype(tf.float32)
            mtraj = self.abstract_traj(traj)
        metrics = self.manager.update(mtraj, tape)
        metrics = {f"manager_{k}": v for k, v in metrics.items()}
        return traj, metrics

    def train_worker(self, imagine, start, goal):
        tf.print("[Hierarchy.train_worker]")
        start = start.copy()
        metrics = {}
        sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
        with tf.GradientTape(persistent=True) as tape:
            worker = lambda s: self.worker.actor(
                sg(
                    {
                        **s,
                        "goal": goal,
                        "delta": goal - self.feat(s).astype(tf.float32),
                    }
                )
            ).sample()
            traj = imagine(worker, start, self.config.imag_horizon)
            traj["goal"] = tf.repeat(goal[None], 1 + self.config.imag_horizon, 0)
            traj["reward_extr"] = self.extr_reward(traj)
            traj["reward_expl"] = self.expl_reward(traj)
            traj["reward_goal"] = self.goal_reward(traj)
            traj["delta"] = traj["goal"] - self.feat(traj).astype(tf.float32)
        mets = self.worker.update(traj, tape)
        metrics.update({f"worker_{k}": v for k, v in mets.items()})
        return traj, metrics

    def train_vae_replay(self, data):
        """
        Replay Bufferにある実際の画像からVAEを学習する
        """

        # tf.print("[Hierarchy.train_vae_replay]")
        metrics = {}
        feat = self.feat(data).astype(tf.float32)

        # inputs: [skill]
        if "context" in self.config.goal_decoder.inputs:
            if self.config.vae_span:
                context = feat[:, 0]
                goal = feat[:, -1]
            else:
                assert feat.shape[1] > self.config.train_skill_duration
                context = feat[:, : -self.config.train_skill_duration]
                goal = feat[:, self.config.train_skill_duration :]
        else:
            goal = context = feat

        with tf.GradientTape() as tape:
            # Goal Encoderを通す
            # print("Input to Goal Encoder")
            # print("goal", goal.shape)
            # print("context", context.shape)
            enc = self.enc({"goal": goal, "context": context})
            # print("Output from Goal Encoder", enc)

            dec = self.dec({"skill": enc.sample(), "context": context})
            # print("Output from Goal Decoder", dec.sample().shape)

            # 再構成のロス
            rec = -dec.log_prob(tf.stop_gradient(goal))

            # KLロス
            if self.config.goal_kl:
                kl = tfd.kl_divergence(enc, self.prior)
                kl, mets = self.kl(kl)
                metrics.update({f"goalkl_{k}": v for k, v in mets.items()})
                assert rec.shape == kl.shape, (rec.shape, kl.shape)
            else:
                kl = 0.0
            loss = (rec + kl).mean()

        # Goal encoderとGoal decoderのパラメータ更新
        metrics.update(self.opt(tape, loss, [self.enc, self.dec]))

        metrics["goalrec_mean"] = rec.mean()
        metrics["goalrec_std"] = rec.std()
        return metrics

    def train_vae_imag(self, traj):
        """
        世界モデルで生成したTrajectoryからVAEを学習する
        """
        tf.print("[Hierarchy.train_vae_imag]")
        metrics = {}
        feat = self.feat(traj).astype(tf.float32)
        if "context" in self.config.goal_decoder.inputs:
            if self.config.vae_span:
                context = feat[0]
                goal = feat[-1]
            else:
                assert feat.shape[0] > self.config.train_skill_duration
                context = feat[: -self.config.train_skill_duration]
                goal = feat[self.config.train_skill_duration :]
        else:
            goal = context = feat
        with tf.GradientTape() as tape:
            enc = self.enc({"goal": goal, "context": context})
            dec = self.dec({"skill": enc.sample(), "context": context})
            rec = -dec.log_prob(tf.stop_gradient(goal.astype(tf.float32)))
            if self.config.goal_kl:
                kl = tfd.kl_divergence(enc, self.prior)
                kl, mets = self.kl(kl)
                metrics.update({f"goalkl_{k}": v for k, v in mets.items()})
            else:
                kl = 0.0
            loss = (rec + kl).mean()
        metrics.update(self.opt(tape, loss, [self.enc, self.dec]))
        metrics["goalrec_mean"] = rec.mean()
        metrics["goalrec_std"] = rec.std()
        return metrics

    def propose_goal(self, start, impl):
        """
        ログ用にデコードされたゴールを生成
        """
        feat = self.feat(start).astype(tf.float32)
        if impl == "replay":
            # 状態をランダムに選択
            target = tf.random.shuffle(feat).astype(tf.float32)

            # ランダムに選んだ潜在表現を離散ベクトルにエンコード
            skill = self.enc({"goal": target, "context": feat}).sample()

            # 離散ベクトルを連続ベクトルへデコード
            return self.dec({"skill": skill, "context": feat}).mode()

        if impl == "replay_direct":
            # 実際の状態表現をランダムに選択
            return tf.random.shuffle(feat).astype(tf.float32)

        if impl == "manager":
            # Managerがskillを決定
            skill = self.manager.actor(start).sample()

            # 離散表現のskillからゴールの連続ベクトルへデコード
            goal = self.dec({"skill": skill, "context": feat}).mode()

            # goal = feat + goal if self.config.manager_delta else goal
            return goal

        if impl == "prior":
            # ランダムに離散ベクトルをサンプリング
            skill = self.prior.sample(len(start["is_terminal"]))

            # 連続ベクトルにデコードしてみる
            return self.dec({"skill": skill, "context": feat}).mode()

        raise NotImplementedError(impl)

    def propose_skill_and_goal(self, start, impl):
        """
        ログ用にデコード前のskillとエンコードされたゴールを生成
        """
        feat = self.feat(start).astype(tf.float32)
        if impl == "replay":
            # 状態をランダムに選択
            target = tf.random.shuffle(feat).astype(tf.float32)

            # ランダムに選んだ潜在表現を離散ベクトルにエンコード
            skill = self.enc({"goal": target, "context": feat}).sample()

            # 離散ベクトルを連続ベクトルへデコード
            return skill, self.dec({"skill": skill, "context": feat}).mode()

        if impl == "manager":
            # Managerがskillを決定
            skill = self.manager.actor(start).sample()

            # 離散表現のskillからゴールの連続ベクトルへデコード
            goal = self.dec({"skill": skill, "context": feat}).mode()

            # goal = feat + goal if self.config.manager_delta else goal
            return skill, goal

        if impl == "prior":
            # ランダムに離散ベクトルをサンプリング
            skill = self.prior.sample(len(start["is_terminal"]))

            # 連続ベクトルにデコードしてみる
            return skill, self.dec({"skill": skill, "context": feat}).mode()

        raise NotImplementedError(impl)

    def goal_reward(self, traj):
        # feat: ["deter""]
        feat = self.feat(traj).astype(tf.float32)

        goal = tf.stop_gradient(traj["goal"].astype(tf.float32))
        skill = tf.stop_gradient(traj["skill"].astype(tf.float32))
        context = tf.stop_gradient(
            tf.repeat(feat[0][None], 1 + self.config.imag_horizon, 0)
        )
        if self.config.goal_reward == "dot":
            return tf.einsum("...i,...i->...", goal, feat)[1:]
        elif self.config.goal_reward == "dir":
            return tf.einsum("...i,...i->...", tf.nn.l2_normalize(goal, -1), feat)[1:]
        elif self.config.goal_reward == "normed_inner":
            norm = tf.linalg.norm(goal, axis=-1, keepdims=True)
            return tf.einsum("...i,...i->...", goal / norm, feat / norm)[1:]
        elif self.config.goal_reward == "normed_squared":
            norm = tf.linalg.norm(goal, axis=-1, keepdims=True)
            return -((goal / norm - feat / norm) ** 2).mean(-1)[1:]
        elif self.config.goal_reward == "cosine_lower":
            gnorm = tf.linalg.norm(goal, axis=-1, keepdims=True) + 1e-12
            fnorm = tf.linalg.norm(feat, axis=-1, keepdims=True) + 1e-12
            fnorm = tf.maximum(gnorm, fnorm)
            return tf.einsum("...i,...i->...", goal / gnorm, feat / fnorm)[1:]
        elif self.config.goal_reward == "cosine_lower_pos":
            gnorm = tf.linalg.norm(goal, axis=-1, keepdims=True) + 1e-12
            fnorm = tf.linalg.norm(feat, axis=-1, keepdims=True) + 1e-12
            fnorm = tf.maximum(gnorm, fnorm)
            cos = tf.einsum("...i,...i->...", goal / gnorm, feat / fnorm)[1:]
            return tf.nn.relu(cos)
        elif self.config.goal_reward == "cosine_frac":
            gnorm = tf.linalg.norm(goal, axis=-1) + 1e-12
            fnorm = tf.linalg.norm(feat, axis=-1) + 1e-12
            goal /= gnorm[..., None]
            feat /= fnorm[..., None]
            cos = tf.einsum("...i,...i->...", goal, feat)
            mag = tf.minimum(gnorm, fnorm) / tf.maximum(gnorm, fnorm)
            return (cos * mag)[1:]
        elif self.config.goal_reward == "cosine_frac_pos":
            gnorm = tf.linalg.norm(goal, axis=-1) + 1e-12
            fnorm = tf.linalg.norm(feat, axis=-1) + 1e-12
            goal /= gnorm[..., None]
            feat /= fnorm[..., None]
            cos = tf.einsum("...i,...i->...", goal, feat)
            mag = tf.minimum(gnorm, fnorm) / tf.maximum(gnorm, fnorm)
            return tf.nn.relu(cos * mag)[1:]
        elif self.config.goal_reward == "cosine_max":
            # here
            gnorm = tf.linalg.norm(goal, axis=-1, keepdims=True) + 1e-12
            fnorm = tf.linalg.norm(feat, axis=-1, keepdims=True) + 1e-12
            norm = tf.maximum(gnorm, fnorm)
            return tf.einsum("...i,...i->...", goal / norm, feat / norm)[1:]
        elif self.config.goal_reward == "cosine_max_pos":
            gnorm = tf.linalg.norm(goal, axis=-1, keepdims=True) + 1e-12
            fnorm = tf.linalg.norm(feat, axis=-1, keepdims=True) + 1e-12
            norm = tf.maximum(gnorm, fnorm)
            cos = tf.einsum("...i,...i->...", goal / norm, feat / norm)[1:]
            return tf.nn.relu(cos)
        elif self.config.goal_reward == "normed_inner_clip":
            norm = tf.linalg.norm(goal, axis=-1, keepdims=True)
            cosine = tf.einsum("...i,...i->...", goal / norm, feat / norm)[1:]
            return tf.clip_by_value(cosine, -1.0, 1.0)
        elif self.config.goal_reward == "normed_inner_clip_pos":
            norm = tf.linalg.norm(goal, axis=-1, keepdims=True)
            cosine = tf.einsum("...i,...i->...", goal / norm, feat / norm)[1:]
            return tf.clip_by_value(cosine, 0.0, 1.0)
        elif self.config.goal_reward == "diff":
            goal = tf.nn.l2_normalize(goal[:-1], -1)
            diff = tf.concat([feat[1:] - feat[:-1]], 0)
            return tf.einsum("...i,...i->...", goal, diff)
        elif self.config.goal_reward == "norm":
            return -tf.linalg.norm(goal - feat, axis=-1)[1:]
        elif self.config.goal_reward == "squared":
            return -((goal - feat) ** 2).sum(-1)[1:]
        elif self.config.goal_reward == "epsilon":
            return ((goal - feat).mean(-1) < 1e-3).astype(tf.float32)[1:]
        elif self.config.goal_reward == "enclogprob":
            return self.enc({"goal": goal, "context": context}).log_prob(skill)[1:]
        elif self.config.goal_reward == "encprob":
            return self.enc({"goal": goal, "context": context}).prob(skill)[1:]
        elif self.config.goal_reward == "enc_normed_cos":
            dist = self.enc({"goal": goal, "context": context})
            probs = dist.distribution.probs_parameter()
            norm = tf.linalg.norm(probs, axis=[-2, -1], keepdims=True)
            return tf.einsum("...ij,...ij->...", probs / norm, skill / norm)[1:]
        elif self.config.goal_reward == "enc_normed_squared":
            dist = self.enc({"goal": goal, "context": context})
            probs = dist.distribution.probs_parameter()
            norm = tf.linalg.norm(probs, axis=[-2, -1], keepdims=True)
            return -((probs / norm - skill / norm) ** 2).mean([-2, -1])[1:]
        else:
            raise NotImplementedError(self.config.goal_reward)

    def elbo_reward(self, traj):
        feat = self.feat(traj).astype(tf.float32)
        context = tf.repeat(feat[0][None], 1 + self.config.imag_horizon, 0)
        enc = self.enc({"goal": feat, "context": context})
        dec = self.dec({"skill": enc.sample(), "context": context})
        ll = dec.log_prob(feat)
        kl = tfd.kl_divergence(enc, self.prior)
        if self.config.adver_impl == "abs":
            return tf.abs(dec.mode() - feat).mean(-1)[1:]
        elif self.config.adver_impl == "squared":
            # default
            return ((dec.mode() - feat) ** 2).mean(-1)[1:]
        elif self.config.adver_impl == "elbo_scaled":
            return (kl - ll / self.kl.scale())[1:]
        elif self.config.adver_impl == "elbo_unscaled":
            return (kl - ll)[1:]
        raise NotImplementedError(self.config.adver_impl)

    def split_traj(self, traj):
        """
        Worker用にTrajectoryを分割
        """
        traj = traj.copy()
        k = self.config.train_skill_duration
        assert len(traj["action"]) % k == 1
        reshape = lambda x: x.reshape([x.shape[0] // k, k] + x.shape[1:])
        for key, val in list(traj.items()):
            val = tf.concat([0 * val[:1], val], 0) if "reward" in key else val
            # (1 2 3 4 5 6 7 8 9 10) -> ((1 2 3 4) (4 5 6 7) (7 8 9 10))
            val = tf.concat([reshape(val[:-1]), val[k::k][:, None]], 1)
            # N val K val B val F... -> K val (N B) val F...
            val = val.transpose([1, 0] + list(range(2, len(val.shape))))
            val = val.reshape([val.shape[0], np.prod(val.shape[1:3])] + val.shape[3:])
            val = val[1:] if "reward" in key else val
            traj[key] = val
        # Bootstrap sub trajectory against current not next goal.
        traj["goal"] = tf.concat([traj["goal"][:-1], traj["goal"][:1]], 0)
        traj["weight"] = (
            tf.math.cumprod(self.config.discount * traj["cont"]) / self.config.discount
        )
        return traj

    def abstract_traj(self, traj):
        """
        Manager用にTrajectoryを抽象化
        """
        traj = traj.copy()
        traj["action"] = traj.pop("skill")
        k = self.config.train_skill_duration
        reshape = lambda x: x.reshape([x.shape[0] // k, k] + x.shape[1:])
        weights = tf.math.cumprod(reshape(traj["cont"][:-1]), 1)
        for key, value in list(traj.items()):
            if "reward" in key:
                traj[key] = (reshape(value) * weights).mean(1)
            elif key == "cont":
                traj[key] = tf.concat([value[:1], reshape(value[1:]).prod(1)], 0)
            else:
                traj[key] = tf.concat([reshape(value[:-1])[:, 0], value[-1:]], 0)
        traj["weight"] = (
            tf.math.cumprod(self.config.discount * traj["cont"]) / self.config.discount
        )
        return traj

    def abstract_traj_old(self, traj):
        traj = traj.copy()
        traj["action"] = traj.pop("skill")
        mult = tf.math.cumprod(traj["cont"][1:], 0)
        for key, value in list(traj.items()):
            if "reward" in key:
                traj[key] = (mult * value).mean(0)[None]
            elif key == "cont":
                traj[key] = tf.stack([value[0], value[1:].prod(0)], 0)
            else:
                traj[key] = tf.stack([value[0], value[-1]], 0)
        return traj

    def report(self, data):
        metrics = {}
        for impl in ("manager", "prior", "replay"):
            for key, video in self.report_worker(data, impl).items():
                metrics[f"impl_{impl}_{key}"] = video
        return metrics

    def report_worker(self, data, impl):
        # Prepare initial state.
        decoder = self.wm.heads["decoder"]
        states, _ = self.wm.rssm.observe(
            self.wm.encoder(data)[:6], data["action"][:6], data["is_first"][:6]
        )
        start = {k: v[:, 4] for k, v in states.items()}
        start["is_terminal"] = data["is_terminal"][:6, 4]

        # ゴール（連続表現）を決定
        # goal = self.propose_goal(start, impl)

        # ゴール（デコード前の離散表現 + デコード後の連続表現）を決定
        skill, goal = self.propose_skill_and_goal(start, impl)

        # Worker rollout.
        worker = lambda s: self.worker.actor(
            {
                **s,
                "goal": goal,
                "skill": skill,
                "delta": goal - self.feat(s).astype(tf.float32),
            }
        ).sample()

        traj = self.wm.imagine(worker, start, self.config.worker_report_horizon)
        # Decoder into images.
        initial = decoder(start)
        target = decoder({"deter": goal, "stoch": self.wm.rssm.get_stoch(goal)})
        rollout = decoder(traj)
        # Stich together into videos.
        videos = {}
        for k in rollout.keys():
            if k not in decoder.cnn_shapes:
                continue
            length = 1 + self.config.worker_report_horizon
            rows = []
            rows.append(tf.repeat(initial[k].mode()[:, None], length, 1))
            if target is not None:
                rows.append(tf.repeat(target[k].mode()[:, None], length, 1))
            rows.append(rollout[k].mode().transpose((1, 0, 2, 3, 4)))
            videos[k] = tfutils.video_grid(tf.concat(rows, 2))
        return videos
