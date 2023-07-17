import collections
import re
import warnings
import sys

import embodied
from ..agents.director import agent
import numpy as np


def train(
    agent: agent.Agent,
    env: embodied.Env,
    replay: embodied.Replay,
    logger: embodied.Logger,
    args: embodied.Config,
):
    print("[run.train.train]")
    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    print("Logdir", logdir)
    should_train = embodied.when.Every(args.train_every)
    should_log = embodied.when.Every(args.log_every)
    should_expl = embodied.when.Until(args.expl_until)
    should_video = embodied.when.Every(args.eval_every)
    step = logger.step

    timer = embodied.Timer()
    timer.wrap("agent", agent, ["policy", "train", "report", "save"])
    timer.wrap("env", env, ["step"])
    if hasattr(replay, "_sample"):
        timer.wrap("replay", replay, ["_sample"])

    nonzeros = set()

    def per_episode(ep):
        # 1エピソード終了時の処理

        print("[run.train.train.per_episode]")
        metrics = {}
        # エピソードの長さ
        length = len(ep["reward"]) - 1
        # エピソードの報酬の合計
        score = float(ep["reward"].astype(np.float64).sum())
        print(f"Episode has {length} steps and return {score:.1f}.")
        # metricsに保存
        metrics["length"] = length
        metrics["score"] = score
        #
        metrics["reward_rate"] = (ep["reward"] - ep["reward"].min() >= 0.1).mean()
        logs = {}
        for key, value in ep.items():
            if not args.log_zeros and key not in nonzeros and (value == 0).all():
                continue
            nonzeros.add(key)
            if re.match(args.log_keys_sum, key):
                logs[f"sum_{key}"] = ep[key].sum()
            if re.match(args.log_keys_mean, key):
                logs[f"mean_{key}"] = ep[key].mean()
            if re.match(args.log_keys_max, key):
                logs[f"max_{key}"] = ep[key].max(0).mean()
        if should_video(step):
            for key in args.log_keys_video:
                metrics[f"policy_{key}"] = ep[key]
        logger.add(metrics, prefix="episode")
        logger.add(logs, prefix="logs")
        logger.add(replay.stats, prefix="replay")
        logger.write()

    # エピソードまたはステップごとに行う関数を登録する
    driver = embodied.Driver(env)
    # エピソード終了時
    driver.on_episode(lambda ep, worker: per_episode(ep))
    # ステップ数のカウント
    driver.on_step(lambda tran, _: step.increment())
    # replay bufferに保存
    driver.on_step(replay.add)

    print(f"Replay buffer has {len(replay)} transitions.")
    train_fill = max(0, args.train_fill - len(replay))
    if train_fill:
        print(f"Fill train dataset ({train_fill} steps).")
        random_agent = embodied.RandomAgent(env.act_space)
        driver(random_agent.policy, steps=train_fill, episodes=1)

    dataset = iter(agent.dataset(replay.dataset))
    state = [None]  # To be writable from train step function below.
    assert args.pretrain > 0  # At least one step to initialize variables.
    for _ in range(args.pretrain):
        print("Pretraining...")
        _, state[0], _ = agent.train(next(dataset), state[0])

    print("Pretraining done.")

    metrics = collections.defaultdict(list)
    batch = [None]

    def train_step(tran, worker):
        # 毎回のタイムステップごとに呼び出される
        # ここで学習を行う
        # print("[run.train.train.train_step]")

        # train_everyの回数ごとに学習を行う
        if should_train(step):
            # train_steps = 1
            for _ in range(args.train_steps):
                # Replay bufferからtrajectoryを取得
                # batch_size数 x replay_chunk数のタイムステップ分の情報が入っている
                batch[0] = next(dataset)

                print(f"Batch[0] has {len(batch[0])} elements:")
                for key, value in batch[0].items():
                    if hasattr(value, "shape"):
                        print(key, value.shape, value.dtype)
                    else:
                        print(key, value)
                print(f"State[0] has {len(state[0])} elements:")
                for key, value in state[0].items():
                    if hasattr(value, "shape"):
                        print(key, value.shape, value.dtype)
                    else:
                        print(key, value)

                # エージェントの学習（世界モデル ＋ Goal Autoencoder + Manager + Worker）
                outs, state[0], mets = agent.train(batch[0], state[0])

                # metricsに保存
                [metrics[key].append(value) for key, value in mets.items()]

                # Replay bufferの優先度を更新
                if "priority" in outs:
                    replay.prioritize(outs["key"], outs["priority"])

        # log_everyの回数ごとにログを出力する
        if should_log(step):
            with warnings.catch_warnings():  # Ignore empty slice warnings.
                warnings.simplefilter("ignore", category=RuntimeWarning)
                for name, values in metrics.items():
                    logger.scalar("train/" + name, np.nanmean(values, dtype=np.float64))
                    metrics[name].clear()
            logger.add(agent.report(batch[0]), prefix="report")
            logger.add(timer.stats(), prefix="timer")
            logger.write(fps=True)

    # 1ステップごとにtrain_stepを実行するように登録
    driver.on_step(train_step)

    checkpoint = embodied.Checkpoint(logdir / "checkpoint.pkl")
    checkpoint.step = step
    checkpoint.agent = agent
    checkpoint.replay = replay
    checkpoint.load_or_save()

    print("Start training loop.")

    # エージェントの方策
    policy = lambda *args: agent.policy(
        *args, mode="explore" if should_expl(step) else "train"
    )

    # 上限ステップ数まで学習を行う
    while step < args.steps:
        print("Step", step)
        print("driver(policy, steps=args.eval_every)")
        # eval_everyの回数だけ学習をしながら環境ステップをまわす
        driver(policy, steps=args.eval_every)
        checkpoint.save()
