import collections
import re
import warnings
import sys
import tensorflow as tf
import numpy as np

import embodied
from ..agents.director import agent


def eval(
    agent: agent.Agent,
    env: embodied.Env,
    train_replay: embodied.Replay,
    eval_replay: embodied.Replay,
    logger: embodied.Logger,
    args: embodied.Config,
):
    print("eval()")

    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    step = logger.step

    def per_episode(ep):
        metrics = {}
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())

        print(f"Episode has {length} steps and return {score:.1f}.")

        metrics["length"] = length
        metrics["score"] = score

    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, worker: per_episode(ep))
    driver.on_step(lambda tran, _: step.increment())
    driver.on_step(train_replay.add)

    # print(f"Replay buffer has {len(train_replay)} transitions.")
    # train_fill = max(0, args.train_fill - len(train_replay))
    # if train_fill:
    #     print(f"Fill train dataset ({train_fill} steps).")
    #     random_agent = embodied.RandomAgent(env.act_space)
    #     driver(random_agent.policy, steps=train_fill, episodes=1)

    dataset = iter(agent.dataset(train_replay.dataset))
    state = [None]  # To be writable from train step function below.
    assert args.pretrain > 0  # At least one step to initialize variables.
    for _ in range(args.pretrain):
        print("Pretraining...")
        _, state[0], _ = agent.train(next(dataset), state[0])

    print("Pretraining done.")

    def train_step(tran, worker):
        reward = tran["reward"]
        if reward != 0:
            print(f"Step {step.value}: Got reward {reward}.")

    # 1ステップごとにtrain_stepを実行するように登録
    driver.on_step(train_step)

    checkpoint = embodied.Checkpoint(logdir / "checkpoint.pkl")
    checkpoint.step = step
    checkpoint.agent = agent
    checkpoint.train_replay = train_replay
    checkpoint.eval_replay = eval_replay

    values = tf.nest.map_structure(lambda x: x.numpy(), agent.variables)
    amount = len(tf.nest.flatten(values))
    count = int(sum(np.prod(x.shape) for x in tf.nest.flatten(values)))
    print(f"Model has {amount} tensors and {count} parameters.")

    print("Loading checkpoint...")
    checkpoint.load_or_save()

    print("Start training loop.")

    # エージェントの方策
    policy = lambda *args: agent.policy(*args, mode="train")

    driver(policy, episodes=1)
