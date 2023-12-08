import collections
from typing import Type

import numpy as np

from .convert import convert


class Driver:
    _CONVERSION = {
        np.floating: np.float32,
        np.signedinteger: np.int32,
        np.uint8: np.uint8,
        bool: bool,
    }

    def __init__(self, env, **kwargs):
        assert len(env) > 0
        self._env = env
        self._kwargs = kwargs
        self._on_steps = []
        self._on_episodes = []
        self.reset()

    def reset(self):
        self._obs = {
            k: convert(np.zeros((len(self._env),) + v.shape, v.dtype))
            for k, v in self._env.obs_space.items()
        }
        self._obs["is_last"] = np.ones(len(self._env), bool)
        self._eps = [collections.defaultdict(list) for _ in range(len(self._env))]
        self._state = None

    def on_step(self, callback):
        self._on_steps.append(callback)

    def on_episode(self, callback):
        self._on_episodes.append(callback)

    def __call__(self, policy, steps=0, episodes=0):
        step, episode = 0, 0
        # ステップをまわす
        print("[Driver.__call__]")
        # print(f"steps: {steps}, episodes: {episodes}")
        print("Performing env steps...")
        while step < steps or episode < episodes:
            step, episode = self._step(policy, step, episode)
        print("Env steps done.")
        # print(f"step: {step}, episode: {episode}")

    def _step(self, policy, step, episode):
        # 1ステップ進める
        # print("[Driver._step]")
        # print(f"step: {step}, episode: {episode}")

        # 画像の観測と潜在変数から、方策によって行動を決定
        acts, self._state = policy(self._obs, self._state, **self._kwargs)
        acts["reset"] = np.zeros(len(self._env), bool)
        if self._obs["is_last"].any():
            acts = {
                k: v * self._expand(1 - self._obs["is_last"], len(v.shape))
                for k, v in acts.items()
            }
            acts["reset"] = self._obs["is_last"]
        # 型変換
        acts = {k: convert(v) for k, v in acts.items()}
        # print(f"acts: {acts}")
        assert all(len(x) == len(self._env) for x in acts.values()), acts

        # 環境へ行動を出力して観測を得る
        self._obs = self._env.step(acts)
        assert all(len(x) == len(self._env) for x in self._obs.values()), self._obs
        # 型変換
        self._obs = {k: convert(v) for k, v in self._obs.items()}

        # transitions (選択した行動と、その結果得られた観測)
        trns = {**self._obs, **acts}

        if self._obs["is_first"].any():
            for i, first in enumerate(self._obs["is_first"]):
                if not first:
                    continue
                # episodeの最初なら、episodeの情報をクリア
                self._eps[i].clear()

        for i in range(len(self._env)):
            # 環境ごとのtransitionを作成
            trn = {k: v[i] for k, v in trns.items()}

            # episodeの情報に追加
            [self._eps[i][k].append(v) for k, v in trn.items()]

            # on_stepのコールバック関数を実行
            # ここで
            # ・エージェントの学習
            # ・ステップ数のインクリメント
            # ・Replay Bufferへの保存
            # が実行される
            # 引数としてtransitionと環境の番号を渡す
            [fn(trn, i, **self._kwargs) for fn in self._on_steps]

            # Agentのステップを進める（not equal to Env step）
            step += 1

        # エピソードの末尾
        if self._obs["is_last"].any():
            print(trn.keys())
            for i, done in enumerate(self._obs["is_last"]):
                if not done:
                    continue
                ep = {k: convert(v) for k, v in self._eps[i].items()}

                # on_episodeのコールバック関数を実行
                # epにはobsとactの情報が入っている
                [fn(ep.copy(), i, **self._kwargs) for fn in self._on_episodes]

                # エピソード数をインクリメント
                episode += 1
        return step, episode

    def _expand(self, value, dims):
        while len(value.shape) < dims:
            value = value[..., None]
        return value
