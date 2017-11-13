import gym
import roboschool
import scipy.signal

import numpy as np
import tensorflow as tf


class PPORunner(object):
    def __init__(self, env, agent, algs, viewer):
        self.env = env
        self.agent = agent
        self.algs = algs
        self.viewer = viewer

        self.render = False
        self.gamma = 0.995
        self.lamb = 0.98
        self.batch_size = 20
        self.max_episodes = 100000

    def _run_episode(self):
        state = self.env.reset()
        obs, acts, rews = [], [], []
        done = False
        while not done:
            if self.render:
                self.env.render()
            obs.append(state)
            action = self.agent.get_action(state).reshape((1, -1)).astype(np.float32)
            acts.append(action)
            state, reward, done, info = self.env.step(np.squeeze(action, axis=0))
            rews.append(reward)

        return np.asarray(obs), np.asarray(acts), np.asarray(rews)

    def _run_policy(self):
        trajectories = []
        for e in range(self.batch_size):
            obs, acts, rews = run_episode(self.env, self.agent)
            acts = np.reshape(acts, (len(rews), self.agent.act_dim))
            trajectory = {
            'obs': obs,
            'acts': acts,
            'rewards': rewards
            }
            trajectories.append(trajectory)

        mean_step = np.mean([len(t['rewards']) for t in trajectories])
        score = np.mean([t['rewards'].sum() for t in trajectories])

        return trajectories, score, mean_step

    def learn(self):
        e = 0
        while e < self.max_episodes:
            self.trajectories, score, mean_step = self._run_policy()
            e += self.trajectories
            obs, acts, advs, rets = self._process_traj()
            stats = self.algs.update(obs, acts, advs, rets, score, mean_step)
            self.viewer.print(stats)
            self.viewer.show(stats)

    def _process_traj(self):
        self._add_value()
        self._add_disc_sum_rew()
        self._add_gae()
        return self._build_train_set()

    def discount(self, x):
        return scipy.signal.lfilter([1.0], [1.0, -self.gamma], x[::-1])[::-1]

    def _add_disc_sum_rew(self):
        for trajectory in self.trajectories:
            rewards = trajectory['rewards'] * (1 - self.gamma)
            disc_sum_rew = self.discount(rewards, self.gamma)
            trajectory['disc_sum_rew'] = disc_sum_rew

    def _add_value(self):
        for trajectory in self.trajectories:
            obs = trajectory['obs']
            values = self.agent.get_value(obs)
            trajectory['values'] = np.squeeze(np.asarray(values))

    def _add_gae(self):
        for trajectory in self.trajectories:
            rewards = trajectory['rewards'] * (1 - self.gamma)
            values = trajectory['values']
            # temporal differences
            tds = rewards - values + np.append(values[1:] * self.gamma, 0)
            advantages = self.discount(tds, self.gamma * self.lamb)
            advs = np.asarray(advantages)
            trajectory['advs'] = advs

    def _build_train_set(self):
        observes = np.concatenate([t['obs'] for t in self.trajectories])
        actions = np.concatenate([t['acts'] for t in self.trajectories])
        disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in self.trajectories])
        advantages = np.concatenate([t['advs'] for t in self.trajectories])
        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        return observes, actions, advantages, disc_sum_rew


