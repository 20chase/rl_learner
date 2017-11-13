import numpy as np
import tensorflow as tf

from collections import OrderedDict
from sklearn.utils import shuffle


class ProximalPolicyGradient(object):
    def __init__(self, agent, session):
        self.sess = session
        self.agent = agent
        self._set_parameter(agent)
        self._build_ph()
        self._actor_learning()
        self._critic_learning()

    def _set_parameter(self):
        self.time_step = 0
        self.means = self.agent.means
        self.log_vars = self.agent.log_vars
        self.value = self.agent.value
        # dim
        self.obs_dim = self.agent.obs_dim
        self.act_dim = self.agent.act_dim
        # actor 
        self.eta = 50
        self.actor_lr = 3e-4
        self.kl_targ = 0.003
        self.beta = 1
        self.lr_multiplier = 1.0
        self.actor_epochs = 20
        # critic
        self.critic_lr = 1e-3
        self.replay_buffer_x = None
        self.replay_buffer_y = None
        self.critic_epochs = 10
        # flag
        self.save_network = False

    def _build_ph(self):
        self.obs_ph = self.agent.obs_ph
        self.act_ph = tf.placeholder(tf.float32, [None, self.act_dim], 'act_ph')
        self.adv_ph = tf.placeholder(tf.float32, [None, ], 'adv_ph')
        self.ret_ph = tf.placeholder(tf.float32, [None, ], 'ret_ph')

        self.lr_ph = tf.placeholder(tf.float32, name='lr_ph')
        self.beta_ph = tf.placeholder(tf.float32, name='beta_ph')

        self.old_log_vars_ph = tf.placeholder(tf.float32, [self.act_dim, ],
                                              'old_log_vars')
        self.old_means_ph = tf.placeholder(tf.float32, [None, self.act_dim],
                                           'old_means')

    def _actor_learning(self):
        # logprob
        self.logp = -0.5 * tf.reduce_sum(self.log_vars) + -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.means) / tf.exp(self.log_vars), axis=1)
        self.logp_old = -0.5 * tf.reduce_sum(self.old_log_vars_ph) + -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.old_means_ph) / tf.exp(self.old_log_vars_ph), axis=1)

        with tf.variable_scope('kl'):
            self.kl = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.exp(self.old_log_vars_ph - self.log_vars)) + 
                tf.reduce_sum(tf.square(self.means - self.old_means_ph) / tf.exp(self.log_vars), axis=1) -
                self.act_dim +
                tf.reduce_sum(self.log_vars) - tf.reduce_sum(self.old_log_vars_ph))

        with tf.variable_scope('entropy'):
            self.entropy = 0.5 * (self.act_dim * (np.log(2 * np.pi) + 1) +
                                  tf.reduce_sum(self.log_vars))

        with tf.variable_scope('actor_loss'):
            loss1 = -tf.reduce_mean(self.adv_ph * tf.exp(self.logp - self.logp_old))
            loss2 = tf.reduce_mean(self.beta_ph * self.kl)
            loss3 = self.eta * tf.square(tf.maximum(0.0, self.kl - 2.0 * self.kl_targ))
            self.actor_loss = loss1 + loss2 + loss3

        self.actor_opt = tf.train.AdamOptimizer(self.lr_ph).minimize(self.actor_loss)

    def _critic_learning(self):
        with tf.variable_scope('critic_loss'):
            self.critic_loss = tf.reduce_mean(tf.square(tf.squeeze(self.value) - self.ret_ph))
        self.critic_opt = tf.train.AdamOptimizer(self.critic_lr).minimize(self.critic_loss)

    def update(self, obs, acts, advs, rets, score, mean_step):
        self.time_step += 1
        feed_dict = self.update_actor(obs, acts, advs)
        feed_dict[self.ret_ph] = rets
        self.update_critic(obs, rets)

        stats = self._visualize_stats(feed_dict, score, mean_step)

        if self.save_network and self.time_step % 10 == 0.:
            self.agent.save_network(self.agent.model_name)

        return stats

    def update_actor(self, obs, acts, advs):
        feed_dict = {
        self.obs_ph: obs,
        self.act_ph: acts,
        self.adv_ph: advs,
        self.ret_ph: rets,
        self.beta_ph: self.beta,
        self.lr_ph: self.actor_lr * self.lr_multiplier
        }

        old_means_np, old_log_vars_np = self.sess.run([self.means, self.log_vars],
                                                      feed_dict)
        feed_dict[self.old_log_vars_ph] = old_log_vars_np
        feed_dict[self.old_means_ph] = old_means_np

        for e in range(self.actor_epochs):
            self.sess.run(self.actor_opt, feed_dict)
            kl = self.sess.run(self.kl, feed_dict)
            if kl > self.kl_targ * 4:
                break

        if kl > self.kl_targ * 2:
            self.beta = np.minimum(35, 1.5 * self.beta)
            if self.beta > 30 and self.lr_multiplier > 0.1:
                self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2.0:
            self.beta = np.maximum(1.0 / 35.0, self.beta / 1.5)
            if self.beta < (1.0 / 30.0) and self.lr_multiplier < 10:
                self.lr_multiplier *= 1.5

        return feed_dict

    def update_critic(self, x, y):
        num_batches = max(x.shape[0] // 256, 1)
        batch_size = x.shape[0] // num_batches
        if self.replay_buffer_x is None:
            x_train, y_train = x, y
        else:
            x_train = np.concatenate([x, self.replay_buffer_x])
            y_train = np.concatenate([y, self.replay_buffer_y])
        self.replay_buffer_x = x
        self.replay_buffer_y = y
        for e in range(self.critic_epochs):
            x_train, y_train = shuffle(x_train, y_train)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                obs = x_train[start:end, :]
                ret = y_train[start:end]
                feed_dict = {self.obs_ph: obs,
                             self.ret_ph: ret}
                self.sess.run(self.critic_opt, feed_dict=feed_dict)

    def _visualize_stats(self, feed_dict, score, mean_step):
        kl, entropy, actor_loss, critic_loss = self.sess.run(
            [self.kl, self.entropy, self.actor_loss, self.critic_loss],
            feed_dict)

        stats = OrderedDict()
        stats["Iteration"] = self.time_step
        stats["Score"] = score
        stats["MeanStep"] = mean_step
        stats["LearningRate"] = self.actor_lr * self.lr_multiplier
        stats["Beta"] = self.beta
        stats["KL-divergence"] = kl
        stats["Entropy"] = entropy
        stats["ActorLoss"] = actor_loss
        stats["CriticLoss"] = critic_loss

        return stats
