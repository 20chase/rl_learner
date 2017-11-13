import time

import tensorflow as tf

from tabulate import tabulate


class Viewer(object):
    def __init__(self, agent, session):
        self.sess = session
        self.time_step = 0

        self._build_tensorboard()

        self.merge_all = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('./tensorboard/{}/{}'.format(
            agent.model_name,
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            ), self.sess.graph)

    def _build_tensorboard(self):
        self.score_tb = tf.placeholder(tf.float32, name='score_tb')
        self.mean_step_tb = tf.placeholder(tf.float32, name='mean_step_tb')

        self.actor_loss_tb = tf.placeholder(tf.float32, name='actor_loss_tb')
        self.critic_loss_tb = tf.placeholder(tf.float32, name='critic_loss_tb')

        self.entropy_tb = tf.placeholder(tf.float32, name='entropy_tb')
        self.kl_tb = tf.placeholder(tf.float32, name='kl_tb')

        self.lr_tb = tf.placeholder(tf.float32, name='lr_tb')
        self.beta_tb = tf.placeholder(tf.float32, name='beta_tb')

        with tf.name_scope('overall'):
            tf.summary.scalar('score', self.score_tb)
            tf.summary.scalar('mean_step', self.mean_step_tb)

        with tf.name_scope('loss'):
            tf.summary.scalar('actor_loss', self.actor_loss_tb)
            tf.summary.scalar('critic_loss', self.critic_loss_tb)

        with tf.name_scope('param'):
            tf.summary.scalar('entropy', self.entropy_tb)
            tf.summary.scalar('kl', self.kl_tb)

        with tf.name_scope('ppo_adaptive'):
            tf.summary.scalar('lr', self.lr_tb)
            tf.summary.scalar('beta', self.beta_tb)

    def show(self, stats):
        self.time_step += 1

        for k, v in stats.items():
            if k == "Score":
                feed_dict[self.score_tb] = v
                continue

            if k == "MeanStep":
                feed_dict[self.mean_step_tb] = v

            if k == "ActorLoss":
                feed_dict[self.actor_loss_tb] = v
                continue

            if k == 'CriticLoss':
                feed_dict[self.critic_loss_tb] = v
                continue

            if k == 'Entropy':
                feed_dict[self.entropy_tb] = v
                continue

            if k == 'KL-divergence':
                feed_dict[self.kl_tb] = v
                continue

            if k == 'LearningRate':
                feed_dict[self.lr_tb] = v
                continue

            if k == 'Beta':
                feed_dict[self.beta_tb] = v
                continue

        summary = self.sess.run(self.merge_all, feed_dict)
        self.writer.add_summary(summary, self.time_step)

    def print(self, stats):
        print("*********** Iteration {} ************".format(stats["Iteration"]))
        table = []
        for k, v in stats.items():
            table.append([k, v])

        print(tabulate(table, tablefmt="grid"))







        


