import numpy as np
import tensorflow as tf
import tensorlayer as tl


class ActorCritic(object):
    def __init__(self, env):
        self.sess = tf.session()
        self._init_param(env)
        self._build_ph()

        self.actor_model, self.means, self.logvars = self._build_actor()
        self.critic_model, self.value = self._build_critic()

    def _init_param(self, env):
        # env param
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.act_high = env.action_space.high

        # actor param
        self.init_logvars = -1.

    def _build_ph(self):
        self.obs_ph = tf.placeholder(tf.float32, [None, self.obs_dim], 'obs_ph')

    def _build_actor(self):
        # build actor network
        hid1_size = self.obs_dim * 10  
        hid3_size = self.act_dim * 10
        hid2_size = int(np.sqrt(hid1_size * hid3_size))

        network = tl.layers.InputLayer(self.obs_ph, name='actor_input')
        network = tl.layers.DenseLayer(network, 
            n_units=hid1_size, 
            act=tf.nn.tanh,
            W_init = tf.random_normal_initializer(stddev=np.sqrt(1.0 / self.obs_dim)),
            name='actor_tanh1')
        network = tl.layers.DenseLayer(network, 
            n_units=hid2_size, 
            act=tf.nn.tanh,
            W_init = tf.random_normal_initializer(stddev=np.sqrt(1.0 / hid1_size)),
            name='actor_tanh2')
        network = tl.layers.DenseLayer(network, 
            n_units=hid3_size, 
            act=tf.nn.tanh,
            W_init = tf.random_normal_initializer(stddev=np.sqrt(1.0 / hid2_size)),
            name='actor_tanh3')
        network = tl.layers.DenseLayer(network, 
            n_units=self.act_dim, 
            act=tf.nn.tanh,
            W_init = tf.random_normal_initializer(stddev=np.sqrt(1.0 / hid3_size)),
            name='actor_means')

        logvar_speed = (10 * hid3_size) // 48
        with tf.variable_scope('actor_logvars'):
            logvars_variable = tf.get_variable('logvars', (logvar_speed, self.act_dim), tf.float32,
                                       tf.constant_initializer(0.0))

        means = network.outputs
        logvars = tf.reduce_sum(logvars_variable, axis=0) - self.init_logvars

        with tf.variable_scope('sample_action'):
            self.sampled_act = (means +
                                tf.exp(logvars / 2.0) *
                                tf.random_normal(shape=(self.act_dim,)))

        return [network, logvars_variable], means, logvars

    def _build_critic(self):
        hid1_size = self.obs_dim * 10  
        hid3_size = 5  
        hid2_size = int(np.sqrt(hid1_size * hid3_size))

        network = tl.layers.InputLayer(self.obs_ph, name='critic_input')
        network = tl.layers.DenseLayer(network, 
            n_units=hid1_size, 
            act=tf.nn.tanh,
            W_init = tf.random_normal_initializer(stddev=np.sqrt(1.0 / self.obs_dim)),
            name='critic_tanh1')
        network = tl.layers.DenseLayer(network, 
            n_units=hid2_size, 
            act=tf.nn.tanh,
            W_init = tf.random_normal_initializer(stddev=np.sqrt(1.0 / hid1_size)),
            name='critic_tanh2')
        network = tl.layers.DenseLayer(network, 
            n_units=hid3_size, 
            act=tf.nn.tanh,
            W_init = tf.random_normal_initializer(stddev=np.sqrt(1.0 / hid2_size)),
            name='critic_tanh3')
        network = tl.layers.DenseLayer(network,
            n_units=1,
            W_init = tf.random_normal_initializer(stddev=np.sqrt(1.0 / hid3_size)),
            name='value')

        value = network.outputs

        return network, value

    def get_value(self, obs):
        feed_dict = {self.obs_ph: obs}
        return self.sess.run(self.value, feed_dict)

    def get_action(self, obs, train=False):
        if train == 0:
            action = self.sample(obs)
        else:
            action = self.determine(obs)

        return self.convert_action(action)

    def sample(self, obs):
        feed_dict = {self.obs_ph: obs}
        return self.sess.run(self.sampled_act, feed_dict=feed_dict)

    def determine(self, obs):
        feed_dict = {self.obs_ph: obs}
        return self.sess.run(self.means, feed_dict)

    def convert_action(self, action):
        return action * self.act_high







        






