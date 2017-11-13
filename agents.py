import numpy as np
import tensorflow as tf
import tensorlayer as tl


class ActorCritic(object):
    def __init__(self, model_name, session):
        self.model_name = model_name
        self.sess = session
    
        # env param
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.act_high = env.action_space.high

        # actor param
        self.init_logvars = -1.

        self.model = self._build_network()

    def _build_network(self):
        # define obs_ph
        self.obs_ph = tf.placeholder(tf.float32, [None, self.obs_dim], 'obs_ph')
        # build actor network
        hid1_size = self.obs_dim * 10  
        hid3_size = self.act_dim * 10
        hid2_size = int(np.sqrt(hid1_size * hid3_size))

        self.actor_network = tl.layers.InputLayer(self.obs_ph, name = 'actor_network_input')
        self.actor_network = tl.layers.DenseLayer(self.actor_network, n_units = hid1_size, act = tf.nn.tanh, 
            W_init = tf.random_normal_initializer(stddev=np.sqrt(1.0 / self.obs_dim)), name = 'actor_tanh1')
        self.actor_network = tl.layers.DenseLayer(self.actor_network, n_units = hid2_size, act = tf.nn.tanh,
            W_init = tf.random_normal_initializer(stddev=np.sqrt(1.0 / float(hid1_size))), name = 'actor_tanh2')
        self.actor_network = tl.layers.DenseLayer(self.actor_network, n_units = hid3_size, act = tf.nn.tanh, 
            W_init = tf.random_normal_initializer(stddev=np.sqrt(1.0 / float(hid2_size))), name = 'actor_tanh3')
        self.actor_network = tl.layers.DenseLayer(self.actor_network, n_units = self.act_dim, act = tf.nn.tanh,
            W_init = tf.random_normal_initializer(stddev=np.sqrt(1.0 / float(hid3_size))), name = 'means')

        # build critic network
        hid1_size = self.obs_dim * 10  
        hid3_size = 5  
        hid2_size = int(np.sqrt(hid1_size * hid3_size))

        self.critic_network = tl.layers.InputLayer(self.obs_ph, name = 'critic_network_input')
        self.critic_network = tl.layers.DenseLayer(self.critic_network, n_units = hid1_size, act = tf.nn.tanh, 
            W_init = tf.random_normal_initializer(stddev=np.sqrt(1.0 / self.obs_dim)), name = 'critic_tanh1')
        self.critic_network = tl.layers.DenseLayer(self.critic_network, n_units = hid2_size, act = tf.nn.tanh,
            W_init = tf.random_normal_initializer(stddev=np.sqrt(1.0 / float(hid1_size))), name = 'critic_tanh2')
        self.critic_network = tl.layers.DenseLayer(self.critic_network, n_units = hid3_size, act = tf.nn.tanh, 
            W_init = tf.random_normal_initializer(stddev=np.sqrt(1.0 / float(hid2_size))), name = 'critic_tanh3')
        self.critic_network = tl.layers.DenseLayer(self.critic_network, n_units = 1, act = tf.nn.tanh,
            W_init = tf.random_normal_initializer(stddev=np.sqrt(1.0 / float(hid3_size))), name = 'value')

        # build variance network
        logvar_speed = (10 * hid3_size) // 48
        log_vars = tf.get_variable('logvars', (logvar_speed, self.act_dim), tf.float32,
                                   tf.constant_initializer(0.0))

        self.means = self.actor_network.outputs
        self.log_vars = tf.reduce_sum(log_vars, axis=0) - self.init_logvars
        self.value = self.critic_network.outputs

        # sample action from norm distributiion
        with tf.variable_scope('sample_action'):
            self.sampled_act = (self.means +
                                tf.exp(self.log_vars / 2.0) *
                                tf.random_normal(shape=(self.act_dim,)))

        return [self.actor_network, self.critic_network]

    def get_value(self, obs):
        feed_dict = {self.obs_ph: obs}
        return self.sess.run(self.value, feed_dict)

    def get_action(self, obs, train=True):
        if train:
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

    def save_network(self, model_name):
        for i in range(len(self.model)):
            tl.files.save_npz(self.model[i].all_params,
                              name='./model/{}_{}.npz'.format(model_name, i),
                              sess=self.sess)

    def load_network(self, model_name):
        for i in range(len(self.model)):
            params = tl.files.load_npz(name='./model/{}_{}.npz'.format(model_name, i))
            tl.files.assign_params(self.sess, params, self.model[i])








        






