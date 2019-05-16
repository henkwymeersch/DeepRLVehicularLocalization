import tensorflow as tf
import numpy as np
from environment import calc_scnr_rwd


class PolicyGradient:
    def __init__(self, params):
        state_dim = len(params['state_def'])
        n_neurons = params['n_neurons']
        self.n_actions = len(params['actions'])
        self._state = tf.placeholder(shape=[None, state_dim], dtype=tf.float32, name='state')
        w1 = tf.get_variable('w1', [state_dim, n_neurons],
                             initializer=tf.contrib.layers.xavier_initializer(seed=1))
        b1 = tf.get_variable('b1', [1, n_neurons],
                             initializer=tf.contrib.layers.xavier_initializer(seed=1))
        w2 = tf.get_variable('w2', [n_neurons, n_neurons],
                             initializer=tf.contrib.layers.xavier_initializer(seed=1))
        b2 = tf.get_variable('b2', [1, n_neurons],
                             initializer=tf.contrib.layers.xavier_initializer(seed=1))
        w3 = tf.get_variable('w3', [n_neurons, n_neurons],
                             initializer=tf.contrib.layers.xavier_initializer(seed=1))
        b3 = tf.get_variable('b3', [1, n_neurons],
                             initializer=tf.contrib.layers.xavier_initializer(seed=1))
        w4 = tf.get_variable('w4', [n_neurons, len(params['actions'])],
                             initializer=tf.contrib.layers.xavier_initializer(seed=1))
        b4 = tf.get_variable('b4', [1, len(params['actions'])],
                             initializer=tf.zeros_initializer())

        n1 = tf.nn.relu(tf.add(tf.matmul(self._state, w1), b1))
        n2 = tf.nn.relu(tf.add(tf.matmul(n1, w2), b2))
        n3 = tf.nn.relu(tf.add(tf.matmul(n2, w3), b3))
        self._prob = tf.nn.softmax(tf.add(tf.matmul(n3, w4), b4), name='p')  # probability
        self._rwd = tf.placeholder(shape=[None, ], dtype=tf.float32, name='reward')
        self._actions = tf.placeholder(shape=[None, ], dtype=tf.int32, name='actions')

        neg_log_prob = tf.reduce_sum(-tf.multiply(tf.log(self._prob), tf.one_hot(self._actions, self.n_actions)),
                                     axis=1)
        self._loss = tf.reduce_mean(neg_log_prob * self._rwd)
        self._optimizer = tf.train.AdamOptimizer(params['lr']).minimize(self._loss)
        # self._optimizer = tf.train.GradientDescentOptimizer(params['lr']).minimize(self._loss)

    def train(self, data, params, sess, epsd, prvs_loss):
        states = (np.array(data[params['state_def']], dtype=np.float32) - params['m']) / params['s']
        # rewards = (np.array(data['reward']) - params['m_reward']) / params['s_reward']
        rewards = calc_scnr_rwd(data, params)
        # rewards = np.empty(data.shape[0])
        # for idx in range(data.shape[0]):
        #     rewards[idx] = scnr_rewards[data.loc[idx, 'scnr']]
        rewards = (rewards - params['m_reward']) / params['s_reward']
        actions = np.array(data['action'])
        _, loss_value = sess.run([self._optimizer, self._loss], feed_dict={self._state: states,
                                                                           self._rwd: rewards,
                                                                           self._actions: actions})

        # if not np.isnan(loss_value):
        #     if os.path.exists(params['saving_path']):
        #         s2t.send2trash(params['saving_path'])
        #     tf.saved_model.simple_save(sess, params['saving_path'], {'state': self._state}, {'prob': self._prob})
        return loss_value

    def choose_action(self, states, sess, params):
        states = (states - params['m']) / params['s']
        prob_weights = sess.run(self._prob, feed_dict={self._state: states})

        actions = np.empty(states.shape[0], dtype='int')
        for state_idx in range(states.shape[0]):
            actions[state_idx] = np.random.choice(range(prob_weights.shape[1]), p=prob_weights[state_idx, :])
        return actions
