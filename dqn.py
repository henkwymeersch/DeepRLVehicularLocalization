import tensorflow as tf
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from environment import normalize_state


class DQN:
    def __init__(self, params):
        self.epochs = params['n_epochs']
        self.patience = params['patience']
        self.thrhld_earlystopping = params['thrhld_earlystopping']
        self.batch_size = params['batch_size']
        self.n_actions = len(params['actions'])
        state_dim = len(params['state_def'])
        num_neurons = params['n_neurons']

        self._state = tf.placeholder(shape=[None, state_dim], dtype=tf.float32, name='state')
        self._target_q = tf.placeholder(shape=[None, len(params['actions'])], dtype=tf.float32, name='target_q')
        self._action_mask = tf.placeholder(shape=[None, len(params['actions'])], dtype=tf.float32)

        eval_c_name = ['eval_c_name', tf.GraphKeys.GLOBAL_VARIABLES]
        w1 = tf.get_variable('w1', [state_dim, num_neurons],
                             initializer=tf.contrib.layers.xavier_initializer(seed=1), collections=eval_c_name)
        b1 = tf.get_variable('b1', [1, num_neurons],
                             initializer=tf.contrib.layers.xavier_initializer(seed=1), collections=eval_c_name)
        w2 = tf.get_variable('w2', [num_neurons, num_neurons],
                             initializer=tf.contrib.layers.xavier_initializer(seed=1), collections=eval_c_name)
        b2 = tf.get_variable('b2', [1, num_neurons],
                             initializer=tf.contrib.layers.xavier_initializer(seed=1), collections=eval_c_name)
        w3 = tf.get_variable('w3', [num_neurons, num_neurons],
                             initializer=tf.contrib.layers.xavier_initializer(seed=1), collections=eval_c_name)
        b3 = tf.get_variable('b3', [1, num_neurons],
                             initializer=tf.contrib.layers.xavier_initializer(seed=1), collections=eval_c_name)
        w4 = tf.get_variable('w4', [num_neurons, len(params['actions'])],
                             initializer=tf.contrib.layers.xavier_initializer(seed=1), collections=eval_c_name)
        b4 = tf.get_variable('b4', [1, len(params['actions'])],
                             initializer=tf.contrib.layers.xavier_initializer(seed=1), collections=eval_c_name)

        n1 = tf.nn.relu(tf.add(tf.matmul(self._state, w1), b1))
        n2 = tf.nn.relu(tf.add(tf.matmul(n1, w2), b2))
        n3 = tf.nn.relu(tf.add(tf.matmul(n2, w3), b3))
        self._q = tf.add(tf.matmul(n3, w4), b4, name='q')

        self._loss = tf.losses.mean_squared_error(tf.multiply(self._target_q, self._action_mask),
                                                  tf.multiply(self._q, self._action_mask))
        self._optimizer = tf.train.AdamOptimizer(0.001).minimize(self._loss)

        if params['double_dqn']:
            tgt_c_name = ['tgt_c_name', tf.GraphKeys.GLOBAL_VARIABLES]
            w1m = tf.get_variable('w1m', [state_dim, num_neurons],
                                  initializer=tf.contrib.layers.xavier_initializer(seed=1), collections=tgt_c_name)
            b1m = tf.get_variable('b1m', [1, num_neurons],
                                  initializer=tf.contrib.layers.xavier_initializer(seed=1), collections=tgt_c_name)
            w2m = tf.get_variable('w2m', [num_neurons, num_neurons],
                                  initializer=tf.contrib.layers.xavier_initializer(seed=1), collections=tgt_c_name)
            b2m = tf.get_variable('b2m', [1, num_neurons],
                                  initializer=tf.contrib.layers.xavier_initializer(seed=1), collections=tgt_c_name)
            w3m = tf.get_variable('w3m', [num_neurons, num_neurons],
                                  initializer=tf.contrib.layers.xavier_initializer(seed=1), collections=tgt_c_name)
            b3m = tf.get_variable('b3m', [1, num_neurons],
                                  initializer=tf.contrib.layers.xavier_initializer(seed=1), collections=tgt_c_name)
            w4m = tf.get_variable('w4m', [num_neurons, len(params['actions'])],
                                  initializer=tf.contrib.layers.xavier_initializer(seed=1), collections=tgt_c_name)
            b4m = tf.get_variable('b4m', [1, len(params['actions'])],
                                  initializer=tf.contrib.layers.xavier_initializer(seed=1), collections=tgt_c_name)

            n1m = tf.nn.relu(tf.add(tf.matmul(self._state, w1m), b1m))
            n2m = tf.nn.relu(tf.add(tf.matmul(n1m, w2m), b2m))
            n3m = tf.nn.relu(tf.add(tf.matmul(n2m, w3m), b3m))
            self._q_m = tf.add(tf.matmul(n3m, w4m), b4m, name='q')
            self._loss_m = tf.losses.mean_squared_error(tf.multiply(self._target_q, self._action_mask),
                                                        tf.multiply(self._q_m, self._action_mask))
            self._optimizer_m = tf.train.AdamOptimizer(params['lr']).minimize(self._loss_m)

    def train(self, data: pd.DataFrame, sess, epsd, params, prvs_loss):
        # data_crnt_epsd = data.loc[data['epsd'] == epsd]
        n_samples = data.shape[0]
        batch_size = np.min([self.batch_size, data.shape[0]])
        n_batches = np.min([int(np.floor(data.shape[0] / batch_size)), params['num_iterations']])
        all_loss_values = np.empty(params['n_trainings_in_epsd'])
        all_idcs = list(data.index)
        sample_idcs = random.sample(range(n_samples), n_samples)
        for batch_idx in range(params['n_trainings_in_epsd']):
            selected_rows = sample_idcs[(batch_idx * batch_size): ((batch_idx + 1) * batch_size)]
            idcs_selected_rows = [all_idcs[i] for i in selected_rows]
            states = np.array(data.loc[idcs_selected_rows, params['state_def']], dtype=np.float32)
            states = normalize_state(states, params['m'], params['s'])
            q = np.reshape(np.array(data.loc[idcs_selected_rows, 'q']), (batch_size, 1))

            if params['inherit_q']:
                current_q = sess.run(self._q, feed_dict={self._state: states})
                for idx in range(batch_size):
                    current_q[idx, data.loc[selected_rows[idx], 'action']] = q[idx, 0]
                q = current_q
                mask = np.ones(q.shape)
            else:
                mask = sess.run(tf.one_hot(data.loc[idcs_selected_rows, 'action'].astype('int'), self.n_actions))
                q = np.hstack([q, q])
            _, loss_value = sess.run([self._optimizer, self._loss], feed_dict={self._target_q: q,
                                                                               self._action_mask: mask,
                                                                               self._state: states})

            all_loss_values[batch_idx] = loss_value

        return np.mean(all_loss_values)

    def predict(self, states, sess, m, s, model_reloaded=False):
        states = normalize_state(np.array(states), m, s)
        if model_reloaded:
            return sess.run('q:0', feed_dict={'state:0': states})
        else:
            return sess.run(self._q, feed_dict={self._state: states})

    def update_q(self, data: pd.DataFrame, sess, params):
        double_dqn = params['double_dqn']
        if double_dqn:
            m = np.reshape(params['m'], (1, params['m'].size))
            s = np.reshape(params['s'], (1, params['s'].size))
            states_p = np.array(data[params['state_p_def']])
            states_p = (np.array(states_p) - m) / s
            next_q = sess.run(self._q_m, feed_dict={self._state: states_p})
            next_q[np.isnan(next_q)] = 0
            next_q = np.max(next_q, axis=1)
            return data['reward_p'] + params['discounting'] * next_q
        else:
            states = (np.array(data[params['state_def']]) - params['m']) / params['s']
            current_q = sess.run(self._q, feed_dict={self._state: states})

            states_p = (np.array(data[params['state_p_def']]) - params['m']) / params['s']
            next_q = sess.run(self._q, feed_dict={self._state: states_p})
            next_q[np.isnan(next_q)] = 0
            idcs = np.array(data['action'])
            next_q = np.max(next_q, axis=1)

            current_q = np.choose(idcs.astype(int), current_q.transpose())
            return current_q * params['alpha'] + (1 - params['alpha']) * \
                   (data['reward_p'] + params['discounting'] * next_q)

    def plot_prediction(self, sess, data, params):
        plt.plot(data['q'])
        # plt.plot(data['reward_p'])
        states = np.array(data[params['state_def']])
        states = normalize_state(states, params['m'], params['s'])
        predicted_q = sess.run(self._q, {self._state: states})
        actions = data['action']
        predicted_q = [predicted_q[idx, actions[idx]] for idx in range(predicted_q.shape[0])]
        plt.plot(predicted_q)


def training_fn(data: pd.DataFrame, action, params, n_samples=2000, batch_size=128):
    n_samples = min(n_samples, data.shape[0])
    selected_rows = random.sample(range(data.shape[0]), n_samples)

    training_input = data.loc[selected_rows, params['state_def']]
    training_input = training_input.loc[data['action'] == action]
    selected_rows = training_input.index
    y = data.loc[selected_rows, 'q']
    return tf.estimator.inputs.pandas_input_fn(x=training_input, y=y, batch_size=batch_size, shuffle=True)


def prediction_fn(data: pd.DataFrame):
    return tf.estimator.inputs.pandas_input_fn(x=data, shuffle=False)


def epsilon_greedy(epsilon, q=None):
    num_predictions = len(q)
    num_actions = len(q[0])
    if np.random.rand() < epsilon:
        return np.random.randint(0, num_actions, num_predictions)
        # return 1 -  np.random.randint(0, num_actions, num_predictions) * 0
    else:
        return [np.argmax(this_q) for this_q in q]


def return_state_p(data: pd.DataFrame, params):
    state = data[params['state_p_def']].copy()
    name_mapping = dict(zip(params['state_p_def'], params['state_def']))
    return state.rename(columns=name_mapping)


