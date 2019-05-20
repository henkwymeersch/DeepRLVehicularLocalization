import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import image


class Scenario:
    def __init__(self, num_nds, num_lanes, params, grid=True):
        fim_gps = params['fim_gps']
        fim_gps_master = params['fim_gps_master']
        pos_var = params['pos_var']

        num_nds_p_lane = [round(num_nds / num_lanes)] * num_lanes
        num_nds_p_lane[-1] = num_nds - np.sum(num_nds_p_lane[:-1])
        if grid:
            self.nds = list()
            for lane_idx in range(num_lanes):
                for nd_idx in range(int(num_nds_p_lane[lane_idx])):
                    # position: x, y, orientation
                    position = np.array([nd_idx * params['xlim'] / (num_nds_p_lane[lane_idx] - 1)
                                         + pos_var * np.random.randn(),
                                         (lane_idx + 0.5) * params['lane_width'],
                                         0])
                    nd = {'position': position,
                          'prior_fim': fim_gps,
                          'adj': list()}
                    self.nds.append(nd)
        else:
            # randomly distributed nodes to be implemented.
            raise NotImplementedError

        # define sets of adj
        self.links = list()
        for nd_idx1 in range(len(self.nds)):
            for nd_idx2 in range(len(self.nds)):
                if nd_idx1 != nd_idx2:
                    if in_fov(self.nds[nd_idx1], self.nds[nd_idx2], params):
                        self.nds[nd_idx1]['adj'].append(nd_idx2)
                        if [nd_idx2, nd_idx1] not in self.links:
                            self.links.append([nd_idx1, nd_idx2])

        master_idx = np.random.randint(len(self.nds))
        self.nds[master_idx]['prior_fim'] = fim_gps_master
        self.master_idx = master_idx
        self.fim_gps = fim_gps
        self.fim_gps_master = fim_gps_master
        self.actions = np.zeros((num_nds, num_nds))
        self.cumulative_actions = np.zeros((num_nds, num_nds))
        self.xlim = params['xlim'] + 2
        self.ylim = params['num_lanes'] * params['lane_width'] + 2

        # Initiate covariance matrix
        self.var = np.zeros((2 * len(self.nds), 2 * len(self.nds)))
        for nds_idx in range(len(self.nds)):
            if 'master_idx' in locals() and nds_idx == master_idx:
                self.var[nds_idx * 2, nds_idx * 2] = 1 / fim_gps_master[0, 0]
                self.var[nds_idx * 2 + 1, nds_idx * 2 + 1] = 1 / fim_gps_master[1, 1]
            else:
                self.var[nds_idx * 2, nds_idx * 2] = 1 / fim_gps[0, 0]
                self.var[nds_idx * 2 + 1, nds_idx * 2 + 1] = 1 / fim_gps[1, 1]

    @property
    def initial_var(self):
        var = np.zeros((2 * len(self.nds), 2 * len(self.nds)))
        for nds_idx in range(len(self.nds)):
            if nds_idx == self.master_idx:
                var[nds_idx * 2, nds_idx * 2] = 1 / self.fim_gps_master[0, 0]
                var[nds_idx * 2 + 1, nds_idx * 2 + 1] = 1 / self.fim_gps_master[1, 1]
            else:
                var[nds_idx * 2, nds_idx * 2] = 1 / self.fim_gps[0, 0]
                var[nds_idx * 2 + 1, nds_idx * 2 + 1] = 1 / self.fim_gps[1, 1]
        return var

    def gen_state(self, nd_idx1, nd_idx2, params):
        x = self.nds[nd_idx2]['position'][0] - self.nds[nd_idx1]['position'][0]
        x_pert = np.sqrt(self.var[nd_idx2 * 2, nd_idx2 * 2] + self.var[nd_idx1 * 2, nd_idx1 * 2]) * np.random.randn()
        y = self.nds[nd_idx2]['position'][1] - self.nds[nd_idx1]['position'][1]
        y_pert = np.sqrt(self.var[nd_idx2 * 2 + 1, nd_idx2 * 2 + 1] + self.var[nd_idx1 * 2 + 1, nd_idx1 * 2 + 1]) \
                 * np.random.randn()
        delta_x = x + x_pert
        delta_y = y + y_pert

        var1x = self.var[nd_idx1 * 2, nd_idx1 * 2]
        var1y = self.var[nd_idx1 * 2 + 1, nd_idx1 * 2 + 1]
        var2x = self.var[nd_idx2 * 2, nd_idx2 * 2]
        var2y = self.var[nd_idx2 * 2 + 1, nd_idx2 * 2 + 1]
        varxx = self.var[nd_idx1 * 2, nd_idx2 * 2]
        varyy = self.var[nd_idx1 * 2 + 1, nd_idx2 * 2 + 1]

        # add covariance and a scalar to indicate neighbors
        # n_unfinished_nds = len(self.dependencies[nd_idx1]) + len(self.dependencies[nd_idx2])
        adj_nds = set([d['nd'] for d in self.dependencies[nd_idx1]] +
                                   [d['nd'] for d in self.dependencies[nd_idx2]])
        n_unfinished_nds = 0
        for nd_idx in adj_nds:
            if self.pebs[nd_idx] > params['objective_peb']:
                n_unfinished_nds += 1

        return delta_x, delta_y, var1x, var1y, var2x, var2y, varxx, varyy, n_unfinished_nds

    def update_var(self, action, nd_idx1, nd_idx2, params):
        min_var = 1e-5
        if action != 0:
            rltv_pos = self.nds[int(nd_idx1)]['position'][:2] - self.nds[int(nd_idx2)]['position'][:2]
            l = np.linalg.norm(rltv_pos)
            jacobian = np.array([[rltv_pos[0] / l, rltv_pos[1] / l,
                                  -rltv_pos[0] / l, -rltv_pos[1] / l],
                                 [-rltv_pos[1] / l ** 2, rltv_pos[0] / l ** 2,
                                  rltv_pos[1] / l ** 2, -rltv_pos[0] / l ** 2]])
            indices_c = np.array([nd_idx1 * 2, nd_idx1 * 2 + 1, nd_idx2 * 2, nd_idx2 * 2 + 1], dtype=np.int8)
            c = self.var[np.reshape(indices_c, (4, 1)), indices_c]
            sigma_l = params['sigma_l']
            sigma_theta = params['sigma_alpha']
            sigma = np.diag([sigma_l ** 2, sigma_theta ** 2])
            # Kalman gain
            k = c.dot(jacobian.transpose()).dot(np.linalg.pinv((sigma + jacobian.dot(c.dot(jacobian.transpose())))))
            var_2be_updated = self.var[indices_c, :]
            updated_var = var_2be_updated - (k.dot(jacobian)).dot(var_2be_updated)
            updated_var = np.maximum(updated_var, min_var)
            self.var[indices_c, :] = updated_var
            self.var[:, indices_c] = updated_var.transpose()

            self.actions[nd_idx1, nd_idx2] = action

    def decide_greedily(self, nd_idx1, nd_idx2, params):
        if self.pebs[nd_idx1] < params['objective_peb'] and self.pebs[nd_idx2] < params['objective_peb']:
            return 0
        else:
            min_var = 1e-5
            rltv_pos = self.nds[int(nd_idx1)]['position'][:2] - self.nds[int(nd_idx2)]['position'][:2]
            l = np.linalg.norm(rltv_pos)
            jacobian = np.array([[rltv_pos[0] / l, rltv_pos[1] / l,
                                  -rltv_pos[0] / l, -rltv_pos[1] / l],
                                 [-rltv_pos[1] / l ** 2, rltv_pos[0] / l ** 2,
                                  rltv_pos[1] / l ** 2, -rltv_pos[0] / l ** 2]])
            indices_c = np.array([nd_idx1 * 2, nd_idx1 * 2 + 1, nd_idx2 * 2, nd_idx2 * 2 + 1], dtype=np.int8)
            c = self.var[np.reshape(indices_c, (4, 1)), indices_c]
            sigma_l = params['sigma_l']
            sigma_theta = params['sigma_alpha']
            sigma = np.diag([sigma_l ** 2, sigma_theta ** 2])
            # Kalman gain
            k = c.dot(jacobian.transpose()).dot(np.linalg.pinv((sigma + jacobian.dot(c.dot(jacobian.transpose())))))
            var_2be_updated = self.var[indices_c, :]
            updated_var = var_2be_updated - (k.dot(jacobian)).dot(var_2be_updated)
            updated_var = np.maximum(updated_var, min_var)
            peb1 = np.sqrt(updated_var[0, indices_c[0]] + updated_var[1, indices_c[1]])
            peb2 = np.sqrt(updated_var[2, indices_c[2]] + updated_var[3, indices_c[3]])

            if self.pebs[nd_idx1] > params['objective_peb'] > peb1:
                return 1
            elif self.pebs[nd_idx2] > params['objective_peb'] > peb2:
                return 1
            else:
                return 0

    def archive_actions(self):
        self.cumulative_actions = self.cumulative_actions + self.actions
        self.actions = np.zeros((len(self.nds), len(self.nds)))

    def reset(self):
        self.var = np.zeros((2 * len(self.nds), 2 * len(self.nds)))
        for nds_idx in range(len(self.nds)):
            if nds_idx == self.master_idx:
                self.var[nds_idx * 2, nds_idx * 2] = 1 / self.fim_gps_master[0, 0]
                self.var[nds_idx * 2 + 1, nds_idx * 2 + 1] = 1 / self.fim_gps_master[1, 1]
            else:
                self.var[nds_idx * 2, nds_idx * 2] = 1 / self.fim_gps[0, 0]
                self.var[nds_idx * 2 + 1, nds_idx * 2 + 1] = 1 / self.fim_gps[1, 1]
        self.actions = self.actions * 0
        self.cumulative_actions = self.cumulative_actions * 0

    @property
    def var_nds(self):
        return np.diag(self.var)

    @property
    def pebs(self):
        var = np.diag(self.var)
        return np.sqrt(var[:: 2] + var[1::2])

    def plot(self, pebs=None, pic=None):
        rc('font', **{'size': 15, 'family': 'Serif'})
        rc('text', **{'usetex': True})
        if pic is not None:
            im = image.imread(pic)
        for nd_idx, nd in enumerate(self.nds):
            if pebs is None:
                plt.text(nd['position'][0] + 1, nd['position'][1] + 1, str(nd_idx), fontsize=10, color='red')
            else:
                plt.text(nd['position'][0] + 1, nd['position'][1] + 1, str(nd_idx) + ', ' + '{:.2f}'.format(pebs[nd_idx]),
                         fontsize=14, color='red', bbox=dict(facecolor='white'))
            if pic is None:
                plt.plot(nd['position'][0], nd['position'][1], 'r.')
            else:
                plt.imshow(im, extent=(nd['position'][0] - 0.6, nd['position'][0] + 0.6,
                                       nd['position'][1] - 0.3, nd['position'][1] + 0.3), aspect='auto')

        for node_index1 in range(len(self.nds)):
            for node_index2 in range(len(self.nds)):
                if node_index1 != node_index2:
                    if self.cumulative_actions[node_index1, node_index2] + self.actions[node_index1, node_index2] > 0:
                        plt.plot([self.nds[node_index1]['position'][0], self.nds[node_index2]['position'][0]],
                                 [self.nds[node_index1]['position'][1], self.nds[node_index2]['position'][1]], 'k')

        for node_index1 in range(len(self.nds)):
            for node_index2 in range(len(self.nds)):
                if node_index1 != node_index2:
                    if self.actions[node_index1, node_index2] > 0:
                        plt.plot([self.nds[node_index1]['position'][0], self.nds[node_index2]['position'][0]],
                                 [self.nds[node_index1]['position'][1], self.nds[node_index2]['position'][1]], 'r')

        plt.axis('equal')
        plt.xlim((-1, self.xlim))
        plt.ylim((-1, self.ylim))
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')

    def objective_achieved(self, params):
        return all([peb < params['objective_peb'] for peb in self.pebs])

    def pass_msg_ngbrs(self, params):
        n_iters = 1
        dependencies = list()
        dstc2anchor = list()
        pebs = self.pebs
        for nd_idx in range(len(self.nds)):
            if pebs[nd_idx] > params['objective_peb']:
                dependencies.append([{'nd': nd_idx, 'dstc': 0}])
                dstc2anchor.append(1e5)
            else:
                dependencies.append(list())
                dstc2anchor.append(0)
        for _ in range(n_iters):
            for nd_idx in range(len(self.nds)):
                for adj in self.nds[nd_idx]['adj']:
                    for dependent in dependencies[adj]:
                        if dstc2anchor[dependent['nd']] - dstc2anchor[nd_idx] >= dependent['dstc'] + 1 and \
                                dependent['nd'] not in [d['nd'] for d in dependencies[nd_idx]]:
                            dependencies[nd_idx].append({'nd': dependent['nd'], 'dstc': dependent['dstc'] + 1})
                            dstc2anchor[dependent['nd']] = dstc2anchor[nd_idx] + dependent['dstc'] + 1
                            pass
        self.dependencies = dependencies


def in_fov(node1, node2, params):
    psi1 = node1['position'][2]
    p1 = node1['position'][0: 2]
    p2 = node2['position'][0: 2]
    r12 = np.linalg.norm(p1 - p2)
    p12_g = p2 - p1
    p2_e1 = rotation_mat(psi1).dot(p12_g)
    a12 = np.arctan2(p2_e1[1], p2_e1[0])
    if r12 < params['radar.r_max'] and abs(a12) <= params['radar.fov'] / 2:
        return True
    else:
        return False


def rotation_mat(angle):
    return np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])


def find_next_state_idcs(data: pd.DataFrame):
    state_descriptions = np.array(data[['epsd', 'scnr', 'nd_idx1', 'nd_idx2', 'exe_crt_agt']], dtype=np.int16)
    state_p_descriptions = copy.copy(state_descriptions)
    state_p_descriptions[:, -1] = state_p_descriptions[:, -1] + 1
    state_descriptions = state_descriptions[:, np.newaxis, :]
    state_p_descriptions = state_p_descriptions[np.newaxis, :, :]
    diff = np.all(np.abs(state_descriptions == state_p_descriptions), axis=2)
    idcs_states_p = list(np.where(diff == True)[0])
    nones = [None] * (data.shape[0] - len(idcs_states_p))
    idcs_states_p = idcs_states_p + nones
    return idcs_states_p


def find_state_p(data: pd.DataFrame, idcs_next_states, params):
    valid_idcs = np.where(np.array(idcs_next_states) != None)[0]
    idcs_next_states_without_none = [idcs_next_states[p] for p in valid_idcs]
    delta_x = [None] * len(idcs_next_states)
    delta_y = [None] * len(idcs_next_states)
    var1x = [None] * len(idcs_next_states)
    var1y = [None] * len(idcs_next_states)
    var2x = [None] * len(idcs_next_states)
    var2y = [None] * len(idcs_next_states)
    varxx = [None] * len(idcs_next_states)
    varyy = [None] * len(idcs_next_states)
    n_ngbrs = [0] * len(idcs_next_states)

    for index, value in zip(valid_idcs, idcs_next_states_without_none):
        delta_x[index] = data.loc[value, 'delta_x']
        delta_y[index] = data.loc[value, 'delta_y']
        var1x[index] = data.loc[value, 'var1x']
        var1y[index] = data.loc[value, 'var1y']
        var2x[index] = data.loc[value, 'var2x']
        var2y[index] = data.loc[value, 'var2y']
        varxx[index] = data.loc[value, 'varxx']
        varyy[index] = data.loc[value, 'varyy']
        n_ngbrs[index] = int(data.loc[value, 'n_ngbrs'])

    return delta_x, delta_y, var1x, var1y, var2x, var2y, varxx, varyy, n_ngbrs


def calc_reward(action, nd_idx1, nd_idx2, prvs_var, updt_var, params):
    reward = 0
    if updt_var is None:
        return reward
    if np.sqrt(prvs_var[nd_idx1 * 2] + prvs_var[nd_idx1 * 2 + 1]) > params['objective_peb'] > \
            np.sqrt(updt_var[nd_idx1 * 2] + updt_var[nd_idx1 * 2 + 1]):
        reward += params['terminal_reward']
    if np.sqrt(prvs_var[nd_idx2 * 2] + prvs_var[nd_idx2 * 2 + 1]) > params['objective_peb'] > \
            np.sqrt(updt_var[nd_idx2 * 2] + updt_var[nd_idx2 * 2 + 1]):
        reward += params['terminal_reward']
    if not params['sparse_reward']:
        pass  # to be implemented
    if action != 0:
        reward -= params['cost_mea'] * action
    return reward


def calc_reward_v2(data: pd.DataFrame, state_p_idcs, params, sparse=None):
    n_rows = data.shape[0]
    reward = np.zeros(n_rows)

    non_terminal_idcs = np.where(np.array(state_p_idcs) != None)[0]
    non_terminal_state_p_idcs = [state_p_idcs[n] for n in non_terminal_idcs]

    var = np.array(data.loc[0: n_rows, 'var1x'] + data.loc[0: n_rows, 'var1y'])
    non_terminal_var_p = np.array(data.loc[non_terminal_state_p_idcs, 'var1x'] +
                                  data.loc[non_terminal_state_p_idcs, 'var1y'])
    var_p = np.zeros(var.shape)
    var_p[non_terminal_idcs] += non_terminal_var_p
    claim_reward = np.array([v > params['objective_peb'] ** 2 > vp for v, vp in zip(var, var_p)] * 1)
    reward += params['terminal_reward'] * claim_reward

    var = np.array(data.loc[0: n_rows, 'var2x'] + data.loc[0: n_rows, 'var2y'])
    non_terminal_var_p = np.array(data.loc[non_terminal_state_p_idcs, 'var2x'] +
                                  data.loc[non_terminal_state_p_idcs, 'var2y'])
    var_p = np.zeros(var.shape)
    var_p[non_terminal_idcs] += non_terminal_var_p
    claim_reward = np.array([v > params['objective_peb'] ** 2 > vp for v, vp in zip(var, var_p)] * 1)
    reward += params['terminal_reward'] * claim_reward

    if not sparse:
        raise NotImplementedError

    reward -= params['cost_mea'] * data.loc[0: n_rows, 'action']

    return np.array(reward)


def calc_reward_greedy(data: pd.DataFrame, all_ber, params, sparse=None):
    n_rows = data.shape[0]
    reward = np.zeros(n_rows)

    non_terminal_idcs = range(n_rows)

    var = np.array(data.loc[0: n_rows, 'var1x'] + data.loc[0: n_rows, 'var1y'])

    var_p = np.array(list(ber[0] for ber in all_ber)) ** 2
    claim_reward = np.array([v > params['objective_peb'] ** 2 > vp for v, vp in zip(var, var_p)] * 1)
    reward += params['terminal_reward'] * claim_reward

    var = np.array(data.loc[0: n_rows, 'var2x'] + data.loc[0: n_rows, 'var2y'])
    var_p = np.array(list(ber[1] for ber in all_ber)) ** 2
    claim_reward = np.array([v > params['objective_peb'] ** 2 > vp for v, vp in zip(var, var_p)] * 1)
    reward += params['terminal_reward'] * claim_reward

    if not sparse:
        raise NotImplementedError

    reward -= params['cost_mea'] * data.loc[0: n_rows, 'action']

    return np.array(reward)


def find_var4reward(data: pd.DataFrame, all_var, state_idx, init_var):
    if state_idx is None:
        return None
    else:
        iter_idx = data['iter'][state_idx]
        epsd = data['epsd'][state_idx]
        scnr_idx = data['scnr'][state_idx]
        idx = data.index[(data['epsd'] == epsd) &
                           (data['iter'] == iter_idx - 1) &
                           (data['scnr'] == scnr_idx)].tolist()
        if len(idx) == 0:
            return np.diag(init_var)
        else:
            var = all_var[idx[0]]
            return var


def calc_reward_p(data: pd.DataFrame, reward, entry_idx, idcs_next_states):
    scnr_idx = data['scnr']
    iter_min = data['iter'][entry_idx]
    if idcs_next_states[entry_idx] is None:
        return reward[entry_idx]
    else:
        iter_max = data['iter'][idcs_next_states[entry_idx]]
        entry_idcs_in_between = list(data.loc[(data['scnr'] == scnr_idx) &
                                    (data['iter'] > iter_min) &
                                    (data['iter'] < iter_max)].index)
        if len(entry_idcs_in_between) == 0:
            return reward[entry_idx]

        cumulative_reward = np.sum(list(reward[i] for i in entry_idcs_in_between))

        return reward[entry_idx] + cumulative_reward / len(entry_idcs_in_between) / 3


def calc_reward_p_v2(data: pd.DataFrame, reward, idcs_next_states, epsd_idx, params):
    reward_plus = np.zeros(data.shape[0])
    for scnr_idx in range(params['num_scenarios']):
        data_crnt_scnr = data.loc[(data['epsd'] == epsd_idx) & (data['scnr'] == scnr_idx)]
        for idx, row in data_crnt_scnr.iterrows():
            if idcs_next_states[idx] is not None:
                iter_min = data.loc[idx, 'iter']
                iter_max = data.loc[idcs_next_states[idx], 'iter']
                entry_idcs_in_between = list(data_crnt_scnr.loc[(data['iter'] > iter_min) & (data['iter'] < iter_max)].index)

                if len(entry_idcs_in_between) > 0:
                    reward_plus[idx] += np.mean([reward[index] for index in entry_idcs_in_between])
    return reward + reward_plus / params['selfishness']


def calc_mean_std(data, params):
    d = np.array(data[params['state_def']], dtype='float')
    m = np.mean(d, axis=0)
    s = np.std(d, axis=0)
    s[0] *= 10
    s[1] *= 10
    return m, s


def normalize_state(states, m, s):
    m = np.reshape(m, (1, m.size))
    s = np.reshape(s, (1, s.size))
    states = (states - m) / s
    return states


def calc_scnr_rwd(data, params):
    rewards = np.zeros(data.shape[0])
    dsct = params['discounting']
    for scnr_idx in range(params['num_scenarios']):
        scnr_reward = np.array(data.loc[data['scnr'] == scnr_idx, 'reward'])
        scnr_idcs = data.index[data['scnr'] == scnr_idx].tolist()

        for idx in range(len(scnr_idcs)):
            rewards[scnr_idcs[idx]] = np.sum(scnr_reward)
    return rewards - np.mean(rewards)
