from pg import *
from environment import *
import os
import shutil
import pickle as pl


np.seterr(all='raise')
params = {'lane_width': 4,
          'num_scenarios': 100,
          'pos_var': 0.3,
          'num_episodes': 3000,
          'num_trainings_after_simulation': 12,
          'n_epochs': 300,
          'patience': 16,
          'thrhld_earlystopping': 0.02,
          'batch_size': 128,
          'n_neurons': 100,
          'num_iterations': 40,
          'num_nds': 9,
          'num_lanes': 3,
          'actions': [0, 1],
          'radar.fov': 2 * np.pi,
          'radar.r_max': 7.5,
          'sig_gps': 3.4,
          'noise_l_max': 0.2,
          'noise_alpha_max': 0.02,
          'sigma_l': 0.1,
          'sigma_alpha': 0.1 * np.pi / 180,
          'lr': 0.0001,
          'fim_gps': None,
          'fim_gps_master': None,
          'objective_peb': 0.12,
          'cost_mea': 0.1,
          'terminal_reward': 1.2,
          'discounting': 1,
          'state_def': ['delta_x', 'delta_y', 'var1x', 'var1y', 'var2x', 'var2y', 'varxx', 'varyy', 'n_ngbrs'],
          'state_p_def': ['delta_x_p', 'delta_y_p', 'var1x_p', 'var1y_p', 'var2x_p', 'var2y_p', 'varxx_p', 'varyy_p',
                          'n_ngbrs_p'],
          'saving_path': 'tf_models/current',
          'xlim': 40,
          'selfishness': 1,
          'round_robin': True,
          'sparse_reward': True,
          'double_dqn': False,
          'updating_interval4double_dqn': 20,
          'min_loss': 0.008,
          'm': np.array([1.5, 2.7, 0.3, 0.3, 1.7, 1.7, 0, 0, 3]),
          's': np.array([40, 40, 1.1, 1.1, 2.6, 2.6, 0.1, 0.1, 2.5]),
          'm_reward': 0,
          's_reward': 1.6}
params['xlim'] = (params['num_nds'] / params['num_lanes'] - 1) * 5
if params['num_lanes'] == 3:
    params['noise_l_max'] = 0.25
    params['noise_alpha_max'] = 0.025
elif params['num_lanes'] == 1:
    params['noise_l_max'] = 0.2
    params['noise_alpha_max'] = 0.02

headers = ['epsd', 'iter', 'scnr', 'nd_idx1', 'nd_idx2', 'exe_crt_agt', 'delta_x', 'delta_y', 'var1x', 'var1y',
            'var2x', 'var2y', 'varxx', 'varyy', 'n_ngbrs', 'action', 'reward']

sig_gps = params['sig_gps']
gps_fim = np.diag([1 / sig_gps ** 2, 1 / sig_gps ** 2]) * 2
gps_fim_master = np.diag([1 / sig_gps ** 2, 1 / sig_gps ** 2]) * 1e9
params['fim_gps'] = gps_fim
params['fim_gps_master'] = gps_fim_master
all_mean_rewards = np.empty(params['num_episodes'])

loss = 1
old_loss = 10
training_idx = 0
converged_training = 0
prvs_rwd = 0

scenarios = list()
for scenario_idx in range(params['num_scenarios']):
    scenarios.append(Scenario(params['num_nds'], params['num_lanes'], params))
    scenarios[scenario_idx].pass_msg_ngbrs(params)


with tf.Session() as sess:
    pg = PolicyGradient(params)
    sess.run(tf.global_variables_initializer())
    for epsd_idx in range(params['num_episodes']):
        data_this_epsd = pd.DataFrame(columns=headers)
        exe_agts = np.zeros((params['num_scenarios'], 200), dtype=int)
        for scenario in scenarios:
            scenario.reset()
        for itr_idx in range(params['num_iterations']):
            raw_data = list()
            for scnr_idx, scenario in enumerate(scenarios):
                if params['round_robin']:
                    agt_idx = itr_idx % len(scenario.links)
                else:
                    agt_idx = np.random.randint(0, len(scenario.links))
                agt = scenario.links[agt_idx]
                exe_agts[scnr_idx, agt_idx] += 1

                delta_x, delta_y, var1x, var1y, var2x, var2y, varxx, varyy, n_nbgrs = \
                    scenario.gen_state(agt[0], agt[1], params)
                action = 0  # action is set to 0 here because we need the state description to predict.

                entry = [epsd_idx, itr_idx, scnr_idx, agt[0], agt[1], exe_agts[scnr_idx, agt_idx],
                         delta_x, delta_y, var1x, var1y, var2x, var2y, varxx, varyy, n_nbgrs, action, 0]
                raw_data.append(entry)

            data_this_epsd_iter = pd.DataFrame(raw_data, columns=headers)
            # Select action
            actions = pg.choose_action(data_this_epsd_iter[params['state_def']], sess, params)
            data_this_epsd_iter['action'] = actions
            for row_idx in range(data_this_epsd_iter.shape[0]):
                scnr_idx = data_this_epsd_iter.loc[row_idx, 'scnr']
                prvs_var = np.copy(np.diag(scenarios[scnr_idx].var))
                nd_idx1 = data_this_epsd_iter.loc[row_idx, 'nd_idx1']
                nd_idx2 = data_this_epsd_iter.loc[row_idx, 'nd_idx2']
                scenarios[scnr_idx].update_var(actions[row_idx], nd_idx1, nd_idx2, params)
                updt_var = np.copy(np.diag(scenarios[scnr_idx].var))
                reward = calc_reward(data_this_epsd_iter.loc[row_idx, 'action'], nd_idx1, nd_idx2,
                                     prvs_var, updt_var, params)
                data_this_epsd_iter.loc[row_idx, 'reward'] = reward

            data_this_epsd = pd.concat([data_this_epsd, data_this_epsd_iter], axis=0, ignore_index=True)

        if epsd_idx == 0 and False:
            m, s = calc_mean_std(data_this_epsd, params)
            print('m = np.array({})'.format(list(m)))
            print('s = np.array({})'.format(list(s)))
            params['m'] = m
            params['s'] = s
        # Train DNN
        new_loss = pg.train(data_this_epsd, params, sess, epsd_idx, loss)
        mean_reward = np.sum(data_this_epsd['reward']) / params['num_scenarios']
        all_mean_rewards[epsd_idx] = mean_reward
        n_all_reached = sum(list(sum(scenario.pebs < params['objective_peb']) == params['num_nds']
                                 for scenario in scenarios))
        print('Mean reward: {0:.2f}, No. completed scenarios: {1} '
              'for episode {2}.'.format(mean_reward, n_all_reached, epsd_idx))
        if np.abs(mean_reward - prvs_rwd) < params['thrhld_earlystopping']:
            converged_training += 1
        else:
            converged_training = 0
        prvs_rwd = mean_reward
        if converged_training >= params['patience'] or training_idx % 100 == 0:
            if os.path.exists(params['saving_path']):
                shutil.rmtree(params['saving_path'])
            tf.saved_model.simple_save(sess, params['saving_path'], {'state': pg._state}, {'prob': pg._prob})
        if converged_training >= params['patience']:
            break
        training_idx += 1

pl.dump(all_mean_rewards, open('tf_models/current/all_mean_rewards_pg.p', 'wb'))
print('It is ended.')
