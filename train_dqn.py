from environment import *
from dqn import *
import os, shutil
import pickle as pl


np.seterr(all='raise')
params = {'lane_width': 4,
          'num_scenarios': 40,
          'num_episodes': 650,
          'num_trainings_after_simulation': 12,
          'n_epochs': 80,
          'n_trainings_in_epsd': 2,
          'patience': 16,
          'thrhld_earlystopping': 0.005,
          'batch_size': 512,
          'n_neurons': 100,
          'num_iterations': 80,
          'num_nds': 9,
          'num_lanes': 3,
          'actions': [0, 1],
          'radar.fov': 2 * np.pi,
          'radar.r_max': 7.5,
          'sig_gps': 3.4,
          'noise_l_max': 0.2,
          'noise_alpha_max': 0.02,
          'length_epsilon=0': 300,
          'sigma_l': 0.1,
          'sigma_alpha': 0.1 * np.pi / 180,
          'lr': 5e-5,
          'alpha': 0.5,
          'fim_gps': None,
          'fim_gps_master': None,
          'greedy': False,
          'drl1': False,
          'inherit_q': True,
          'objective_peb': 0.12,
          'pos_var': 0.3,
          'cost_mea': 0.1,
          'terminal_reward': 1.2,
          'discounting': 0.75,
          'state_def': ['delta_x', 'delta_y', 'var1x', 'var1y', 'var2x', 'var2y', 'varxx', 'varyy', 'n_ngbrs'],
          'state_p_def': ['delta_x_p', 'delta_y_p', 'var1x_p', 'var1y_p', 'var2x_p', 'var2y_p', 'varxx', 'varyy',
                          'n_ngbrs_p'],
          'saving_path': 'tf_models/current',
          'xlim': 40,
          'selfishness': 3,
          'round_robin': True,
          'sparse_reward': True,
          'double_dqn': False,
          'updating_interval4double_dqn': 20,
          'min_loss': 0.008,
          'm': np.array([1.5, 2.7, 0.3, 0.3, 1.7, 1.7, 0, 0, 3]),
          's': np.array([40, 40, 1.1, 1.1, 2.6, 2.6, 2.1, 2.1, 2.5])}
params['xlim'] = (params['num_nds'] / params['num_lanes'] - 1) * 5
if params['num_lanes'] == 3:
    params['noise_l_max'] = 0.25
    params['noise_alpha_max'] = 0.025
elif params['num_lanes'] == 1:
    params['noise_l_max'] = 0.2
    params['noise_alpha_max'] = 0.02

if params['greedy']:
    params['discounting'] = 0
    params['selfishness'] = 1e9
    params['objective_peb'] = 0.12

if params['drl1']:
    params['discounting'] = 0.7
    params['selfishness'] = 1e9

# attributes that can be determined instantly
headers1 = ['epsd', 'iter', 'scnr', 'nd_idx1', 'nd_idx2', 'exe_crt_agt', 'delta_x', 'delta_y', 'var1x', 'var1y',
            'var2x', 'var2y', 'varxx', 'varyy', 'n_ngbrs', 'action']
# attributes that must be determined after the simulation
headers2 = ['reward', 'reward_p', 'delta_x_p', 'delta_y_p', 'var1x_p', 'var1y_p', 'var2x_p', 'var2y_p', 'varxx_p',
            'varyy_p', 'n_ngbrs_p',
            'q']
# All attributes
headers = headers1 + headers2

sig_gps = params['sig_gps']
gps_fim = np.diag([1 / sig_gps ** 2, 1 / sig_gps ** 2]) * 2
gps_fim_master = np.diag([1 / sig_gps ** 2, 1 / sig_gps ** 2]) * 1e9
params['fim_gps'] = gps_fim
params['fim_gps_master'] = gps_fim_master
all_losses = list()

loss = 1
training_idx = 0
converged_training = 0

scenarios = list()
for scenario_idx in range(params['num_scenarios']):
    scenarios.append(Scenario(params['num_nds'], params['num_lanes'], params))
    scenarios[scenario_idx].pass_msg_ngbrs(params)


with tf.Session() as sess:
    dqn = DQN(params)
    sess.run(tf.global_variables_initializer())
    for epsd_idx in range(params['num_episodes']):
        if params['drl1']:
            states_p = list()
        if epsd_idx % 10 == 0:
            print('Episode {}...'.format(epsd_idx))
        data_this_epsd = pd.DataFrame(columns=headers1)
        all_var = list()
        all_ber = list()
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
                         delta_x, delta_y, var1x, var1y, var2x, var2y, varxx, varyy, n_nbgrs, action]
                raw_data.append(entry)

            data_this_epsd_iter = pd.DataFrame(raw_data, columns=headers1)
            # Select action
            q = dqn.predict(data_this_epsd_iter[params['state_def']], sess, params['m'], params['s'])
            # epsilon = np.max([0.995 ** epsd_idx, 0]) + 0.1
            epsilon = max(1 - epsd_idx / (params['num_episodes'] - params['length_epsilon=0']), 0.0) + 0.0
            actions = epsilon_greedy(epsilon, q)
            data_this_epsd_iter['action'] = actions
            for row_idx in range(data_this_epsd_iter.shape[0]):
                scnr_idx = data_this_epsd_iter.loc[row_idx, 'scnr']
                nd_idx1 = data_this_epsd_iter.loc[row_idx, 'nd_idx1']
                nd_idx2 = data_this_epsd_iter.loc[row_idx, 'nd_idx2']
                scenarios[scnr_idx].update_var(actions[row_idx], nd_idx1, nd_idx2, params)
                all_var.append(np.diag(scenarios[scnr_idx].var))
                all_ber.append(list(scenarios[scnr_idx].pebs[nd_idx] for nd_idx in [nd_idx1, nd_idx2]))

                if params['drl1']:
                    states_p.append(scenarios[row_idx].gen_state(nd_idx1, nd_idx2, params))

            data_this_epsd = pd.concat([data_this_epsd, data_this_epsd_iter], axis=0, ignore_index=True)

        # During the simulation, state_p, reward and reward_p are not updated
        # because they are not instantly known after the action.
        # Now we must calculate them before putting results to the final data frame.
        print('Finding states prime...')
        idcs_next_states = find_next_state_idcs(data_this_epsd)

        print('Calculating reward...')
        # reward
        if params['greedy']:
            reward = calc_reward_greedy(data_this_epsd, np.array(all_ber), params)
        else:
            reward = calc_reward_v2(data_this_epsd, idcs_next_states, params)

        # Q values, 0 for now.
        q = np.zeros(len(reward))

        if params['drl1']:
            reward_p = reward
            delta_x_p = [row[0] for row in states_p]
            delta_y_p = [row[1] for row in states_p]
            var1x_p = [row[2] for row in states_p]
            var1y_p = [row[3] for row in states_p]
            var2x_p = [row[4] for row in states_p]
            var2y_p = [row[5] for row in states_p]
            varxx_p = [row[6] for row in states_p]
            varyy_p = [row[7] for row in states_p]
            n_ngbrs_p = [row[8] for row in states_p]
        else:
            print('Calculating reward prime...')
            if not params['greedy']:
                reward_p = calc_reward_p_v2(data_this_epsd, reward, idcs_next_states, epsd_idx, params)
            else:
                reward_p = reward

            # state prime
            delta_x_p, delta_y_p, var1x_p, var1y_p, var2x_p, var2y_p, varxx_p, varyy_p, n_ngbrs_p = \
                find_state_p(data_this_epsd, idcs_next_states, params)
        postponed_data = pd.DataFrame({'reward': reward,
                                       'reward_p': reward_p,
                                       'delta_x_p': delta_x_p,
                                       'delta_y_p': delta_y_p,
                                       'var1x_p': var1x_p,
                                       'var1y_p': var1y_p,
                                       'var2x_p': var2x_p,
                                       'var2y_p': var2y_p,
                                       'varxx_p': varxx_p,
                                       'varyy_p': varyy_p,
                                       'n_ngbrs_p': n_ngbrs_p,
                                       'q': q})
        postponed_data['n_ngbrs_p'] = postponed_data['n_ngbrs_p'].astype('int')
        data_this_epsd = pd.concat([data_this_epsd, postponed_data], axis=1)

        # data = pd.concat([data, data_this_epsd], axis=0, ignore_index=True, sort=False)  # concatenate data

        if epsd_idx == 0 and False:
            m, s = calc_mean_std(data_this_epsd, params)
            print('m = np.array({})'.format(list(m)))
            print('s = np.array({})'.format(list(s)))
            params['m'] = m
            params['s'] = s
        # Train DNN
        # data['q'] = data['reward_p']
        if training_idx > 0:
            updated_q = dqn.update_q(data_this_epsd, sess, params)  # difference in double DQN
            data_this_epsd['q'] = updated_q
        new_loss = dqn.train(data_this_epsd, sess, epsd_idx, params, loss)

        # debug
        action_portion = np.sum(data_this_epsd['action']) / data_this_epsd.shape[0]

        print('Training finished with loss {0} and action portion {1}.'.format(new_loss, action_portion))
        if new_loss < loss * 1.5:
            # set a lower threshold of loss, such that the model can be saved more frequently.
            loss = np.max([new_loss, params['min_loss']])
        if params['double_dqn'] and training_idx % params['updating_interval4double_dqn'] == 0:
            # Update theta_m
            t_params = tf.get_collection('tgt_c_name')
            e_params = tf.get_collection('eval_c_name')
            replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
            sess.run(replace_target_op)
        training_idx += 1
        all_losses.append(new_loss)

        print('Episode {} finished.'.format(epsd_idx))
        all_ber.clear()

        if new_loss < params['thrhld_earlystopping']:
            converged_training += 1
        else:
            converged_training = 0
        if (training_idx + 1) % 20 == 0:
            if os.path.exists(params['saving_path']):
                shutil.rmtree(params['saving_path'])
            tf.saved_model.simple_save(sess, params['saving_path'], {'state': dqn._state}, {'q': dqn._q})
        if converged_training >= params['patience']:
            if os.path.exists(params['saving_path']):
                shutil.rmtree(params['saving_path'])
            tf.saved_model.simple_save(sess, params['saving_path'], {'state': dqn._state}, {'q': dqn._q})
            break

pl.dump(all_losses, open(params['saving_path'] + '/all_losses.p', 'wb'))
print('It is ended.')
