from environment import *
from dqn import *
import pickle as pl


np.seterr(all='raise')
params = {'lane_width': 4,
          'num_scenarios': 1000,
          'pos_var': 0.3,
          'num_episodes': 6,
          'num_trainings_after_simulation': 10,
          'num_iterations': 400,
          'num_nds': 10,
          'num_lanes': 2,
          'actions': [0, 1],
          'radar.fov': 2 * np.pi,
          'radar.r_max': 7.5,
          'sig_gps': 3.4,
          'noise_l_max': 0.2,
          'noise_alpha_max': 0.02,
          'sigma_l': 0.1,
          'sigma_alpha': 0.1 * np.pi / 180,
          'fim_gps': None,
          'fim_gps_master': None,
          'objective_peb': 0.12,
          'cost_mea': 0.1,
          'terminal_reward': 1.2,
          'discounting': 0.7,
          'state_def': ['delta_x', 'delta_y', 'var1x', 'var1y', 'var2x', 'var2y', 'varxx', 'varyy', 'n_ngbrs'],
          'state_p_def': ['delta_x_p', 'delta_y_p', 'var1x_p', 'var1y_p', 'var2x_p', 'var2y_p', 'varxx_p', 'varyy_p',
                          'n_ngbrs_p'],
          'saving_path': 'tf_models/current',
          'xlim': 10,
          'round_robin': False,
          'greedy': False,
          'random': True,
          'sparse_reward': True,
          'm': np.array([1.5, 2.7, 0.3, 0.3, 1.7, 1.7, 0, 0, 3]),
          's': np.array([40, 40, 1.1, 1.1, 2.6, 2.6, 2.1, 2.1, 2.5])}
params['xlim'] = (params['num_nds'] / params['num_lanes'] - 1) * 5
if params['num_lanes'] == 3:
    params['noise_l_max'] = 0.25
    params['noise_alpha_max'] = 0.025
elif params['num_lanes'] == 1:
    params['noise_l_max'] = 0.2
    params['noise_alpha_max'] = 0.02

# attributes that can be determined instantly
headers1 = ['epsd', 'iter', 'scnr', 'nd_idx1', 'nd_idx2', 'exe_crt_agt', 'delta_x', 'delta_y', 'var1x', 'var1y',
            'var2x', 'var2y', 'varxx', 'varyy', 'n_ngbrs', 'action']
# attributes that must be determined after the simulation
headers2 = ['reward', 'reward_p', 'delta_x_p', 'delta_y_p', 'var1x_p', 'var1y_p', 'var2x_p', 'var2y_p', 'varxx_p',
            'varyy_p', 'n_ngbrs_p', 'q']
# All attributes
headers = headers1 + headers2

sig_gps = params['sig_gps']
gps_fim = np.diag([1 / sig_gps ** 2, 1 / sig_gps ** 2]) * 2
gps_fim_master = np.diag([1 / sig_gps ** 2, 1 / sig_gps ** 2]) * 1e9
params['fim_gps'] = gps_fim
params['fim_gps_master'] = gps_fim_master

data = pd.DataFrame(columns=headers)
n_objective_reached = np.ones(params['num_iterations'])
# data = pd.DataFrame()
# for h, t in zip(headers, types):
#     data[h] = pd.Series(dtype=t)

scenarios = list()
for scenario_idx in range(params['num_scenarios']):
    scenarios.append(Scenario(params['num_nds'], params['num_lanes'], params))
    scenarios[scenario_idx].pass_msg_ngbrs(params)
epsd_idx = 0


with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    tf.saved_model.loader.load(sess, ['serve'], params['saving_path'])
    # dqn.restore(sess, 'tf_models')
    graph = tf.get_default_graph()
    print(graph.get_operations())

    data_this_epsd = pd.DataFrame(columns=headers1)
    all_var = list()
    exe_agts = np.zeros((params['num_scenarios'], 200), dtype=int)
    for scenario in scenarios:
        scenario.reset()
    for itr_idx in range(params['num_iterations']):
        for scenario_index, scenario in enumerate(scenarios):
            scenario.archive_actions()
        raw_data = list()
        for scnr_idx, scenario in enumerate(scenarios):
            if params['round_robin']:
                agt_idx = itr_idx % len(scenario.links)
            else:
                agt_idx = np.random.randint(0, len(scenario.links))
            agt = scenario.links[agt_idx]
            exe_agts[scnr_idx, agt_idx] += 1

            delta_x, delta_y, var1x, var1y, var2x, var2y, varxx, varyy, n_nbgrs =\
                scenario.gen_state(agt[0], agt[1], params)
            action = 0  # action is set to 0 here because we need the state description to predict.

            entry = [epsd_idx, itr_idx, scnr_idx, agt[0], agt[1], exe_agts[scnr_idx, agt_idx],
                     delta_x, delta_y, var1x, var1y, var2x, var2y, varxx, varyy, n_nbgrs, action]
            raw_data.append(entry)

        data_this_epsd_iter = pd.DataFrame(raw_data, columns=headers1)
        input_state = (np.array(data_this_epsd_iter[params['state_def']]) - params['m']) / params['s']
        q = sess.run('q:0', feed_dict={'state:0': input_state})
        if params['random']:
            actions = epsilon_greedy(1, q)
        else:
            actions = epsilon_greedy(0, q)

        # debug
        # if q[0, 0] < q[0, 1] + 0.3:
        #     actions = [1]
        # else:
        #     actions = [0]

        if params['greedy']:
            agts = data_this_epsd_iter[['nd_idx1', 'nd_idx2']].values.tolist()
            actions = [scenario.decide_greedily(agt[0], agt[1], params) for scenario, agt in zip(scenarios, agts)]

        for idx, scenario in enumerate(scenarios):
            if scenario.objective_achieved(params):
                actions[idx] = 0

        # for idx, scenario in enumerate(scenarios):
        #     if np.sum(scenario.cumulative_actions) >= 15:
        #         actions[idx] = 0

        data_this_epsd_iter['action'] = actions
        for row_idx in range(data_this_epsd_iter.shape[0]):
            scnr_idx = data_this_epsd_iter.loc[row_idx, 'scnr']
            nd_idx1 = data_this_epsd_iter.loc[row_idx, 'nd_idx1']
            nd_idx2 = data_this_epsd_iter.loc[row_idx, 'nd_idx2']
            scenarios[scnr_idx].update_var(actions[row_idx], nd_idx1, nd_idx2, params)
            all_var.append(np.diag(scenarios[scnr_idx].var))

        data_this_epsd = pd.concat([data_this_epsd, data_this_epsd_iter], axis=0, ignore_index=True)
        if actions[0] == 1:
            scenarios[0].plot(scenarios[0].pebs, 'car.png')
            pass
        if itr_idx % 100 == 0:
            print(itr_idx)
            if all([scenario.objective_achieved(params) for scenario in scenarios]):
                break

        # n_reached = sum([scenario.objective_achieved(params) for scenario in scenarios] * 1) / params['num_scenarios']
        n_reached = np.mean(list(map(lambda s: sum((s.pebs < params['objective_peb'] * 1)), scenarios)))
        n_objective_reached[itr_idx] = n_reached

    objective_achieved = [scenario.objective_achieved(params) for scenario in scenarios] * 1
    # n_measurements = [np.sum(scenario.cumulative_actions) for scenario in scenarios]
    # pl.dump([objective_achieved, n_measurements], open('results/performance_greedy.p', 'wb'))
