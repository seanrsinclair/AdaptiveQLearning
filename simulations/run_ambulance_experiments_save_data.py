import sys
sys.path.insert(1, '../')
import numpy as np
import gym
from adaptive_Agent import AdaptiveDiscretization
from eNet_model_Agent import eNetModelBased
from eNet_Agent import eNet
from adaptive_model_Agent import AdaptiveModelBasedDiscretization
from data_Agent import dataUpdateAgent
from src import environment
from src import experiment
from src import agent
import pickle
import multiprocessing as mp
from joblib import Parallel, delayed
import time


def run_single_algo(algorithm, env, dictionary, path, scaling_list, num_iters, epLen, nEps):
    opt_param = scaling_list[0]
    opt_reward = .01

    epsilon = (nEps * epLen)**(-1 / 4)
    action_net = np.arange(start=0, stop=1, step=epsilon)
    state_net = np.arange(start=0, stop=1, step=epsilon)


    # Running Experiments
    for scaling in scaling_list:

        agent_list = []
        for _ in range(num_iters):
            if algorithm == 'adaMB_One':
                agent_list.append(AdaptiveModelBasedDiscretization(epLen, nEps, scaling, 0, 2, False, False))

            elif algorithm == 'adaMB_Full':
                agent_list.append(AdaptiveModelBasedDiscretization(epLen, nEps, scaling, 0, 2, False, True))
            
            elif algorithm == 'adaMB_One_Flag':
                agent_list.append(AdaptiveModelBasedDiscretization(epLen, nEps, scaling, 0, 2, True, False))

            elif algorithm == 'adaMB_Full_Flag':
                agent_list.append(AdaptiveModelBasedDiscretization(epLen, nEps, scaling, 0, 2, True, True))

            elif algorithm == 'adaMB_One_3':
                agent_list.append(AdaptiveModelBasedDiscretization(epLen, nEps, scaling, 0, 3, False, False))

            elif algorithm == 'adaMB_Full_3':
                agent_list.append(AdaptiveModelBasedDiscretization(epLen, nEps, scaling, 0, 3, False, True))

            elif algorithm == 'adaMB_One_Flag_3':
                agent_list.append(AdaptiveModelBasedDiscretization(epLen, nEps, scaling, 0, 3, True, False))

            elif algorithm == 'adaMB_Full_Flag_3':
                agent_list.append(AdaptiveModelBasedDiscretization(epLen, nEps, scaling, 0, 3, True, True))

            elif algorithm == 'adaQL':
                agent_list.append(AdaptiveDiscretization(epLen, nEps, scaling))
            elif algorithm == 'epsMB_One':
                epsilon = (nEps * epLen)**(-1 / 4)
                action_net = np.arange(start=0, stop=1, step=epsilon)
                state_net = np.arange(start=0, stop=1, step=epsilon)
                agent_list.append(eNetModelBased(action_net, state_net, epLen, scaling, 0, False))

            elif algorithm == 'epsMB_Full':
                epsilon = (nEps * epLen)**(-1 / 4)
                action_net = np.arange(start=0, stop=1, step=epsilon)
                state_net = np.arange(start=0, stop=1, step=epsilon)
                agent_list.append(eNetModelBased(action_net, state_net, epLen, scaling, 0, True))

            elif algorithm == 'epsQL':
                epsilon = (nEps * epLen)**(-1 / 4)
                action_net = np.arange(start=0, stop=1, step=epsilon)
                state_net = np.arange(start=0, stop=1, step=epsilon)
                agent_list.append(eNet(action_net, state_net, epLen, scaling))           

        exp = experiment.Experiment(env, agent_list, dictionary)
        _ = exp.run()
        dt_data = exp.save_data()

        if (dt_data.groupby(['episode']).mean().tail(1))['epReward'].iloc[0] > opt_reward:
            opt_reward = (dt_data.groupby(['episode']).mean().tail(1))['epReward'].iloc[0]
            opt_param = scaling
            dt_final_data = dt_data
            opt_agent_list = agent_list

    # Saving Data 
    dt_final_data.to_csv(path+'.csv')
    agent = opt_agent_list[-1]
    filehandler = open(path+'.obj', 'wb')
    pickle.dump(agent, filehandler)

    return (algorithm,opt_param,opt_reward)

''' Defining parameters to be used in the experiment'''



ambulance_list = ['shifting', 'beta', 'uniform']
param_list_ambulance = ['0', '1', '25']
algo_list = ['adaMB_One', 'adaMB_Full', 'adaMB_One_Flag', 'adaMB_Full_Flag', 'adaMB_One_3', 'adaMB_Full_3', 'adaMB_One_Flag_3', 'adaMB_Full_Flag_3', 'adaQL', 'epsMB_One', 'epsMB_Full', 'epsQL']


for problem in ambulance_list:
    for param in param_list_ambulance:

        epLen = 5
        nEps = 2000
        numIters = 200
        if problem == 'beta':
            def arrivals(step):
                return np.random.beta(5,2)
        elif problem == 'uniform':
            def arrivals(step):
                return np.random.uniform(0,1)
        elif problem == 'shifting':
            def arrivals(step):
                if step == 0:
                    return np.random.uniform(0, .25)
                elif step == 1:
                    return np.random.uniform(.25, .3)
                elif step == 2:
                    return np.random.uniform(.3, .5)
                elif step == 3:
                    return np.random.uniform(.5, .6)
                else:
                    return np.random.uniform(.6, .65)
        if param == '1':
            alpha = 1
        elif param == '0':
            alpha = 0
        else:
            alpha = 0.25

        starting_state = 0.5

        env = environment.make_ambulanceEnvMDP(epLen, arrivals, alpha, starting_state)

        ##### PARAMETER TUNING FOR AMBULANCE ENVIRONMENT

        scaling_list = [0.01, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1, 1.5, 2, 5]
 

        dictionary = {'seed': 1, 'epFreq' : 1, 'targetPath': './tmp.csv', 'deBug' : False, 'nEps': nEps, 'recFreq' : 10, 'numIters' : numIters}
        scaling_list = [0.01, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1, 1.5, 2, 5]

        print('# CPUs: ' + str(mp.cpu_count()))
        time.sleep(2)
        def get_result(result):
            global results
            results.append(result)

        results = []
        ts = time.time()  

        pool = mp.Pool(mp.cpu_count())

        path = {}
        for algorithm in algo_list:
            path[algorithm] = '../data/ambulance_'+problem+'_'+param+'_'+algorithm

        results = Parallel(n_jobs = mp.cpu_count())(delayed(run_single_algo)(algorithm, env, dictionary, path[algorithm], scaling_list, numIters, epLen, nEps) for algorithm in algo_list)

        # pool.close()
        # pool.join()
        print('Time in parallel:', time.time() - ts)
        print(results)
