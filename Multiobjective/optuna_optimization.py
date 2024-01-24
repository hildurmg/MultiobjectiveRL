import os
import mlflow
import optuna
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import torch as th
from main_train import make_env
from typing import Dict
from optuna_helper_funcs import get_agent_id, set_seed
from stable_baselines3 import PPO, DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.envs.registration import register
from oop_simulation_script import Simulation, CommuteEnv
from morl_baselines.multi_policy.capql.capql import CAPQL
from gymnasium.envs.registration import register
import gymnasium
import json


def make_env(seed):
    register(
        id="traffic-env-v0",
        entry_point="oop_simulation_script:CommuteEnv",
    )
    env = gymnasium.make("traffic-env-v0")
    #env = Monitor(CommuteEnv(reward_type,days_per_eps))
    #env.seed(seed)
    return env

def sample_hyperparameters(
    trial: optuna.trial.Trial
) -> Dict:

    # learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    discount_factor = trial.suggest_categorical("discount_factor", [0.99, 0.8, 0.7])
    batch_size = trial.suggest_categorical("batch_size", [128, 64, 32])

    # NN architecture
    nn_hidden_layers = trial.suggest_categorical("nn_hidden_layers", ["[64, 64]","[64, 32]","[32, 32]"])
    nn_hidden_layers = {"[64, 64]": [64, 64], "[64, 32]": [64, 32], "[32, 32]": [32, 32]}[nn_hidden_layers]

    # Entropy regularization coefficient
    alpha = trial.suggest_categorical("alpha", [0.2, 0.4, 0.6, 0.8])
    
    return {
        #'learning_rate': learning_rate,
        'discount_factor': discount_factor,
        'batch_size': batch_size,
        'nn_hidden_layers': nn_hidden_layers,
        'alpha': alpha,
    }
    

def objective(
    trial: optuna.trial.Trial,
    train_eps: int = 75,
    eval_eps: int = 10,
    days_per_eps: int = 30
) -> float:
    """
    Samples hyperparameters, trains, and evaluates the RL agent.
    It outputs the average reward on 1,000 episodes.
    """

    env = make_env(10) # Seed as argument
    train_eps = train_eps # for training
    eps = train_eps # for naming
    nr_timesteps = train_eps*days_per_eps
    
    trial_dir = "../local_stats/MORL_optuna/"

    if not os.path.exists(trial_dir):
        os.makedirs(trial_dir)

    trial_nos = [0]
    for filename in os.listdir(trial_dir):
        trial_nos.append(int(filename.split('_')[1]))

    trial_name = 'trial_'+str(max(trial_nos)+1)
    
    tensorboard_dir = "./tensorboards/MORL_hyperparams/CAPQL/"+trial_name
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    with mlflow.start_run():

        agent_dir = trial_dir+trial_name
        # generate unique agent_id
        agent_id = get_agent_id(agent_dir, trial_name)
        
        mlflow.log_param('agent_id', agent_id)

        # hyper-parameters
        args = sample_hyperparameters(trial)
        mlflow.log_params(trial.params)

        # fix seeds to ensure reproducible runs
        #set_seed(env, args['seed'])

        # Load trained model
        print("Creating model...")

        n_actions = env.action_space.shape[-1]

        ref_point=np.array([0, -25])
        agent = CAPQL(
            env,
            gamma=args["discount_factor"],
            net_arch=args["nn_hidden_layers"],
            batch_size=args["batch_size"],
            alpha=args["alpha"],
            learning_starts=200,
            seed=1
        )

        stats_dir = agent_dir + '/stats/'

        if not os.path.exists(stats_dir):
            os.makedirs(stats_dir)

        with open(stats_dir + str(eps) + "CAPQL_params.txt", "w") as fp:
            json.dump(args, fp)  # encode dict into JSON
        
        print("Training agent...")

        #eval_env = make_env(10)
        
        eval_reward = agent.train(
            total_timesteps=nr_timesteps,
            ref_point=ref_point,
            eval_env=env,
            eval_freq=nr_timesteps/10,
            num_eval_weights_for_front=10,
            weights_dir = stats_dir+"weights/"
        )

        #model.save(agent_dir + agent_id)
        #agent.save_to_disk(SAVED_AGENTS_DIR / env_name / agent_id)

        #print("Evaluating agent...")
        #ep_rews, ep_lens = evaluate_policy(model, 
        #                                   env, 
        #                                   n_eval_episodes=eval_eps, 
        #                                   return_episode_rewards=True, 
        #                                   deterministic=True) # test if I can get tt, sw etc. from env after eval
        #ep_rews = np.array(ep_rews)
        #mean_reward = np.mean(ep_rews)
        #std_reward = np.std(ep_rews)
        
        
        # evaluate its performance
        #rewards, steps = evaluate(agent, env, n_episodes=1000, epsilon=0.00)
        #mean_reward = np.mean(rewards)
        #std_reward = np.std(rewards)

        mlflow.log_metric('mean_reward', eval_reward)
        #mlflow.log_metric('std_reward', std_reward)

        np.save(stats_dir + str(eps) + "CAPQL_tt.npy"          , env.get_tt())
        np.save(stats_dir + str(eps) + "CAPQL_sw.npy"          , env.get_sw())
        np.save(stats_dir + str(eps) + "CAPQL_cs.npy"          , env.get_cs())
        np.save(stats_dir + str(eps) + "CAPQL_gc.npy"          , env.get_gc())
        np.save(stats_dir + str(eps) + "CAPQL_tc.npy"          , env.get_tc())
        np.save(stats_dir + str(eps) + "CAPQL_ttcs.npy"        , env.get_ttcs())
        np.save(stats_dir + str(eps) + "CAPQL_ind_inc.npy"     , env.get_income())
        np.save(stats_dir + str(eps) + "CAPQL_toll.npy"        , env.get_toll())
        np.save(stats_dir + str(eps) + "CAPQL_toll_profile.npy", env.get_all_toll_profiles())
        np.save(stats_dir + str(eps) + "CAPQL_actions.npy"     , env.get_actions())
        np.save(stats_dir + str(eps) + "CAPQL_user_flow.npy"   , env.get_user_flow())
        #np.save(stats_dir + "/" + str(final_model_timesteps) + "_ind_cs.npy", env.get_individual_cs())
        #np.save(stats_dir + "/" + str(final_model_timesteps) + "_ind_sw.npy", env.get_individual_sw())


    return mean_reward

    
    
if __name__ == '__main__':

    mlflow_dir = 'local_stats/mlflow/'
    if not os.path.exists(mlflow_dir):
        os.makedirs(mlflow_dir)
    
    #days_per_eps = 30
    #train_eps = 75
    #eval_eps = 10

    parser = ArgumentParser()
    parser.add_argument('--trials', type=int, required=False)
    parser.add_argument('--train_eps', type=int, required=False)
    #parser.add_argument('--eval_eps', type=int, required=False)
    #parser.add_argument('--days_per_eps', type=int, required=False)
    #parser.add_argument('--model_param', type=str, required=False)
    #parser.add_argument('--reward_type', type=str, required=False)
    #parser.add_argument('--force_linear_model', dest='force_linear_model', action='store_true')
    #parser.set_defaults(force_linear_model=False)
    parser.add_argument('--experiment_name', type=str, required=True)
    args = parser.parse_args()

    mlflow.set_tracking_uri(str(mlflow_dir))

    # set Mlflow experiment name
    mlflow.set_experiment(args.experiment_name)

    optuna_db = 'MORLoptuna.db'
    
    # set Optuna study
    study = optuna.create_study(study_name=args.experiment_name,
                                direction='maximize',
                                load_if_exists=True,
                                storage=f'sqlite:///{optuna_db}')

    # Wrap the objective inside a lambda and call objective inside it
    # Nice trick taken from https://www.kaggle.com/general/261870
    #func = lambda trial: objective(trial, force_linear_model=args.force_linear_model, train_eps=args.train_eps, eval_eps=args.eval_eps, days_per_eps=args.days_per_eps, reward_type=args.reward_type)
    
    func = lambda trial: objective(trial, train_eps=args.train_eps)

    # run Optuna
    study.optimize(func, n_trials=args.trials)



    

