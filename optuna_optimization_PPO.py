import os
import mlflow
import optuna
import numpy as np
from argparse import ArgumentParser
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

from oop_simulation_script import Simulation, CommuteEnv


def make_env(seed,reward_type,days_per_eps):
    env = Monitor(CommuteEnv(reward_type,days_per_eps))
    #env.seed(seed)
    return env

def sample_hyperparameters(
    trial: optuna.trial.Trial,
    force_linear_model: bool = False
) -> Dict:

    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    discount_factor = trial.suggest_categorical("discount_factor", [0.95, 0.99])
    batch_size = trial.suggest_categorical("batch_size", [32, 64])

    # Limit the KL divergence between updates, because the clipping is not enough to prevent large update
    target_kl = trial.suggest_categorical("target_kl", [None, 0.05, 0.1, 0.5])

    # Number of epoch when optimizing the surrogate loss
    n_epochs = trial.suggest_categorical("n_epochs", [2, 6, 10])

    # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.93, 0.95, 0.99])

    # Entropy coefficient - for exploration vs exploitation
    ent_coef = trial.suggest_categorical("ent_coef", [0.0, 0.05, 0.1, 0.2])

    # neural network hidden layers
    nn_hidden_layers = trial.suggest_categorical("nn_hidden_layers", ["[64, 64]", "[64, 32]", "[32, 32]"])
    nn_hidden_layers = {"[64, 64]": [64, 64], "[64, 32]": [64, 32], "[32, 32]": [32, 32]}[nn_hidden_layers]

    share_hidden_layers = trial.suggest_categorical("share_hidden_layers", [True, False])


    return {
        'learning_rate': learning_rate,
        'discount_factor': discount_factor,
        'batch_size': batch_size,
        'target_kl': target_kl,
        'n_epochs': n_epochs,
        'gae_lambda': gae_lambda,
        'ent_coef': ent_coef,
        'nn_hidden_layers': nn_hidden_layers,
        'share_hidden_layers': share_hidden_layers,
    }

def objective(
    trial: optuna.trial.Trial,
    force_linear_model: bool = False,
    train_eps: int = 75,
    eval_eps: int = 10,
    days_per_eps: int = 30,
    reward_type: str = "sw" # tt: Travel Time    sw: Social Welfare
) -> float:
    """
    Samples hyperparameters, trains, and evaluates the RL agent.
    It outputs the average reward on 1,000 episodes.
    """

    env = make_env(10,reward_type,days_per_eps)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=30) 
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    timesteps = train_eps*days_per_eps

    trial_name = 'trial1'
    
    tensorboard_dir = "./tensorboards/hyperparams/PPO/"+trial_name
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    with mlflow.start_run():

        agent_dir = "local_models/optuna/"+trial_name
        env_name = "traffic-env-v0"
        # generate unique agent_id
        agent_id = get_agent_id(agent_dir, trial_name)
        
        mlflow.log_param('agent_id', agent_id)

        # hyper-parameters
        args = sample_hyperparameters(trial,
                                       force_linear_model=force_linear_model)
        mlflow.log_params(trial.params)

        # Load trained model
        print("Creating model...")

        if args['share_hidden_layers']:
            policy_kwargs = dict(net_arch=args['nn_hidden_layers'], activation_fn=th.nn.Tanh)
        else:
            policy_kwargs = dict(net_arch=[dict(pi=args['nn_hidden_layers'], vf=args['nn_hidden_layers'])], activation_fn=th.nn.Tanh)
        
        model = PPO("MlpPolicy", 
                    env, 
                    learning_rate=args['learning_rate'],
                    gamma=args['discount_factor'],
                    batch_size=args['batch_size'], 
                    n_steps=200, 
                    target_kl=args['target_kl'], 
                    n_epochs=args['n_epochs'], 
                    gae_lambda=args['gae_lambda'],
                    ent_coef=args['ent_coef'],
                    seed=0, 
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    tensorboard_log=tensorboard_dir + agent_id
                )

    
        model.set_env(env)

        
        print("Training agent...")
        # train loop
        model.learn(total_timesteps=timesteps, tb_log_name="PPO", reset_num_timesteps=False)
        model.save(agent_dir + agent_id)


        print("Evaluating agent...")
        ep_rews, ep_lens = evaluate_policy(model, 
                                           env, 
                                           n_eval_episodes=eval_eps, 
                                           return_episode_rewards=True, 
                                           deterministic=True) # test if I can get tt, sw etc. from env after eval
        ep_rews = np.array(ep_rews)
        mean_reward = np.mean(ep_rews)
        std_reward = np.std(ep_rews)
        
        mlflow.log_metric('mean_reward', mean_reward)
        mlflow.log_metric('std_reward', std_reward)

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
    parser.add_argument('--eval_eps', type=int, required=False)
    parser.add_argument('--days_per_eps', type=int, required=False)
    parser.add_argument('--reward_type', type=str, required=False)
    parser.add_argument('--force_linear_model', dest='force_linear_model', action='store_true')
    parser.set_defaults(force_linear_model=False)
    parser.add_argument('--experiment_name', type=str, required=True)
    args = parser.parse_args()

    mlflow.set_tracking_uri(str(mlflow_dir))

    # set Mlflow experiment name
    mlflow.set_experiment(args.experiment_name)

    optuna_db = 'optuna.db'
    
    # set Optuna study
    study = optuna.create_study(study_name=args.experiment_name,
                                direction='maximize',
                                load_if_exists=True,
                                storage=f'sqlite:///{optuna_db}')

    # Wrap the objective inside a lambda and call objective inside it
    # Nice trick taken from https://www.kaggle.com/general/261870
    #func = lambda trial: objective(trial, force_linear_model=args.force_linear_model, train_eps=args.train_eps, eval_eps=args.eval_eps, days_per_eps=args.days_per_eps, reward_type=args.reward_type)
    
    func = lambda trial: objective(trial, train_eps=args.train_eps, eval_eps=args.eval_eps, days_per_eps=args.days_per_eps, reward_type=args.reward_type)

    # run Optuna
    study.optimize(func, n_trials=args.trials)



    

