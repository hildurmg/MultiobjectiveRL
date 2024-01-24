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
    discount_factor = trial.suggest_categorical("discount_factor", [0.9, 0.95, 0.99])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    #memory_size = trial.suggest_categorical("memory_size", [int(1e4), int(5e4), int(1e5)])

    # we update the target model parameters every 'freq_steps_update_target' steps
    policy_delay = trial.suggest_categorical('policy_delay', [2, 10]) #10, 100, 1000

    # how many consecutive gradient descent steps to perform when we update the main model parameters
    n_gradient_steps = trial.suggest_categorical("n_gradient_steps", [-1, 1, 4])


    # neural network hidden layers
    nn_hidden_layers = trial.suggest_categorical("nn_hidden_layers", ["[64, 64]", "[64, 32]", "[32, 32]"])
    nn_hidden_layers = {"[64, 64]": [64, 64], "[64, 32]": [64, 32], "[32, 32]": [32, 32]}[nn_hidden_layers]

    share_hidden_layers = trial.suggest_categorical("share_hidden_layers", [True, False])

    # sigma to use in the action noise creation
    action_noise_sigma = trial.suggest_categorical("action_noise_sigma", [0, 0.1, 0.5, 1.0])

    return {
        'learning_rate': learning_rate,
        'discount_factor': discount_factor,
        'batch_size': batch_size,
        'policy_delay': policy_delay,
        'n_gradient_steps': n_gradient_steps,
        'nn_hidden_layers': nn_hidden_layers,
        'action_noise_sigma': action_noise_sigma,
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
    
    tensorboard_dir = "./tensorboards/hyperparams/TD3/"+trial_name
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

        n_actions = env.action_space.shape[-1]

        # Setting up model and hyperparameters
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=args['action_noise_sigma']*np.ones(n_actions))

        if args['share_hidden_layers']:
            policy_kwargs = dict(net_arch=args['nn_hidden_layers'])
        else:
            policy_kwargs = dict(net_arch=dict(pi=args['nn_hidden_layers'], qf=args['nn_hidden_layers']))

        
        model = TD3("MlpPolicy", 
                    env, 
                    learning_rate=args['learning_rate'], 
                    device='cuda', 
                    gamma=args['discount_factor'],
                    batch_size=args['batch_size'],
                    policy_kwargs=policy_kwargs, 
                    buffer_size=100, 
                    learning_starts=30, 
                    gradient_steps=args['n_gradient_steps'],
                    policy_delay=args['policy_delay'],
                    action_noise=action_noise, 
                    verbose=1, 
                    tensorboard_log=tensorboard_dir + agent_id
                )

    
        model.set_env(env)
        
        print("Training agent...")
        
        # train loop
        model.learn(total_timesteps=timesteps, tb_log_name="TD3", reset_num_timesteps=False)

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



    

