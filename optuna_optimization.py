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
    force_linear_model: bool = False,
    model_param: str = 'ppo'
) -> Dict:

    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    discount_factor = trial.suggest_categorical("discount_factor", [0.9, 0.95, 0.99])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    #memory_size = trial.suggest_categorical("memory_size", [int(1e4), int(5e4), int(1e5)])

    # we update the main model parameters every 'freq_steps_train' steps
    freq_steps_train = trial.suggest_categorical('freq_steps_train', [1, 4, 8, 16, 128]) #8, 16, 128, 256

    # we update the target model parameters every 'freq_steps_update_target' steps
    freq_steps_update_target = trial.suggest_categorical('freq_steps_update_target', [2, 10, 100]) #10, 100, 1000

    # minimum memory size we want before we start training
    # e.g.    0 --> start training right away.
    # e.g 1,000 --> start training when there are at least 1,000 sample trajectories in the agent's memory
    n_steps_warm_up_memory = trial.suggest_categorical("n_steps_warm_up_memory", [30, 100, 500]) #1000, 5000

    # how many consecutive gradient descent steps to perform when we update the main model parameters
    n_gradient_steps = trial.suggest_categorical("n_gradient_steps", [-1, 1, 4, 16])

    # model architecture to approximate q values
    #if force_linear_model:
    #    # linear model
    #    nn_hidden_layers = None
    #else:
    #    # neural network hidden layers
    #    # nn_hidden_layers = trial.suggest_categorical("nn_hidden_layers", ["None", "[64, 64]", "[256, 256]"])
    #    # TODO: added to speed up iterations.
    #    nn_hidden_layers = trial.suggest_categorical("nn_hidden_layers", [[256, 256]])
    #    # nn_hidden_layers = {"None": None, "[64, 64]": [64, 64], "[256, 256]": [256, 256]}[nn_hidden_layers]

    # how large do we let the gradients grow before capping them?
    # Explosive gradients can be an issue and this hyper-parameters helps mitigate it.
    #max_grad_norm = trial.suggest_categorical("max_grad_norm", [1, 10])

    # should we scale the inputs before feeding them to the model?
    #normalize_state = trial.suggest_categorical('normalize_state', [True, False])

    # start value for the exploration rate
    #epsilon_start = trial.suggest_categorical("epsilon_start", [0.9])

    # final value for the exploration rate
    #epsilon_end = trial.suggest_uniform("epsilon_end", 0, 0.2)

    # for how many steps do we decrease epsilon from its starting value to
    # its final value `epsilon_end`
    #steps_epsilon_decay = trial.suggest_categorical("steps_epsilon_decay", [int(1e3), int(1e4), int(1e5)])
    
    # sigma to use in the action noise creation
    action_noise_sigma = trial.suggest_categorical("action_noise_sigma", [0, 0.1, 0.5, 1.0])

    ## -- For PPO
    # The number of steps to run for each environment per update
    n_steps = trial.suggest_categorical("n_steps", [100, 150, 200, 500])

    # Limit the KL divergence between updates, because the clipping is not enough to prevent large update
    target_kl = trial.suggest_categorical("target_kl", [0.01, 0.05, 0.1, 0.5])

    # Number of epoch when optimizing the surrogate loss
    n_epochs = trial.suggest_categorical("n_epochs", [2, 4, 8, 10])

    # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.93, 0.95, 0.97, 0.99])


    seed = trial.suggest_int('seed', 0, 2 ** 30 - 1)

    if model_param == 'td3':
        return {
            'learning_rate': learning_rate,
            'discount_factor': discount_factor,
            'batch_size': batch_size,
            #'memory_size': memory_size,
            'freq_steps_train': freq_steps_train,
            'freq_steps_update_target': freq_steps_update_target,
            'n_steps_warm_up_memory': n_steps_warm_up_memory,
            'n_gradient_steps': n_gradient_steps,
            #'nn_hidden_layers': nn_hidden_layers,
            #'max_grad_norm': max_grad_norm,
            #'normalize_state': normalize_state,
            #'epsilon_start': epsilon_start,
            #'epsilon_end': epsilon_end,
            #'steps_epsilon_decay': steps_epsilon_decay,
            'action_noise_sigma': action_noise_sigma,
            'seed': seed,
        }
    else:
        return {
            'learning_rate': learning_rate,
            'discount_factor': discount_factor,
            'batch_size': batch_size,
            'n_steps': n_steps,
            'target_kl': target_kl,
            'n_epochs': n_epochs,
            'gae_lambda': gae_lambda,
            #'freq_steps_train': freq_steps_train,
            #'freq_steps_update_target': freq_steps_update_target,
            #'n_steps_warm_up_memory': n_steps_warm_up_memory,
            #'n_gradient_steps': n_gradient_steps,
            #'action_noise_sigma': action_noise_sigma,
            'seed': seed,
        }

def objective(
    trial: optuna.trial.Trial,
    force_linear_model: bool = False,
    train_eps: int = 75,
    eval_eps: int = 10,
    days_per_eps: int = 30,
    model_param: str = 'ppo',
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
    
    tensorboard_dir = "./tensorboards/hyperparams/"+model_param+"/"+trial_name
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
                                       force_linear_model=force_linear_model,
                                       model_param=model_param)
        mlflow.log_params(trial.params)

        # fix seeds to ensure reproducible runs
        set_seed(env, args['seed'])

        # Load trained model
        print("Creating model...")

        n_actions = env.action_space.shape[-1]

        if model_param == 'td3':
            # Setting up model and hyperparameters
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=args['action_noise_sigma']*np.ones(n_actions))
            policy_kwargs = dict(net_arch=dict(pi=[64, 64], qf=[64, 64]))
            model = TD3("MlpPolicy", 
                        env, 
                        learning_rate=args['learning_rate'], 
                        device='cuda', 
                        gamma=args['discount_factor'],
                        batch_size=args['batch_size'],
                        train_freq=args['freq_steps_train'],
                        policy_kwargs=policy_kwargs, 
                        buffer_size=100, 
                        learning_starts=args['n_steps_warm_up_memory'], 
                        gradient_steps=args['n_gradient_steps'],
                        policy_delay=args['freq_steps_update_target'],
                        action_noise=action_noise, 
                        verbose=1, 
                        tensorboard_log=tensorboard_dir + agent_id
                    )
        else:
            policy_kwargs = dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])], activation_fn=th.nn.Tanh)
            model = PPO("MlpPolicy", 
                        env, 
                        learning_rate=args['learning_rate'],
                        gamma=args['discount_factor'],
                        batch_size=args['batch_size'], 
                        n_steps=args['n_steps'], 
                        target_kl=args['target_kl'], 
                        n_epochs=args['n_epochs'], 
                        gae_lambda=args['gae_lambda'], 
                        seed=0, 
                        policy_kwargs=policy_kwargs,
                        verbose=1,
                        tensorboard_log=tensorboard_dir + agent_id
                    )

    
        model.set_env(env)

        # create agent object
        #agent = QAgent(
            #env,
            #learning_rate=args['learning_rate'],
            #discount_factor=args['discount_factor'],
            #batch_size=args['batch_size'],
            #memory_size=args['memory_size'],
            #freq_steps_train=args['freq_steps_train'],
            #freq_steps_update_target=args['freq_steps_update_target'],
            #n_steps_warm_up_memory=args['n_steps_warm_up_memory'],
            #n_gradient_steps=args['n_gradient_steps'],
            #nn_hidden_layers=args['nn_hidden_layers'], Gæti kannski bætt inn í policy_kwargs
            #max_grad_norm=args['max_grad_norm'],
            #normalize_state=args['normalize_state'],

            #log_dir=TENSORBOARD_LOG_DIR / env_name / agent_id
        #)
        
        print("Training agent...")
        
        # train loop
        model.learn(total_timesteps=timesteps, tb_log_name="train_run", reset_num_timesteps=False)
        #train(agent,
        #      env,
        #      n_episodes=n_episodes_to_train,
        #      log_dir=TENSORBOARD_LOG_DIR / env_name / agent_id)

        model.save(agent_dir + agent_id)
        #agent.save_to_disk(SAVED_AGENTS_DIR / env_name / agent_id)

        print("Evaluating agent...")
        ep_rews, ep_lens = evaluate_policy(model, 
                                           env, 
                                           n_eval_episodes=eval_eps, 
                                           return_episode_rewards=True, 
                                           deterministic=True) # test if I can get tt, sw etc. from env after eval
        ep_rews = np.array(ep_rews)
        mean_reward = np.mean(ep_rews)
        std_reward = np.std(ep_rews)
        
        
        # evaluate its performance
        #rewards, steps = evaluate(agent, env, n_episodes=1000, epsilon=0.00)
        #mean_reward = np.mean(rewards)
        #std_reward = np.std(rewards)
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
    parser.add_argument('--model_param', type=str, required=False)
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
    
    func = lambda trial: objective(trial, train_eps=args.train_eps, eval_eps=args.eval_eps, days_per_eps=args.days_per_eps, reward_type=args.reward_type, model_param=args.model_param)

    # run Optuna
    study.optimize(func, n_trials=args.trials)



    

