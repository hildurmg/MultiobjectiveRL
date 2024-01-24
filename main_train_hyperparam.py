#IMPORTS
import random
import csv
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as th
from stable_baselines3 import PPO, DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

from oop_simulation_script import Simulation, CommuteEnv
    
def make_env(seed,reward_type,days_per_eps):
    def _init():
        env = Monitor(CommuteEnv(reward_type,days_per_eps))
        #env.seed(seed)
        return env
    return _init

def main():
    # Set up environment
    reward_type = "sw" # tt: Travel Time    sw: Social Welfare
    days_per_eps = 30
    eps = 75
    
    env = make_env(10,reward_type,days_per_eps)() # Seed as argument
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=30) 
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    model_param = sys.argv[1] #Fetching model parameter
    discount_param_tr = sys.argv[2] #Fetching discount parameter
    discount_param_ev = sys.argv[3] #Fetching discount parameter
    
    # LCreate model
    print("Creating model")
    
    if model_param == 'td3':
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=1*np.ones(n_actions))

        # Setting up model and hyperparameters
        policy_kwargs = dict(net_arch=dict(pi=[64, 64], qf=[64, 64]))

        model = TD3("MlpPolicy", env,learning_rate=0.0001, device='cuda', policy_kwargs=policy_kwargs, buffer_size=150, learning_starts=30, batch_size=100, action_noise=action_noise, verbose=1, tensorboard_log="./tensorboards/TD3/")
    else:
        policy_kwargs = dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])], activation_fn=th.nn.Tanh)
        model = PPO("MlpPolicy", env, learning_rate=0.0001, n_steps=150, verbose=1, batch_size=150, target_kl=0.05, n_epochs=80, gae_lambda=0.97, seed=0, policy_kwargs=policy_kwargs, tensorboard_log="./tensorboard_logs/PPO_sw/")
    
            
    model.set_env(env)

    print("Model created")
    obs = env.reset()

    # Training the model
    timesteps = eps*days_per_eps
    model.learn(total_timesteps=timesteps, tb_log_name="train_run", reset_num_timesteps=False)

    print("Saving stats...")
    train_path = "./local_stats/training/hyperparam_tuning/action_noise/sigma1/"+str(eps)+model_param
    eval_path = "./local_stats/evaluation/hyperparam_tuning/action_noise/sigma1/"+str(eps)+model_param


    np.save(train_path+"_sw.npy", env.env_method('get_sw')[0])
    np.save(train_path+"_tt.npy", env.env_method('get_tt')[0])
    np.save(train_path+"_cs.npy", env.env_method('get_cs')[0])
    np.save(train_path+"_gc.npy", env.env_method('get_gc')[0])
    np.save(train_path+"_tc.npy", env.env_method('get_tc')[0])
    np.save(train_path+"_ttcs.npy", env.env_method('get_ttcs')[0])
    np.save(train_path+"_ind_cs.npy", env.env_method('get_individual_cs')[0])
    np.save(train_path+"_ind_sw.npy", env.env_method('get_individual_sw')[0])
    np.save(train_path+"_ind_inc.npy", env.env_method('get_income')[0])

    model.save("./local_models/hyperparam_tuning/"+str(eps)+str(model_param)+"_sigma1_tollprice") # Saving the trained model

    # Evaluate trained model and save stats
    print("Evaluating model")
    ep_rews, ep_lens = evaluate_policy(model, env, n_eval_episodes=10, return_episode_rewards=True, deterministic=True) # test if I can get tt, sw etc. from env after eval
    ep_rews = np.array(ep_rews)
    mean_reward = np.mean(ep_rews)
    std_reward = np.std(ep_rews)
    print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print("Saving eval stats")

    np.save(eval_path+"_rew.npy", ep_rews)
    np.save(eval_path+"_sw.npy", env.env_method('get_sw')[0])
    np.save(eval_path+"_tt.npy", env.env_method('get_tt')[0])
    np.save(eval_path+"_cs.npy", env.env_method('get_cs')[0])
    np.save(eval_path+"_gc.npy", env.env_method('get_gc')[0])
    np.save(eval_path+"_tc.npy", env.env_method('get_tc')[0])
    np.save(eval_path+"_ttcs.npy", env.env_method('get_ttcs')[0])
    np.save(eval_path+"_ind_cs.npy", env.env_method('get_individual_cs')[0])
    np.save(eval_path+"_ind_sw.npy", env.env_method('get_individual_sw')[0])
    np.save(eval_path+"_ind_inc.npy", env.env_method('get_income')[0])


if __name__ == "__main__":
    main()
