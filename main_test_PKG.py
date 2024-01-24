#IMPORTS
import random
import csv
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as th
from stable_baselines3 import PPO, DDPG, TD3, SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import time
import os

from oop_simulation_script import Simulation, CommuteEnv
    
def make_env(seed,reward_type,days_per_eps):
    def _init():
        env = Monitor(CommuteEnv(reward_type,days_per_eps))
        #env.seed(seed)
        return env
    return _init

def main():
    # Set up environment
    initial_time = time.perf_counter()
    
    
    reward_type = "gini" # tt: Travel Time    sw: Social Welfare
    days_per_eps = 30
    eps = 300
    
    eval_env = make_env(10,reward_type,days_per_eps)() # Seed as argument
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecFrameStack(eval_env, n_stack=30) 
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)

    model_param = sys.argv[1] #Fetching model parameter
    discount_param = sys.argv[2] #Fetching discount parameter

    #folder="simpler_env/linear_scalarization/0.75gini+0.25sw/"+reward_type
    folder="simpler_env/discount_bottom10perc/disc03/"+reward_type
    #folder="simpler_env/linear_scalarization/discount_bottom10perc/0.75gini+0.25sw/disc03/"+reward_type

    stats_dir = "./local_stats/evaluation/NEW_29nov/"+str(folder)+"_"+str(eps)+str(model_param)+"/"
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    
    # Load trained model
    print("Starting to load model")
    if model_param == 'ppo':
        model = PPO.load("./local_models/"+folder+"_"+str(eps)+"ppo_tollprice.zip")
    elif model_param == 'td3':
        model = TD3.load("./local_models/"+folder+"_"+str(eps)+"td3_tollprice.zip")
    elif model_param == 'sac':
        model = SAC.load("./local_models/"+folder+"_"+str(eps)+"sac_tollprice.zip")

    model.set_env(eval_env)

    print("Model loaded")
    # # Test that the model is able to take steps in the environment
    obs = eval_env.reset()

    # for i in range(3):
    #     print("Episode: " + str(i))
    #     action, _states = model.predict(obs, deterministic=True)
    #     #action = [eval_env.action_space.sample()]
    #     obs, rewards, dones, info = eval_env.step(action)


    #obs = eval_env.reset()
    # Evaluate trained model and save stats
    ep_rews, ep_lens = evaluate_policy(model, eval_env, n_eval_episodes=10, return_episode_rewards=True, deterministic=True) # test if I can get tt, sw etc. from env after eval
    ep_rews = np.array(ep_rews)
    mean_reward = np.mean(ep_rews)
    std_reward = np.std(ep_rews)
    print(ep_rews)
    print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    time_difference = time.perf_counter() - initial_time
    print("TIME: " + str(time_difference))

    print("Saving stats...")
    np.save(stats_dir+str(eps)+model_param+"_rew.npy", ep_rews)
    np.save(stats_dir+str(eps)+model_param+"_sw.npy", eval_env.env_method('get_sw')[0])
    np.save(stats_dir+str(eps)+model_param+"_tt.npy", eval_env.env_method('get_tt')[0])
    np.save(stats_dir+str(eps)+model_param+"_cs.npy", eval_env.env_method('get_cs')[0])
    np.save(stats_dir+str(eps)+model_param+"_gc.npy", eval_env.env_method('get_gc')[0])
    np.save(stats_dir+str(eps)+model_param+"_tc.npy", eval_env.env_method('get_tc')[0])
    np.save(stats_dir+str(eps)+model_param+"_ttcs.npy", eval_env.env_method('get_ttcs')[0])
    np.save(stats_dir+str(eps)+model_param+"_ind_cs.npy", eval_env.env_method('get_individual_cs')[0])
    #np.save(stats_dir+str(eps)+model_param+"_ind_sw.npy", eval_env.env_method('get_individual_sw')[0])
    np.save(stats_dir+str(eps)+model_param+"_ind_inc.npy", eval_env.env_method('get_income')[0])
    np.save(stats_dir+str(eps)+model_param+"_toll.npy", eval_env.env_method('get_toll')[0])
    np.save(stats_dir+str(eps)+model_param+"_toll_profile.npy", eval_env.env_method('get_all_toll_profiles')[0])
    np.save(stats_dir+str(eps)+model_param+"_user_flow.npy", eval_env.env_method('get_user_flow')[0])
    np.save(stats_dir+str(eps)+model_param+"_actions.npy", eval_env.env_method('get_actions')[0])
    #np.save(stats_dir+str(eps)+model_param+"_accumulation.npy", eval_env.env_method('get_accumulation')[0])
    #np.save(stats_dir+str(eps)+model_param+"_event_list.npy", eval_env.env_method('get_event_list')[0])
    np.save(stats_dir+str(eps)+model_param+"_abs_actions.npy", eval_env.env_method('get_abs_actions')[0])
    np.save(stats_dir+str(eps)+model_param+"_gini.npy", eval_env.env_method('get_gini')[0])
    np.save(stats_dir+str(eps)+model_param+"_discount.npy", eval_env.env_method('get_discount')[0])
    np.save(stats_dir+str(eps)+model_param+"_ind_toll.npy", eval_env.env_method('get_toll_paid')[0])
if __name__ == "__main__":
    main()
