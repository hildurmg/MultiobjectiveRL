#IMPORTS
import random
import csv
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as th
from stable_baselines3 import PPO, DDPG, TD3
#from stable_baselines3.common.noise import NormalActionNoise
#from stable_baselines3.common.env_util import SubprocVecEnv, DummyVecEnv
#from stable_baselines3.common.monitor import Monitor
#from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack
#from stable_baselines3.common.evaluation import evaluate_policy
#from stable_baselines3.common.callbacks import EvalCallback

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
    
    eval_env = make_env(10,reward_type,days_per_eps)() # Seed as argument
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecFrameStack(eval_env, n_stack=30) 
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)

    model_param = sys.argv[1] #Fetching model parameter
    discount_param = sys.argv[2] #Fetching discount parameter
    
    # Load trained model
    print("Starting to load model")
    if model_param == 'ppo':
        model = PPO.load("./models/Anders Models/300ppo_tollprice.zip")
    elif model_param == 'td3':
        model = TD3.load("./models/Anders Models/300td3_tollprice.zip")
        
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
    print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    print("Saving stats...")
    np.save("./stats/fixed_discounts_individual/"+str(discount_param)+"/300"+model_param+"_rew.npy", ep_rews)
    np.save("./stats/fixed_discounts_individual/"+str(discount_param)+"/300"+model_param+"_sw.npy", eval_env.env_method('get_sw')[0])
    np.save("./stats/fixed_discounts_individual/"+str(discount_param)+"/300"+model_param+"_tt.npy", eval_env.env_method('get_tt')[0])
    np.save("./stats/fixed_discounts_individual/"+str(discount_param)+"/300"+model_param+"_cs.npy", eval_env.env_method('get_cs')[0])
    np.save("./stats/fixed_discounts_individual/"+str(discount_param)+"/300"+model_param+"_gc.npy", eval_env.env_method('get_gc')[0])
    np.save("./stats/fixed_discounts_individual/"+str(discount_param)+"/300"+model_param+"_tc.npy", eval_env.env_method('get_tc')[0])
    np.save("./stats/fixed_discounts_individual/"+str(discount_param)+"/300"+model_param+"_ttcs.npy", eval_env.env_method('get_ttcs')[0])
    np.save("./stats/fixed_discounts_individual/"+str(discount_param)+"/300"+model_param+"_ind_cs.npy", eval_env.env_method('get_individual_cs')[0])
    np.save("./stats/fixed_discounts_individual/"+str(discount_param)+"/300"+model_param+"_ind_sw.npy", eval_env.env_method('get_individual_sw')[0])
    np.save("./stats/fixed_discounts_individual/"+str(discount_param)+"/300"+model_param+"_ind_inc.npy", eval_env.env_method('get_income')[0])

if __name__ == "__main__":
    main()
