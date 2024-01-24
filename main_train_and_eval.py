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
import os
import json

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
    eps=100 #for naming
    train_eps=100 #for timestep calc

    model_param = sys.argv[1] #Fetching model parameter
    folder="observation_space/with_tt/"

    model_dir = "./local_models/"+folder+reward_type+"_"+str(eps)+str(model_param)+"_tollprice"


    if not os.path.exists(model_dir):
        os.makedirs(model_dir)


    stats_dir = "./local_stats/training/"+str(folder)+reward_type+"_"+str(eps)+str(model_param) + "/"
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)

    eval_dir = "./local_stats/"+str(folder)+reward_type+"_"+str(eps)+str(model_param) + "/"
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    print("Stats Directory:")
    print("\t",stats_dir)
    
    env = make_env(10,reward_type,days_per_eps)() # Seed as argument
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=30) 
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Load trained model
    print("Creating model...")

    # Setting up model and hyperparameters
    if model_param == 'td3':
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1*np.ones(n_actions))
        policy_kwargs = dict(net_arch=dict(pi=[64, 64], qf=[64, 64]))
        model = TD3("MlpPolicy", env, learning_rate=0.0001, device='cuda', policy_kwargs=policy_kwargs, buffer_size=100, learning_starts=30, batch_size=30, action_noise=action_noise, verbose=1, tensorboard_log="./tensorboards/TD3/")

        #args = {
        #    'learning_rate': 0.0001114,
        #    'discount_factor': 0.9,
        #    'batch_size': 64,
        #    'freq_steps_train': 16,
        #    'freq_steps_update_target': 2,
        #    'n_steps_warm_up_memory': 30,
        #    'n_gradient_steps': 1,
        #    'action_noise_sigma': 1.0,
        #}
#
        #action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=args['action_noise_sigma']*np.ones(n_actions))
        #policy_kwargs = dict(net_arch=dict(pi=[64, 64], qf=[64, 64]))
        #model = TD3("MlpPolicy", 
        #            env, 
        #            learning_rate=args['learning_rate'], 
        #            device='cuda', 
        #            gamma=args['discount_factor'],
        #            batch_size=args['batch_size'],
        #            train_freq=args['freq_steps_train'],
        #            policy_kwargs=policy_kwargs, 
        #            buffer_size=100, 
        #            learning_starts=args['n_steps_warm_up_memory'], 
        #            gradient_steps=args['n_gradient_steps'],
        #            policy_delay=args['freq_steps_update_target'],
        #            action_noise=action_noise, 
        #            verbose=1, 
        #            tensorboard_log="./tensorboards/TD3/"
        #        )

    else:
        policy_kwargs = dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])], activation_fn=th.nn.Tanh)
        model = PPO("MlpPolicy", env, learning_rate=0.0001, n_steps=150, verbose=1, batch_size=150, target_kl=0.05, n_epochs=80, gae_lambda=0.97, seed=0, policy_kwargs=policy_kwargs, tensorboard_log="./tensorboard_logs/PPO_sw/")
        
        #args = {
        #    'learning_rate': 0.003514,
        #    'discount_factor': 0.95,
        #    'batch_size': 32,
        #    'n_steps': 200,
        #    'target_kl': 0.05,
        #    'n_epochs': 4,
        #    'gae_lambda': 0.97,
        #}
#
        #policy_kwargs = dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])], activation_fn=th.nn.Tanh)
        #model = PPO("MlpPolicy", 
        #            env, 
        #            learning_rate=args['learning_rate'],
        #            gamma=args['discount_factor'],
        #            batch_size=args['batch_size'], 
        #            n_steps=args['n_steps'], 
        #            target_kl=args['target_kl'], 
        #            n_epochs=args['n_epochs'], 
        #            gae_lambda=args['gae_lambda'], 
        #            seed=0, 
        #            policy_kwargs=policy_kwargs,
        #            verbose=1,
        #            tensorboard_log="./tensorboard_logs/PPO_sw/"
        #        )


    #if model_param == 'td3':
    #    model = TD3.load("./local_models/observation_space/with_actions/tt_75td3_tollprice.zip", device='cuda')
    #else:
    #    model = PPO.load("./local_models/observation_space/with_actions/tt_75ppo_tollprice.zip", device='cuda')
    
    model.set_env(env)

    print("Model created")
    
    #with open(stats_dir + str(eps) + str(model_param) + "_params.txt", "w") as fp:
    #    json.dump(args, fp)  # encode dict into JSON
    
    timesteps = train_eps*days_per_eps
    # Training the model
    model.learn(total_timesteps=timesteps, tb_log_name="train_run", reset_num_timesteps=False)

    model.save(model_dir) # Saving the trained model

    # Saving the observed stats from training the model


    np.save(stats_dir + str(eps) + str(model_param) + "_tt.npy", env.env_method('get_tt')[0])
    np.save(stats_dir + str(eps) + str(model_param) + "_sw.npy", env.env_method('get_sw')[0])
    np.save(stats_dir + str(eps) + str(model_param) + "_cs.npy", env.env_method('get_cs')[0])
    np.save(stats_dir + str(eps) + str(model_param) + "_gc.npy", env.env_method('get_gc')[0])
    np.save(stats_dir + str(eps) + str(model_param) + "_tc.npy", env.env_method('get_tc')[0])
    np.save(stats_dir + str(eps) + str(model_param) + "_ttcs.npy", env.env_method('get_ttcs')[0])
    np.save(stats_dir + str(eps) + str(model_param) + "_ind_cs.npy", env.env_method('get_individual_cs')[0])
    np.save(stats_dir + str(eps) + str(model_param) + "_ind_sw.npy", env.env_method('get_individual_sw')[0])
    np.save(stats_dir + str(eps) + str(model_param) + "_ind_inc.npy", env.env_method('get_income')[0])
    np.save(stats_dir + str(eps) + str(model_param) + "_toll.npy", env.env_method('get_toll')[0])
    np.save(stats_dir + str(eps) + str(model_param) + "_toll_profile.npy", env.env_method('get_all_toll_profiles')[0])
    np.save(stats_dir + str(eps) + str(model_param) + "_actions.npy", env.env_method('get_actions')[0])
    np.save(stats_dir + str(eps) + str(model_param) + "_user_flow.npy", env.env_method('get_user_flow')[0])

    print("Training stats saved!")
    print("Starting evaluation...")

    ep_rews, ep_lens = evaluate_policy(model, env, n_eval_episodes=10, return_episode_rewards=True, deterministic=True) # test if I can get tt, sw etc. from env after eval
    ep_rews = np.array(ep_rews)
    mean_reward = np.mean(ep_rews)
    std_reward = np.std(ep_rews)
    print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    print("Saving eval stats...")
    np.save(eval_dir + str(eps) + str(model_param) + "_rew.npy", ep_rews)
    np.save(eval_dir + str(eps) + str(model_param) + "_tt.npy", env.env_method('get_tt')[0])
    np.save(eval_dir + str(eps) + str(model_param) + "_sw.npy", env.env_method('get_sw')[0])
    np.save(eval_dir + str(eps) + str(model_param) + "_cs.npy", env.env_method('get_cs')[0])
    np.save(eval_dir + str(eps) + str(model_param) + "_gc.npy", env.env_method('get_gc')[0])
    np.save(eval_dir + str(eps) + str(model_param) + "_tc.npy", env.env_method('get_tc')[0])
    np.save(eval_dir + str(eps) + str(model_param) + "_ttcs.npy", env.env_method('get_ttcs')[0])
    np.save(eval_dir + str(eps) + str(model_param) + "_ind_cs.npy", env.env_method('get_individual_cs')[0])
    np.save(eval_dir + str(eps) + str(model_param) + "_ind_sw.npy", env.env_method('get_individual_sw')[0])
    np.save(eval_dir + str(eps) + str(model_param) + "_ind_inc.npy", env.env_method('get_income')[0])
    np.save(eval_dir + str(eps) + str(model_param) + "_toll.npy", env.env_method('get_toll')[0])
    np.save(eval_dir + str(eps) + str(model_param) + "_toll_profile.npy", env.env_method('get_all_toll_profiles')[0])
    np.save(eval_dir + str(eps) + str(model_param) + "_actions.npy", env.env_method('get_actions')[0])
    np.save(eval_dir + str(eps) + str(model_param) + "_user_flow.npy", env.env_method('get_user_flow')[0])
    np.save(eval_dir + str(eps) + str(model_param) + "_accumulation.npy", env.env_method('get_accumulation')[0])
    np.save(eval_dir + str(eps) + str(model_param) + "_event_list.npy", env.env_method('get_event_list')[0])
    np.save(eval_dir + str(eps) + str(model_param) + "_abs_actions.npy", env.env_method('get_abs_actions')[0])
    print("Eval stats saved!")



if __name__ == "__main__":
    main()
