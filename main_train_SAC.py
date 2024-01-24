#IMPORTS

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
from stable_baselines3 import SAC

from oop_simulation_script import Simulation, CommuteEnv
    
def make_env(seed,reward_type,days_per_eps):
    def _init():
        env = Monitor(CommuteEnv(reward_type,days_per_eps))
        #env.seed(seed)
        return env
    return _init

def main():
    # Set up environment
    reward_type = "gini+sw" # tt: Travel Time    sw: Social Welfare
    days_per_eps = 30
    eps=300 #for naming
    train_eps=300 #for timestep calc

    model_param = sys.argv[1] #Fetching model parameter
    #folder="SAC/random_initial_toll/scaled_reward/buffsize300/with_PPO_params/"
    folder="SAC/random_initial_toll/scaled_reward/linear_scalarization/discount_bottom10perc/0.25gini+0.75sw/disc01/"

    """
    model_dir = "./local_models/"+str(folder)+reward_type+"_"+str(eps)+str(model_param)+"_tollprice"

    tensorboard_name = 'SAC_LS-75g_discBottom10perc_disc03'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)


    stats_dir = "./local_stats/training/"+str(folder)+reward_type+"_"+str(eps)+str(model_param) + "/"
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)

    print("Stats Directory:")
    print("\t",stats_dir)
    
    env = make_env(10,reward_type,days_per_eps)() # Seed as argument
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=30) 
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Load trained model
    print("Creating model...")

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1*np.ones(n_actions))

    policy_kwargs = dict(net_arch=dict(pi=[64, 64], qf=[64, 64]))
    
    # Creating new agent
    model = SAC("MlpPolicy", env, 
        policy_kwargs=policy_kwargs, 
        verbose=1, 
        buffer_size = 300, 
        learning_rate=1e-4,
        batch_size=150,
        train_freq=150, #n_steps
        #target_update_interval=150, 
        gradient_steps=80, #n_epochs
        #ent_coef='auto_0.2',
        #action_noise=action_noise,
        #use_sde=True,
        #learning_starts=150,
        tensorboard_log="./tensorboard_logs/SAC_sw/"
    )
    
    # Loading existing agent
    #model = SAC.load("./local_models/SAC/random_initial_toll/scaled_reward/buffsize300/with_PPO_params/sw_300sac_tollprice.zip", device='cuda')

    model.set_env(env)

    print("Model created")
    
    #with open(stats_dir + str(eps) + str(model_param) + "_params.txt", "w") as fp:
    #    json.dump(args, fp)  # encode dict into JSON
    
    timesteps = train_eps*days_per_eps
    # Training the model
    model.learn(total_timesteps=timesteps, tb_log_name=tensorboard_name, reset_num_timesteps=False)

    model.save(model_dir) # Saving the trained model

    # Saving the observed stats from training the model


    np.save(stats_dir + str(eps) + str(model_param) + "_tt.npy", env.env_method('get_tt')[0])
    np.save(stats_dir + str(eps) + str(model_param) + "_sw.npy", env.env_method('get_sw')[0])
    np.save(stats_dir + str(eps) + str(model_param) + "_cs.npy", env.env_method('get_cs')[0])
    #np.save(stats_dir + str(eps) + str(model_param) + "_gc.npy", env.env_method('get_gc')[0])
    #np.save(stats_dir + str(eps) + str(model_param) + "_tc.npy", env.env_method('get_tc')[0])
    #np.save(stats_dir + str(eps) + str(model_param) + "_ttcs.npy", env.env_method('get_ttcs')[0])
    np.save(stats_dir + str(eps) + str(model_param) + "_ind_cs.npy", env.env_method('get_individual_cs')[0])
    #np.save(stats_dir + str(eps) + str(model_param) + "_ind_sw.npy", env.env_method('get_individual_sw')[0])
    #np.save(stats_dir + str(eps) + str(model_param) + "_ind_inc.npy", env.env_method('get_income')[0])
    np.save(stats_dir + str(eps) + str(model_param) + "_toll.npy", env.env_method('get_toll')[0])
    np.save(stats_dir + str(eps) + str(model_param) + "_toll_profile.npy", env.env_method('get_all_toll_profiles')[0])
    np.save(stats_dir + str(eps) + str(model_param) + "_actions.npy", env.env_method('get_actions')[0])
    #np.save(stats_dir + str(eps) + str(model_param) + "_user_flow.npy", env.env_method('get_user_flow')[0])
    np.save(stats_dir + str(eps) + str(model_param) + "_abs_actions.npy", env.env_method('get_abs_actions')[0])
    np.save(stats_dir + str(eps) + str(model_param) + "_gini.npy", env.env_method('get_gini')[0])

    """
    # EVAL
    folder=str(folder)+reward_type

    stats_dir = "./local_stats/evaluation/NEW_SAC_29nov/"+str(folder)+"_"+str(eps)+str(model_param)+"/"
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)

    model = SAC.load("./local_models/"+folder+"_"+str(eps)+"sac_tollprice.zip")

    eval_env = make_env(10,reward_type,days_per_eps)() # Seed as argument
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecFrameStack(eval_env, n_stack=30) 
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)

    model.set_env(eval_env)

    obs = eval_env.reset()

    ep_rews, ep_lens = evaluate_policy(model, eval_env, n_eval_episodes=10, return_episode_rewards=True, deterministic=True) # test if I can get tt, sw etc. from env after eval
    ep_rews = np.array(ep_rews)
    mean_reward = np.mean(ep_rews)
    std_reward = np.std(ep_rews)

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
    np.save(stats_dir+str(eps)+str(model_param) +"_gini.npy", env.env_method('get_gini')[0])
    np.save(stats_dir+str(eps)+str(model_param) +"_ind_toll.npy", env.env_method('get_toll_paid')[0])


if __name__ == "__main__":
    main()
