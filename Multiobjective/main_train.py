#IMPORTS
import random
import csv
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
from morl_baselines.multi_policy.capql.capql import CAPQL
from gymnasium.envs.registration import register
import gymnasium
import os


from oop_simulation_script import Simulation, CommuteEnv
    
def make_env(seed,reward_type,days_per_eps):
    def _init():
        register(
            id="traffic-env-v0",
            entry_point="oop_simulation_script:CommuteEnv",
        )
        env = gymnasium.make("traffic-env-v0")
        #env = Monitor(CommuteEnv(reward_type,days_per_eps))
        #env.seed(seed)
        return env
    return _init

def main():
    # Set up environment
    reward_type = "sw" # tt: Travel Time    sw: Social Welfare
    days_per_eps = 30
    
    env = make_env(10,reward_type,days_per_eps)() # Seed as argument
    nr_timesteps = 6000
    #nr_timesteps = 10
    final_model_timesteps = 6000

    #stats_dir = "./local_stats/MultiObjective/test/"
    stats_dir = "./local_stats/MultiObjective/final_test/evaluation_2/30/"+str(final_model_timesteps)
    #stats_dir = "./local_stats/MultiObjective/simpler_environment/random_initial_toll/gini_cs+sw/"+str(final_model_timesteps)
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)

    #env = DummyVecEnv([lambda: env])
    #env = VecFrameStack(env, n_stack=30) 
    #env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Load trained model
    print("Creating model...")
    ref_point=np.array([0, -25])
    agent = CAPQL(
        env,
        learning_rate=1e-4,
        seed=15,
        net_arch=[64, 64],
        buffer_size=300,
        alpha=0.085,
        learning_starts=150,
        batch_size=150,
        gradient_updates = 80
    )
    #agent.load("./weights/CPQL_run_14_3.tar")
    agent.load("./weights/CPQL_run_30_2.tar")
    #agent.load("./weights/CPQLPerc_3000.tar")

    agent.evaluate_for_front(eval_env=env)

    # agent.train(
    #     total_timesteps=nr_timesteps,
    #     ref_point=ref_point,
    #     eval_env=env,
    #     eval_freq=nr_timesteps/10,
    #     filename="CPQL_run_25",
    #     num_eval_weights_for_front=15,
    #     num_eval_episodes_for_front=5,
    #     #num_eval_weights_for_front=1,
    #     #num_eval_episodes_for_front=1,
    #     train_freq=150
    # )
    # print(pf)

    

    # Execute a policy
    #target = np.array(pf.pop())
    #print(f"Tracking {target}")
    #reward = agent.track_policy(target, env=env)
    #print(f"Obtained {reward}")



    # Saving the observed stats from atraining the model
    np.save(stats_dir + "/" + str(final_model_timesteps) + "_tt.npy"    , env.get_tt())
    np.save(stats_dir + "/" + str(final_model_timesteps) + "_sw.npy"    , env.get_sw())
    np.save(stats_dir + "/" + str(final_model_timesteps) + "_cs.npy"    , env.get_cs())
    np.save(stats_dir + "/" + str(final_model_timesteps) + "_gc.npy"    , env.get_gc())
    np.save(stats_dir + "/" + str(final_model_timesteps) + "_tc.npy"    , env.get_tc())
    np.save(stats_dir + "/" + str(final_model_timesteps) + "_ttcs.npy"  , env.get_ttcs())
    np.save(stats_dir + "/" + str(final_model_timesteps) + "_discount.npy"  , env.get_discount())
    np.save(stats_dir + "/" + str(final_model_timesteps) + "_ind_cs.npy", env.get_individual_cs())
    np.save(stats_dir + "/" + str(final_model_timesteps) + "_ind_sw.npy", env.get_individual_sw())
    np.save(stats_dir + "/" + str(final_model_timesteps) + "_ind_inc.npy", env.get_income())
    np.save(stats_dir + "/" + str(final_model_timesteps) + "_toll.npy", env.get_toll())
    np.save(stats_dir + "/" + str(final_model_timesteps) + "_toll_profile.npy", env.get_all_toll_profiles())
    np.save(stats_dir + "/" + str(final_model_timesteps) + "_user_flow.npy", env.get_user_flow())
    np.save(stats_dir + "/" + str(final_model_timesteps) + "_actions.npy", env.get_actions())
    #np.save(stats_dir + "/" + str(final_model_timesteps) + "_accumulation.npy", env.get_accumulation())
    #np.save(stats_dir + "/" + str(final_model_timesteps) + "_event_list.npy", env.get_event_list())
    np.save(stats_dir + "/" + str(final_model_timesteps) + "_abs_actions.npy", env.get_abs_actions())
    np.save(stats_dir + "/" + str(final_model_timesteps) + "_gini.npy", env.get_gini())


    #model.save("./models/75ppo_sw_tollprice") # Saving the trained model

if __name__ == "__main__":
    main()
