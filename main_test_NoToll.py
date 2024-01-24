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
import os

from oop_simulation_script import Simulation, CommuteEnv
from NoToll_Scenario import FixedTollEnv #,NoTollEnv


def main():

    no_episodes = 200
    terminated = False
    env = FixedTollEnv()
    #env = NoTollEnv()
    for eps in range(no_episodes):
        d = 1
        while not terminated:
            print("Episode:",eps+1,"Day:",d)
            observation, terminated, truncated, info = env.step()
            d+=1
        env.reset()
        terminated = False

    stats_dir = "./local_stats/RandomToll/simpler_env/run2/"
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    
    model_param = 'RandomToll'

    print("Saving stats...")
    
    np.save(stats_dir+"RandomToll_tt.npy", env.get_tt())
    np.save(stats_dir+"RandomToll_sw.npy", env.get_sw())
    np.save(stats_dir+"RandomToll_cs.npy", env.get_cs())
    #np.save(stats_dir+"RandomToll_gc.npy", env.get_gc())
    #np.save(stats_dir+"RandomToll_tc.npy", env.get_tc())
    #np.save(stats_dir+"RandomToll_ttcs.npy", env.get_ttcs())
    np.save(stats_dir+"RandomToll_ind_cs.npy", env.get_individual_cs())
    #np.save(stats_dir+"RandomToll_ind_sw.npy", env.get_individual_sw())
    #np.save(stats_dir+"RandomToll_ind_inc.npy", env.get_income())
    np.save(stats_dir+"RandomToll_toll.npy", env.get_toll())
    np.save(stats_dir+"RandomToll_toll_profile.npy", env.get_all_toll_profiles())
    np.save(stats_dir+"RandomToll_user_flow.npy", env.get_user_flow())
    #np.save(stats_dir+"RandomToll_accumulation.npy", env.get_accumulation())
    #np.save(stats_dir+"RandomToll_event_list.npy", env.get_event_list())
    np.save(stats_dir+"RandomToll_abs_actions.npy", env.get_abs_actions())
    np.save(stats_dir+"RandomToll_gini.npy", env.get_gini())
    #np.save(stats_dir+"RandomToll_vot.npy", env.get_vot())
    #np.save(stats_dir+"RandomToll_toll_paid.npy", env.get_toll_paid())

    print('Finished saving stats!')



if __name__ == "__main__":
    main()
