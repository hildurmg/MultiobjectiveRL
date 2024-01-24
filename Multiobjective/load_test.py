



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

    #env = DummyVecEnv([lambda: env])
    #env = VecFrameStack(env, n_stack=30) 
    #env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Load trained model
    print("Creating model...")
    ref_point=np.array([0, -25])
    
    agent = CAPQL(
        env,
        seed=1
    )

    print("\n\n\n")
    print("---------Model--------------")
    agent.load("./weights/CPQL.tar")
    print(agent)
    print("----------------------------")
    print("\n\n\n")

if __name__ == "__main__":
    main()
