#IMPORTS
import random
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as th
import mo_gymnasium as mo_gym
from mo_gymnasium.utils import MORecordEpisodeStatistics


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
        # register(
        #     id="traffic-env-v0",
        #     entry_point="oop_simulation_script:CommuteEnv",
        # )
        env = mo_gym.make("mo-hopper-v4", cost_objective=False, max_episode_steps=500)
        #env = gymnasium.make("traffic-env-v0")
        #env = Monitor(CommuteEnv(reward_type,days_per_eps))
        #env.seed(seed)
        return env
    return _init

def main():
    # Set up environment
    reward_type = "sw" # tt: Travel Time    sw: Social Welfare
    days_per_eps = 3
    
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

    pf = agent.train(
        total_timesteps=6,
        ref_point=ref_point,
        eval_env=env,
        eval_freq=3,
    )
    print(pf)

    

    # Execute a policy
    #target = np.array(pf.pop())
    #print(f"Tracking {target}")
    #reward = agent.track_policy(target, env=env)
    #print(f"Obtained {reward}")



    # Saving the observed stats from training the model
    np.save("./stats/training/75ppo_tt.npy", env.get_tt())
    np.save("./stats/training/75ppo_sw.npy", env.get_sw())
    np.save("./stats/training/75ppo_cs.npy", env.get_cs())
    np.save("./stats/training/75ppo_gc.npy", env.get_gc())
    np.save("./stats/training/75ppo_tc.npy", env.get_tc())
    np.save("./stats/training/75ppo_ttcs.npy", env.get_ttcs())
    np.save("./stats/training/75ppo_ind_cs.npy", env.get_individual_cs())

    #model.save("./models/75ppo_sw_tollprice") # Saving the trained model

if __name__ == "__main__":
    main()
