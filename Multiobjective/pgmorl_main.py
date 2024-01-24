import numpy as np

from morl_baselines.common.evaluation import eval_mo
from morl_baselines.multi_policy.pgmorl.pgmorl import PGMORL
from morl_baselines.single_policy.ser.mo_ppo import make_env
from gymnasium.envs.registration import register
import gymnasium
import os


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


if __name__ == "__main__":
    reward_type = "sw" # tt: Travel Time    sw: Social Welfare
    days_per_eps = 30
    train_eps = 20
    eps = 20 # For naming
    
    env=make_env(10,reward_type,days_per_eps)() # Seed as argument
    register(
            id="traffic-env-v0",
            entry_point="oop_simulation_script:CommuteEnv",
        )
    nr_timesteps = train_eps*days_per_eps

    stats_dir = "./local_stats/MultiObjective/simpler_env/random/onlySW/2_warmup_7/"+str(eps)+"PGMORL"
    #stats_dir = "./local_stats/MultiObjective/perc_cs+sw/"+str(final_model_timesteps)
    #stats_dir = "./local_stats/MultiObjective/gini_cs+sw/"+str(final_model_timesteps)
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
        
    algo = PGMORL(
        env_id="traffic-env-v0",
        #env=env,
        origin=np.array([0., -25.]), #Bætti þessu við
        num_envs=1,
        pop_size=5,
        learning_rate=1e-4, #from ppo
        warmup_iterations=2,#80,
        evolutionary_iterations=100,#20,
        steps_per_iteration=30, #from ppo
        num_minibatches=1, #from ppo
        target_kl=0.05, #from ppo
        gae_lambda=0.97, #from ppo
        update_epochs=10,#80, #from ppo
        num_weight_candidates=7,
    )
    # env = make_env(env_id, 422, 1, "PGMORL_test", gamma=0.995)()  # idx != 0 to avoid taking videos

    algo.train(
        total_timesteps=50,
        eval_env=env,
        ref_point=np.array([0., -25.]),
        known_pareto_front=None,
    )


    # Saving the observed stats from atraining the model
    np.save(stats_dir + "/" + str(eps) + "PGMORL_tt.npy"    , env.get_tt())
    np.save(stats_dir + "/" + str(eps) + "PGMORL_sw.npy"    , env.get_sw())
    np.save(stats_dir + "/" + str(eps) + "PGMORL_cs.npy"    , env.get_cs())
    np.save(stats_dir + "/" + str(eps) + "PGMORL_gc.npy"    , env.get_gc())
    np.save(stats_dir + "/" + str(eps) + "PGMORL_tc.npy"    , env.get_tc())
    np.save(stats_dir + "/" + str(eps) + "PGMORL_ttcs.npy"  , env.get_ttcs())
    #np.save(stats_dir + "/" + str(eps) + "PGMORL_ind_cs.npy", env.get_individual_cs())
    #np.save(stats_dir + "/" + str(eps) + "PGMORL_ind_sw.npy", env.get_individual_sw())
    #np.save(stats_dir + "/" + str(eps) + "PGMORL_ind_inc.npy", env.get_income())
    np.save(stats_dir + "/" + str(eps) + "PGMORL_toll.npy", env.get_toll())
    np.save(stats_dir + "/" + str(eps) + "PGMORL_toll_profile.npy", env.get_all_toll_profiles())
    np.save(stats_dir + "/" + str(eps) + "PGMORL_user_flow.npy", env.get_user_flow())
    np.save(stats_dir + "/" + str(eps) + "PGMORL_actions.npy", env.get_actions())
    np.save(stats_dir + "/" + str(eps) + "PGMORL_accumulation.npy", env.get_accumulation())
    np.save(stats_dir + "/" + str(eps) + "PGMORL_event_list.npy", env.get_event_list())
    np.save(stats_dir + "/" + str(eps) + "PGMORL_abs_actions.npy", env.get_abs_actions())



    # # Execution of trained policies
    # for a in algo.archive.individuals:
    #     scalarized, discounted_scalarized, reward, discounted_reward = eval_mo(
    #         agent=a, env=env, w=np.array([1.0, 1.0]), render=True
    #     )
    #     print(f"Agent #{a.id}")
    #     print(f"Scalarized: {scalarized}")
    #     print(f"Discounted scalarized: {discounted_scalarized}")
    #     print(f"Vectorial: {reward}")
    #     print(f"Discounted vectorial: {discounted_reward}")