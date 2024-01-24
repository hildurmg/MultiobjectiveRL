import numpy as np
import pandas as pd
from oop_simulation_script import Simulation
import random
"""
class NoTollEnv():
    # creating a custom gym environment for training and testing models
    def __init__(self):
        super().__init__()
        self.params = {'alpha':1.1, 'omega':0.9, 'theta':5*10**(-1), 'tao':90, 'Number_of_user':3700} # alpha is unused

        # initialising a simulation object
        self.sim = Simulation(self.params)

        self.tt_eps = []
        # social welfare, consumer surplus, generalized cost, travel cost    
        self.sw_eps = []
        self.cs_eps = []
        self.gc_eps = [] 
        self.tc_eps = []
        self.ttcs_eps = []
        self.individual_cs_eps = []
        self.individual_sw_eps = []
        self.income_eps = []
        self.toll_eps = np.array([])
        self.toll_profile_eps = np.array([])
        self.user_flow_eps = np.array([])
        self.action_eps = np.array([])
        self.accumulation_eps = np.array([])
        self.event_list_eps = np.array([])
	
        self.tt_all_eps = []
        self.sw_all_eps = []
        self.cs_all_eps = []
        self.gc_all_eps = [] 
        self.tc_all_eps = []
        self.ttcs_all_eps = []
        self.individual_cs_all_eps = []
        self.individual_sw_all_eps = []
        self.income_all_eps = []
        self.toll_all_eps = np.array([])
        self.toll_profile_all_eps = np.array([])
        self.user_flow_all_eps = np.array([])
        self.action_all_eps = np.array([])
        self.accumulation_all_eps = np.array([])
        self.event_list_all_eps = np.array([])

        self.first_ep = True
        self.day = 0
        
        self.eps_no = 0
        

    def step(self):
        self.day+=1

        
        # creating a dictionary of the toll profile parameters after having applied the action
        action = {'mu': 0, 'sigma': 0.1, 'A': 0}
        #action1 = {'mu': action[0], 'sigma': action[1], 'A': action[2]} # absolute action
        #action = {'mu': 0.0, 'sigma': 0.0, 'A': 0.0} # no action

        # simulating one day on current simulation object
        tao_interval_information, sw, cs, gc, tc, ttcs, individual_cs, individual_sw, income_class, toll_paid, Accumulation, Event_list, alpha_list, ind_toll_paid  = self.sim.step(action) # day idx, number of users, toll paid, travel time
        self.sw_eps.append(sw) # social welfare
        self.cs_eps.append(cs) # consumer surplus
        self.gc_eps.append(gc) # generalized cost
        self.tc_eps.append(tc) # travel cost
        self.ttcs_eps.append(ttcs) # travel time cost
        self.individual_cs_eps.append(individual_cs)
        self.individual_sw_eps.append(individual_sw)
        self.income_eps.append(income_class)


        tt_mu = np.sum(tao_interval_information[3])/np.sum(tao_interval_information[1])
        
        self.tt_eps.append(tt_mu)

        # creating observation to pass to the model
        #observation = np.vstack([tao_interval_information[0],tao_interval_information[1],\
        # tao_interval_information[2],tao_interval_information[3]]) # day idx, number of users, toll paid, travel time
        #observation = np.vstack([tao_interval_information[0],tao_interval_information[1],tao_interval_information[3]]) # day idx, number of users, travel time
        #observation = np.vstack([tao_interval_information[0],tao_interval_information[1],tao_interval_information[2]]) # day idx, number of users, toll paid
        observation = np.vstack([tao_interval_information[0],tao_interval_information[1],np.zeros(15)]) # day idx, number of users
        
        
        # needed for rl function
        info = {}
        
        # finishing episode after 30 days
        if self.sim.get_day()==30:
            terminated = True
        else:
            terminated = False

        truncated = False #For now
        return observation, terminated, truncated, info
        
    # reset environment    
    def reset(self,seed=None):
        
        #print("\nEpisode",self.eps_no)
        self.eps_no += 1
        
        if self.tt_eps != []:
            self.tt_all_eps.append(np.array(self.tt_eps))
            self.sw_all_eps.append(np.array(self.sw_eps))
            self.cs_all_eps.append(np.array(self.cs_eps))
            self.gc_all_eps.append(np.array(self.gc_eps))
            self.tc_all_eps.append(np.array(self.tc_eps))
            self.ttcs_all_eps.append(np.array(self.ttcs_eps))
            self.individual_cs_all_eps.append(np.array(self.individual_cs_eps))
            self.individual_sw_all_eps.append(np.array(self.individual_sw_eps))
            self.income_all_eps.append(np.array(self.income_eps))
            #self.toll_all_eps = np.append(self.toll_all_eps, self.toll_eps)
            #self.toll_profile_all_eps = np.append(self.toll_profile_all_eps, self.toll_profile_eps)
            self.user_flow_all_eps = np.append(self.user_flow_all_eps, self.user_flow_eps)
            self.accumulation_all_eps = np.append(self.accumulation_all_eps, self.accumulation_eps)
            #self.action_all_eps = np.append(self.action_all_eps, self.action_eps)	    

            self.tt_eps = []
            self.sw_eps = []
            self.cs_eps = []
            self.gc_eps = [] 
            self.tc_eps = []
            self.ttcs_eps = []
            self.individual_cs_eps = []
            self.individual_sw_eps = []
            self.income_eps = []
            #self.toll_eps = np.array([])
            #self.toll_profile_eps = np.array([])
            self.user_flow_eps = np.array([])
            self.accumulation_eps = np.array([])
            #self.action_eps = np.array([])

        self.first_ep = False
        self.sim = Simulation(self.params)
        observation = np.zeros((2,15))
        self.day = 0

        return observation


    # defining functions to get statistics from environment
    def get_day(self):
        return self.day
    
    def get_individual_cs(self):
        return np.array(self.individual_cs_all_eps)
    
    def get_individual_sw(self):
        return np.array(self.individual_sw_all_eps)

    def get_tt(self):
        print("TEST IN FUNCTION:",self.tt_all_eps)
        return np.array(self.tt_all_eps)

    def get_sw(self):
        return np.array(self.sw_all_eps)

    def get_cs(self):
        return np.array(self.cs_all_eps)

    def get_gc(self):
        return np.array(self.gc_all_eps)

    def get_tc(self):
        return np.array(self.tc_all_eps)

    def get_ttcs(self):
        return np.array(self.ttcs_all_eps)

    def get_income(self):
        return np.array(self.income_all_eps)

    def get_accumulation(self):
        return np.array(self.accumulation_all_eps)

    def get_user_flow(self):
        return np.array(self.user_flow_all_eps)

    def get_event_list(self):
        return np.array(self.event_list_all_eps)

    def set_capacity(self, cap):
        self.sim.set_capacity(cap)

    def render(self, mode):
        pass
"""

def gini(x):
    N = len(x)
    # Mean Absolute Difference
    mad = abs(np.subtract.outer(x,x)).mean()

    T_neg = abs(sum(cs for cs in x if cs < 0))
    T_pos = abs(sum(cs for cs in x if cs > 0))
    delta_P = (2*(N - 1)/N**2) * (T_pos + T_neg)

    return mad / delta_P 

class FixedTollEnv():
    # creating a custom gym environment for training and testing models
    def __init__(self):
        super().__init__()
        self.params = {'alpha':1.1, 'omega':0.9, 'theta':5*10**(-1), 'tao':90, 'Number_of_user':3700} # alpha is unused

        # initialising a simulation object
        self.sim = Simulation(self.params)

        self.tt_eps = np.array([])
        # social welfare, consumer surplus, generalized cost, travel cost    
        self.sw_eps = np.array([])
        self.cs_eps = np.array([])
        self.gc_eps = np.array([]) 
        self.tc_eps = np.array([])
        self.ttcs_eps = np.array([])
        self.individual_cs_eps = np.array([])
        self.individual_sw_eps = np.array([])
        self.income_eps = np.array([])
        self.toll_eps = np.array([])
        self.toll_profile_eps = np.array([])
        self.user_flow_eps = np.array([])
        self.action_eps = np.array([])
        self.accumulation_eps = np.array([])
        self.abs_actions_eps = np.array([])
        self.event_list_eps = np.array([])
        self.gini_eps = np.array([])
        self.vot_eps = np.array([])
        self.toll_paid_eps = np.array([])

        self.tt_all_eps = np.array([])
        self.sw_all_eps = np.array([])
        self.cs_all_eps = np.array([])
        self.gc_all_eps = np.array([]) 
        self.tc_all_eps = np.array([])
        self.ttcs_all_eps = np.array([])
        self.individual_cs_all_eps = np.array([])
        self.individual_sw_all_eps = np.array([])
        self.income_all_eps = np.array([])
        self.toll_all_eps = np.array([])
        self.toll_profile_all_eps = np.array([])
        self.user_flow_all_eps = np.array([])
        self.action_all_eps = np.array([])
        self.accumulation_all_eps = np.array([])
        self.abs_actions_all_eps = np.array([])
        self.event_list_all_eps = np.array([])
        self.gini_all_eps = np.array([])
        self.vot_all_eps = np.array([])
        self.toll_paid_all_eps = np.array([])

        self.first_ep = True
        self.day = 0
        
        self.eps_no = 0

        #self.mu = random.random()*90.0 #0.0 to 90.0
        #self.sigma = random.random()*50.0 #0.0 to 50.0
        self.mu = 71.5
        self.sigma = 32.6
        self.A = random.random()*20.0 #0.0 to 20.0
        

    def step(self):
        self.day+=1

        
        # creating a dictionary of the toll profile parameters after having applied the action
        action = {'mu': self.mu, 'sigma': self.sigma, 'A': self.A}
        #action1 = {'mu': action[0], 'sigma': action[1], 'A': action[2]} # absolute action
        #action = {'mu': 0.0, 'sigma': 0.0, 'A': 0.0} # no action

        # simulating one day on current simulation object
        tao_interval_information, sw, cs, gc, tc, ttcs, individual_cs, individual_sw, income_class, toll_paid, Accumulation, Event_list, alpha_list, ind_toll_paid  = self.sim.step(action) # day idx, number of users, toll paid, travel time

        tt_mu = np.sum(tao_interval_information[3])/np.sum(tao_interval_information[1])

        self.sw_eps = np.append(self.sw_eps,sw) # social welfare
        self.cs_eps = np.append(self.cs_eps, cs) # consumer surplus
        self.gc_eps = np.append(self.gc_eps, gc) # generalized cost
        self.tc_eps = np.append(self.tc_eps, tc) # travel cost
        self.ttcs_eps = np.append(self.ttcs_eps, ttcs) # travel time cost
        self.individual_cs_eps = np.append(self.individual_cs_eps, individual_cs) #consumer surplus per user
        self.individual_sw_eps = np.append(self.individual_sw_eps, individual_sw)
        self.tt_eps = np.append(self.tt_eps, tt_mu) #travel time
        self.income_eps = np.append(self.income_eps, income_class)                              
        self.toll_eps = np.append(self.toll_eps, toll_paid)
        tp, x = self.get_toll_profile()
        self.toll_profile_eps = np.append(self.toll_profile_eps, tp)
        self.user_flow_eps = np.append(self.user_flow_eps, tao_interval_information[1]) #number of users
        self.accumulation_eps = np.append(self.accumulation_eps, Accumulation)
        self.abs_actions_eps = np.append(self.abs_actions_eps, self.get_params())
        self.event_list_eps = np.append(self.event_list_eps, Event_list)
        self.gini_eps = np.append(self.gini_eps, gini(np.array(individual_cs)))
        self.vot_eps = np.append(self.vot_eps, alpha_list)
        self.toll_paid_eps = np.append(self.toll_paid_eps, ind_toll_paid)

        # creating observation to pass to the model
        #observation = np.vstack([tao_interval_information[0],tao_interval_information[1],\
        # tao_interval_information[2],tao_interval_information[3]]) # day idx, number of users, toll paid, travel time
        #observation = np.vstack([tao_interval_information[0],tao_interval_information[1],tao_interval_information[3]]) # day idx, number of users, travel time
        #observation = np.vstack([tao_interval_information[0],tao_interval_information[1],tao_interval_information[2]]) # day idx, number of users, toll paid
        #observation = np.vstack([tao_interval_information[0],tao_interval_information[1]]) # day idx, number of users
        observation = np.vstack([tao_interval_information[1], np.full(15,self.A)]) # day idx, number of users
        
        
        # needed for rl function
        info = {}
        
        # finishing episode after 30 days
        if self.sim.get_day()==30:
            terminated = True
        else:
            terminated = False

        truncated = False #For now
        return observation, terminated, truncated, info
        
    # reset environment    
    def reset(self,seed=None):
        
        #print("\nEpisode",self.eps_no)
        self.eps_no += 1

        #self.mu = random.random()*90.0 #0.0 to 90.0
        #self.sigma = random.random()*50.0 #0.0 to 50.0
        self.mu = 71.5
        self.sigma = 32.6
        self.A = random.random()*20.0 #0.0 to 20.0
        
        if self.tt_eps != []:
            self.tt_all_eps = np.append(self.tt_all_eps, self.tt_eps)
            self.sw_all_eps = np.append(self.sw_all_eps, self.sw_eps)
            self.cs_all_eps = np.append(self.cs_all_eps, self.cs_eps)
            self.gc_all_eps = np.append(self.gc_all_eps, self.gc_eps)
            self.tc_all_eps = np.append(self.tc_all_eps, self.tc_eps)
            self.ttcs_all_eps = np.append(self.ttcs_all_eps, self.ttcs_eps)
            self.individual_cs_all_eps = np.append(self.individual_cs_all_eps, self.individual_cs_eps)
            self.individual_sw_all_eps = np.append(self.individual_sw_all_eps, self.individual_sw_eps)
            self.income_all_eps = np.append(self.income_all_eps, self.income_eps)
            self.toll_all_eps = np.append(self.toll_all_eps, self.toll_eps)
            self.toll_profile_all_eps = np.append(self.toll_profile_all_eps, self.toll_profile_eps)
            self.user_flow_all_eps = np.append(self.user_flow_all_eps, self.user_flow_eps)
            self.accumulation_all_eps = np.append(self.accumulation_all_eps, self.accumulation_eps)
            self.action_all_eps = np.append(self.action_all_eps, self.action_eps)
            self.abs_actions_all_eps = np.append(self.abs_actions_all_eps, self.abs_actions_eps)
            self.event_list_all_eps = np.append(self.event_list_all_eps, self.event_list_eps)
            self.gini_all_eps = np.append(self.gini_all_eps, self.gini_eps)
            self.vot_all_eps = np.append(self.vot_all_eps, self.vot_eps)
            self.toll_paid_all_eps = np.append(self.toll_paid_all_eps, self.toll_paid_eps)
        
            self.tt_eps = np.array([])
            self.sw_eps = np.array([])
            self.cs_eps = np.array([])
            self.gc_eps =  np.array([])
            self.tc_eps = np.array([])
            self.ttcs_eps = np.array([])
            self.individual_cs_eps = np.array([])
            self.individual_sw_eps = np.array([])
            self.income_eps = np.array([])
            self.toll_eps = np.array([])
            self.toll_profile_eps = np.array([])
            self.user_flow_eps = np.array([])
            self.accumulation_eps = np.array([])
            self.action_eps = np.array([])
            self.abs_actions_eps = np.array([])
            self.event_list_eps = np.array([])
            self.gini_eps = np.array([])
            self.vot_eps = np.array([])
            self.toll_paid_eps = np.array([])

        self.first_ep = False
        self.sim = Simulation(self.params)
        observation = np.zeros((2,15))
        self.day = 0

        return observation


    # defining functions to get statistics from environment
    def get_day(self):
        return self.day
    
    def get_individual_cs(self):
        return np.array(self.individual_cs_all_eps)
    
    def get_individual_sw(self):
        return np.array(self.individual_sw_all_eps)

    def get_tt(self):
        return np.array(self.tt_all_eps)

    def get_sw(self):
        return np.array(self.sw_all_eps)

    def get_cs(self):
        return np.array(self.cs_all_eps)

    def get_gc(self):
        return np.array(self.gc_all_eps)

    def get_tc(self):
        return np.array(self.tc_all_eps)

    def get_ttcs(self):
        return np.array(self.ttcs_all_eps)

    def get_toll_profile(self):
        x = np.linspace(-50,175,15)
        params = {'A': self.A, 'mu': self.mu, 'sigma': self.sigma}
        toll_profile = self.sim.custgauss(x,**params)
        return toll_profile, x

    def get_all_toll_profiles(self):
        return np.array(self.toll_profile_all_eps)
    
    def get_actions(self):
        return np.array(self.action_all_eps)
    
    def get_event_list(self):
        return np.array(self.event_list_all_eps)

    def get_abs_actions(self):
        return np.array(self.abs_actions_all_eps)
    
    def get_user_flow(self):
        return np.array(self.user_flow_all_eps)

    def get_gini(self):
        return np.array(self.gini_all_eps)

    def get_vot(self):
        return np.array(self.vot_all_eps)

    def get_toll_paid(self):
        return np.array(self.toll_paid_all_eps)

    def get_accumulation(self):
        return np.array(self.accumulation_all_eps)
    
    def get_income(self):
        return np.array(self.income_all_eps)
    
    def get_toll(self):
        return np.array(self.toll_all_eps)

    def get_params(self):
        params = {'A': self.A, 'mu': self.mu, 'sigma': self.sigma}
        return params

    def set_params(self, params):
        self.A = params['A']
        self.mu = params['mu']
        self.sigma = params['sigma']

    def set_capacity(self, cap):
        self.sim.set_capacity(cap)

    def render(self, mode):
        pass

