import random
import numpy as np
import pandas as pd
from numba import njit, prange
from numba.typed import List
from gymnasium import Env 
from gymnasium.spaces import Box
from math import floor
import time

def gini(x):
    N = len(x)
    # Mean Absolute Difference
    mad = abs(np.subtract.outer(x,x)).mean()

    T_neg = abs(sum(cs for cs in x if cs < 0))
    T_pos = abs(sum(cs for cs in x if cs > 0))
    delta_P = (2*(N - 1)/N**2) * (T_pos + T_neg)

    return mad / delta_P

@njit
def column_generate(time_slot,case='all'):
    columns=[]
    if case=='all':
        for i in range(Number_of_user):
            columns.append([j for j in range(np.sum(time_slot[i]<Departure_time[i])-tao,\
                                                      np.sum(time_slot[i]<Departure_time[i])+tao+1)])
    elif case=='chosen':
        for i in range(Number_of_user):
            columns.append([np.sum(time_slot[i]<Departure_time[i])])
    return columns

@njit
def V1(x):
    return np.square(1-x/4500)*9.78

@njit
def fic_tt(user, time_point, all_time_matrix, new_timelist, trip_len, Accumulation,Number_of_user):
    
    # get the fictional departure time for the fictional traveler
    star_time=all_time_matrix[user,time_point]
    
    # get the list of events happening after this given departure time point to simulate the expected states
    known_list=new_timelist[new_timelist>star_time]
    
    if len(known_list)==0: # if this fictional departure happens after all real travelers
        # exit the network assuming free flow speed (9.78)
        texp=trip_len[user]/9.78/60
    elif len(known_list)==Number_of_user*2: # this fictional departure happens before all real travelers enter the network
        # compute the left trip length till the first real traveler enter the network
        texp=0
        count=0
        left_len=trip_len[user]-9.78*60*(known_list[0]-star_time)
        
        if left_len<0: # if this fictional traveler ends his trip before the first real traveler enter the network
            # exit the network assuming free flow speed (9.78)
            texp=trip_len[user]/9.78/60
            
        else: # compute travel speed in each time interval between two consecutive events
            V_list=np.array([V1(x) for x in Accumulation[Number_of_user*2-len(known_list):-1]]) #Speed for all times in known list
            len_piece=np.diff(known_list)*V_list*60 # trip length traveled in each time interval between two consecutive events
            cum_len=np.cumsum(len_piece)
            count=np.sum(cum_len<left_len)
            # Time spent in env = time @ interval right after leaving - start time + the time spent in last time interval
            texp=known_list[count+1]-star_time+(left_len-cum_len[count])/V1(Accumulation[count])/60
    else: # it means this fictional departure happens after some real travelers have entered the network
        texp=0
        count=0
        # compute the left trip length till the next closest event occurs (either a departure or arrival)
        left_len=trip_len[user]-V1(Accumulation[Number_of_user*2-len(known_list)-1])*(known_list[0]-star_time)*60
        if left_len<0: # if this fictional traveler end his trip before the next real event occurs
            texp=trip_len[user]/V1(Accumulation[Number_of_user*2-len(known_list)-1])/60
        else:
            # travel speed in each time interval between two consecutive events
            V_list=np.array([V1(x) for x in Accumulation[Number_of_user*2-len(known_list):-1]])
            
            # trip length traveled in each time interval between two consecutive events
            len_piece=np.diff(known_list)*V_list*60
            cum_len=np.cumsum(len_piece)
            count=np.sum(cum_len<left_len)
            if count==0:
                texp=known_list[count]-star_time+(left_len-(known_list[count]-star_time)*V1(1))/9.78/60
            elif count==len(cum_len): # this fictional traveler's is not finished even after all real travelers finish their trips
                texp=known_list[count]-star_time+(left_len-cum_len[count-1])/9.78/60
            else: # this fictional traveler finishes the trip before all real travelers finish their trips
                texp=known_list[count+1]-star_time+(left_len-cum_len[count])/V1(Accumulation[Number_of_user*2-len(known_list)+count])/60
    return texp

@njit(parallel=True)
def T_est(all_time_matrix, new_timelist, trip_len, Accumulation, Number_of_user, tao):
    T_estimate_array=np.zeros((Number_of_user,2*tao+1))
    for i in prange(Number_of_user):
        for j in prange(2*tao+1):
            #fic_tt calculates the time spent in the network for a given trip length
            T_estimate_array[i,j]=fic_tt(i, j, all_time_matrix, new_timelist, trip_len, Accumulation,Number_of_user)
    return T_estimate_array

class Simulation():
    # setting up simulation with initial parameters
    def __init__(self, params):
        np.random.seed(seed=59) # tried 2
        # setting values from parameter input
        self.omega = params['omega'] # Learning rate
        self.theta = params['theta'] #scale parameter
        self.tao = params['tao'] # number of time intervals
        self.Number_of_user = params['Number_of_user'] # number of users
        self.time_slots_names = ['t'+str(i) for i in range(2*self.tao+1)]
        
        self.day = 0
        
        self.window_c_perceived = pd.DataFrame(columns=self.time_slots_names)
        self.c_perceived = pd.DataFrame(columns=self.time_slots_names)
        
        # departure time for each user
        # ATH: Why is the if statement sampling from the same random curve once again when departure time is < 20 or > 150?
        self.Departure_time = np.random.normal(80,18,self.Number_of_user)
        #for i in range(len(self.Departure_time)):
        #    if self.Departure_time[i]<20 or self.Departure_time[i]>150:
        #        self.Departure_time[i] = np.random.normal(80,18,1)
        self.Departure_time[(self.Departure_time < 20) & (self.Departure_time > 150)] = np.random.normal(80,18,1) #Updated for performance

        # trip length for each user
        # ATH: Why 4600? We can't find an explanation for that number anywhere.
        #self.trip_len = np.array(np.zeros(self.Number_of_user))
        #for i in range(self.Number_of_user):
        #    self.trip_len[i] = 4600+np.random.normal(0,(0.02*4600)**2)
        #    while self.trip_len[i]<20:
        #        self.trip_len[i] = 4600+np.random.normal(0,(0.02*4600)**2)

        #self.trip_len = 4600+np.random.normal(0,(0.02*4600)**2,self.Number_of_user) #updated for performance

        self.trip_len=np.load("./fixed_params/trip_len.npy")


        while self.trip_len[self.trip_len < 20].any():
            self.trip_len[self.trip_len < 20] = 4600+np.random.normal(0,(0.02*4600)**2)
        
        self.Wished_Arrival_time = self.Departure_time+self.trip_len/9.78/60 #9.78 is free flow speed (in m/s)
        
        # ATH: Why departure_time - tao + i?
        all_time_slots = [self.Departure_time-self.tao+i for i in range(2*self.tao+1)]
        self.all_time_slot = pd.DataFrame(data=np.transpose(all_time_slots), columns=self.time_slots_names)
       
        self.all_time_matrix = np.array(self.all_time_slot)

        self.E = np.array(np.zeros(self.Number_of_user)) # schedule delay early penalty
        self.L = np.array(np.zeros(self.Number_of_user)) # schedule delay late penalty
        
        # schedule delay penalties and value of time for users
        #self.E=np.random.lognormal(-1.9,0.2,self.Number_of_user)*4
        self.E=np.load("./fixed_params/E.npy")
        self.L=self.E*np.exp(1)
        self.alpha=self.E*np.exp(0.5)
    
        
        #lower_alpha = np.sort(self.alpha)[floor(self.alpha.shape[0]*0.35)]
        lower_alpha = np.sort(self.alpha)[floor(self.alpha.shape[0]*0.1)]
        upper_alpha = np.sort(self.alpha)[floor(self.alpha.shape[0]*0.89)]
        self.income_class = (self.alpha < lower_alpha).astype(int) + ((self.alpha >= lower_alpha).astype(int) & (self.alpha < upper_alpha).astype(int))*2 + (self.alpha >= upper_alpha).astype(int)*3
                
        

        # random utility
        #self.util_rand = np.random.gumbel(-0.57721 / self.theta, 1.0 / self.theta, (self.Number_of_user,2*self.tao+1))
        self.util_rand = np.load("./fixed_params/util.npy")
        self.ur = pd.DataFrame(self.util_rand,columns=self.time_slots_names)

        self.Eachday_dep=pd.DataFrame()
        self.Eachday_dep['d0']=self.Departure_time
        
        
        dep_time_set_vals = [self.Departure_time-self.tao+i for i in range(2*self.tao+1)]
        self.Dep_time_set=pd.DataFrame(data=np.transpose(dep_time_set_vals), columns=self.time_slots_names)
        
        # ATH: Why load this departure array? What is it used for?
        self.Departure_time = np.load("./base_deptime.npy", allow_pickle=True) #npy file contains a numpy array

        self.Acc_df=pd.DataFrame() # record the accumulation on each day
        self.time_label=pd.DataFrame() # time points of the events on each day

        self.cs_list = []
        self.cost_list = []
        self.ttcs_list = []
        self.individual_cost_list = []
        self.individual_cs_list = []
        self.utility = {}
        self.time = time.perf_counter()

        # starting capacity of the traffic network
        self.capacity = 4500

    # running a single day of simulation    
    def step(self, action):
        # simulating users on the traffic network
        #vehicle_information, time_list, Accumulation, Speed = self.within_day_process()
        vehicle_information, time_list, Accumulation, Speed, Event_list = self.within_day_process()
        
        self.Acc_df['d'+str(self.day)]=Accumulation
        self.time_label['d'+str(self.day)]=time_list
        vehicle_information['origin_tl']=self.trip_len
        
        new_timelist = time_list

        #Fixing for numba deprecation warning
        #typed_accumulation = List()
        #[typed_accumulation.append(x) for x in Accumulation]
        
        # T_est calculates the time spent in the network for a given trip length
        #T_estimate = T_est(self.all_time_matrix, new_timelist, self.trip_len, typed_accumulation, self.Number_of_user, self.tao)
        T_estimate = T_est(self.all_time_matrix, new_timelist, self.trip_len, Accumulation, self.Number_of_user, self.tao)
        T_estimated = pd.DataFrame(T_estimate,columns=self.time_slots_names)

        # computing the difference in arrival time and wished arrival time
        #T_estimated_diff = pd.DataFrame()
        T_diff = []
        
        for j in range(2*self.tao+1):
            T_esti = T_estimated['t'+str(j)]+self.all_time_slot['t'+str(j)]-self.Wished_Arrival_time
            #T_estimated_diff['t'+str(j)] = T_esti
            T_diff.append(np.array(T_esti))
        T_diff = np.transpose(T_diff)
        #T_estimated_diff = pd.DataFrame(data=T_diff, columns=self.time_slots_names) #commented out because it was never accessed
        #T_diff = np.array(T_estimated_diff)
        SD = self.schedule_delay(T_diff) #Calculating penalty for schedule delays (early or late)

        c_estimated=pd.DataFrame(columns=self.time_slots_names)
        c_cs=pd.DataFrame(columns=self.time_slots_names)
        ttcs=pd.DataFrame(columns=self.time_slots_names)
        
        #toll_array = np.array([])
        discount = (self.income_class == 1)*0.2#action['discount']

        for j in range(2*self.tao+1):
            c_estimated['t'+str(j)]=self.alpha*T_estimated['t'+str(j)]+SD[:,j]+\
                self.custgauss(self.all_time_slot['t'+str(j)],action['mu'],action['sigma'],action['A'])*np.array(vehicle_information['origin_tl'])*0.0002*np.array(1-discount)
  
            c_cs['t'+str(j)]=self.alpha*T_estimated['t'+str(j)]+SD[:,j] # consumer surplus
            ttcs['t'+str(j)]=self.alpha*T_estimated['t'+str(j)] # travel time cost
        
        if self.day==0:
            self.c_perceived=c_cs
        else:
            self.c_perceived=self.omega*self.c_perceived+(1-self.omega)*c_cs # cost = (yesterday's cost * learning rate) + (today's cost * (1 - learning rate))
        self.window_c_perceived = self.c_perceived
        
        utility_exp=-self.window_c_perceived+self.util_rand # U = V + random error
        
        toll_paid = 0
    
        for j in range(2*self.tao+1):
            utility_exp['t'+str(j)] = utility_exp['t'+str(j)]-self.custgauss(self.all_time_slot['t'+str(j)],action['mu'],action['sigma'],action['A'])*np.array(vehicle_information['origin_tl'])*0.0002*np.array(1-discount)

            toll_paid += self.custgauss(self.all_time_slot['t'+str(j)],action['mu'],action['sigma'],action['A'])*np.array(vehicle_information['origin_tl'])*0.0002*np.array(1-discount)


        columns1 = []
        for i in range(self.Number_of_user):
            columns1.append(['t'+str(np.sum(self.all_time_slot.iloc[i].values<self.Departure_time[i]))])
        window_cs = self.rearrange(df=c_cs,cols=columns1)
        window_c_exp = self.rearrange(df=c_estimated,cols=columns1)
        window_ttcs = self.rearrange(df=ttcs,cols=columns1)
        
        self.cost_list.append(np.sum(window_c_exp))
        self.individual_cost_list.append(window_c_exp)
        
        self.cs_list.append(window_cs.sum())
        self.individual_cs_list.append(window_cs)
        self.ttcs_list.append(window_ttcs.sum())
        
        self.Departure_time=np.diag(self.Dep_time_set[utility_exp.idxmax(axis=1)])
        self.utility['d'+str(self.day)]=np.diag(self.ur[utility_exp.idxmax(axis=1)])
    
        self.Eachday_dep['d'+str(self.day+1)]=self.Departure_time
        
        # computing state variables
        day_idx = np.full(15, np.float32(self.day))
        
        bins = np.histogram(vehicle_information['t_dep'], bins=15, range=(-50,175))[1]
        inds = np.digitize(vehicle_information['t_dep'], bins)

        user_info = np.zeros([3,15])
        for idx, user in enumerate(inds):
            user_info[0,user]+=1 # total user departures in each 15 minute time window
            user_info[1,user]+=toll_paid[idx]  # total toll paid in each 15 minute time window
            user_info[2,user]+=vehicle_information['t_exp'][idx]  # total travel time in each 15 minute time window

        # day index, number of travelers, toll rates and total travel time for 15 min intervals
        tao_interval_information = np.vstack([day_idx, np.float32(user_info)])
        
        self.day+=1
        print("DAY: " + str(self.day))
        #time_difference = time.perf_counter() - self.time
       # print("TIME: " + str(time_difference))
        #self.time = time.perf_counter()
        # return tao_interval_information, sw, cs, gc, tc, ttcs
        sw = -self.cs_list[self.day-1][0]+np.sum(self.utility['d'+str(self.day-1)])
        cs = -self.cost_list[self.day-1][0]+np.sum(self.utility['d'+str(self.day-1)])
        gc = self.cost_list[self.day-1][0]
        tc = self.cs_list[self.day-1][0]
        ttcs = self.ttcs_list[self.day-1][0]
        individual_cs = -self.individual_cost_list[self.day-1][0] + self.utility['d'+str(self.day-1)]
        individual_sw = -self.individual_cs_list[self.day-1][0] + self.utility['d'+str(self.day-1)]


        return tao_interval_information, sw , cs, gc , tc, ttcs , individual_cs, individual_sw, self.income_class, toll_paid, Accumulation, Event_list #, T_percieved, T_experience
        
    

    def custgauss(self, x,mu,sigma,A):
        return A*np.exp(-(x-mu)**2/2/sigma**2)
    
    def within_day_process(self):
        # Step 1
        n=0 # Number of vehicle (accumulation)
        j=0 # index of event
        #t=[] # event time
        #vehicle_index=[]
        t = np.array([])
        vehicle_index=np.array([], dtype=int)
        Accumulation=np.array([])
        Speed=[]
    
        # Predicted arrival time
        Arrival_time=self.Departure_time+self.trip_len/9.78/60
    
        # Step 2
        # Define event list of departures
    
        Event_list1_array=np.zeros((self.Number_of_user,4))
        Event_list1_array[:,0]=np.arange(self.Number_of_user) # vehicle index
        Event_list1_array[:,1]=self.Departure_time # time(min)
        Event_list1_array[:,2]=np.ones(self.Number_of_user) # departure indicator: 1
        Event_list1_array[:,3]=self.trip_len # trip length
    
        # Define event list of arrivals
        Event_list2_array=np.zeros((self.Number_of_user,4))
        Event_list2_array[:,0]=np.arange(self.Number_of_user) # vehicle index
        Event_list2_array[:,1]=Arrival_time # time(min)
        Event_list2_array[:,2]=np.ones(self.Number_of_user)*2 # arrival indicator: 2
        Event_list2_array[:,3]=self.trip_len # trip length

        # S_Event_list_array: 4 columns
        # vehicle_index  time(min)  event_indicator  trip_len

        # Concatenate these two event lists
        S_Event_list_array=np.concatenate((Event_list1_array, Event_list2_array), axis=0)
    
        # Sort the list by time in ascending order
        S_Event_list_array=S_Event_list_array[S_Event_list_array[:, 1].argsort()]
    
        # get time of the first event
        t = np.append(t, S_Event_list_array[0,1])
        #t.append(S_Event_list_array[0,1]) #initial time
    
        # create a dict to store the information of each agent
        vehicle_information = {}
        vehicle_information['vehicle']=np.arange(self.Number_of_user)
        #vehicle_information['trip_len(m)']=self.trip_len.astype(np.float64)
        #vehicle_information['trip_len(m)']=self.trip_len.astype(np.float32)
        vehicle_information['t_exp']=np.zeros(self.Number_of_user)
        vehicle_information['account']=np.zeros(self.Number_of_user) #not used?
        trip_length=self.trip_len.astype(np.float64)

        S_Event_list_array_copy = S_Event_list_array.copy()
        
    
        #Step 3
        # Event-based simulation
        while S_Event_list_array.shape[0]>0:
            j=j+1
            #t.append(S_Event_list_array[0,1]) # record the time of the event
            t = np.append(t, S_Event_list_array[0,1])
            if S_Event_list_array[0,2]==1:     
                #vehicle_index.append(int(S_Event_list_array[0,0])) # record the agent that starts the trip
                vehicle_index = np.append(vehicle_index, int(S_Event_list_array[0,0]))
                # update the untraveled trip length
                # trip_len1=vehicle_information['trip_len(m)']
                # trip_len1[vehicle_index[0:-1]]=trip_len1[vehicle_index[0:-1]]-self.V(n)*60*(t[j]-t[j-1]) #Updating distance which is left for all vehicles (except newest addition)
                # vehicle_information['trip_len(m)']=trip_len1                                      #V(n) is speed given accumulation n
                trip_length[vehicle_index[0:-1]]=trip_length[vehicle_index[0:-1]]-self.V(n)*60*(t[j]-t[j-1])


                # update the accumulation in the network
                n=n+1
            
                # keep track of the accumulation
                Accumulation = np.append(Accumulation, n)
            
                # update the predicted arrival time
                #temp=S_Event_list_array[(np.where((S_Event_list_array[:,2]==2) & (np.isin(S_Event_list_array[:,0],vehicle_index[0:-1])==True)))][:,0]
                #if np.size(temp)==0:
                #    temp = np.array([])
                #S_Event_list_array[(np.where((S_Event_list_array[:,2]==2) & (np.isin(S_Event_list_array[:,0],vehicle_index[0:-1])==True))),1]=\
                #t[j]+vehicle_information['trip_len(m)'][temp.astype(int)]/self.V(n)/60
                
                #Changing to increase performance
                temp=np.where((S_Event_list_array[:,2]==2) & (np.isin(S_Event_list_array[:,0],vehicle_index[0:-1])==True))
                S_Event_list_array[temp,1]=t[j]+trip_length[S_Event_list_array[temp][:,0].astype(int)]/self.V(n)/60

            else: #arrival
                # update the trip length
                # trip_len1=vehicle_information['trip_len(m)']
                # trip_len1[vehicle_index]=trip_len1[vehicle_index]-self.V(n)*60*(t[j]-t[j-1])
                # vehicle_information['trip_len(m)']=trip_len1
                trip_length[vehicle_index]=trip_length[vehicle_index]-self.V(n)*60*(t[j]-t[j-1])


                # update the accumulation in the network
                n=n-1
            
                # keep track of the accumulation
                Accumulation = np.append(Accumulation, n)

                # update t_exp (Actual time from departure to arrival)
                vehicle_information['t_exp'][int(S_Event_list_array[0,0])]=S_Event_list_array[0,1]-self.Departure_time[int(S_Event_list_array[0,0])]

                # remove the agent that finishes the trip
                #vehicle_index.remove(int(S_Event_list_array[0,0]))
                vehicle_index = np.delete(vehicle_index, np.where(vehicle_index == int(S_Event_list_array[0,0])))
        
                # Update the predicted arrival time
                #temp=S_Event_list_array[(np.where((S_Event_list_array[:,2]==2) & (np.isin(S_Event_list_array[:,0],vehicle_index)==True)))][:,0]
                #if np.size(temp)==0:
                #    temp = np.array([])
                #S_Event_list_array[(np.where((S_Event_list_array[:,2]==2) & (np.isin(S_Event_list_array[:,0],vehicle_index)==True))),1]=\
                #t[j]+vehicle_information['trip_len(m)'][temp.astype(int)]/self.V(n)/60
                #Changing to increase performance
                temp=np.where((S_Event_list_array[:,2]==2) & (np.isin(S_Event_list_array[:,0],vehicle_index)==True))
                S_Event_list_array[temp,1]=t[j]+trip_length[S_Event_list_array[temp][:,0].astype(int)]/self.V(n)/60

            # remove event from the list
            S_Event_list_array = np.delete(S_Event_list_array, (0), axis=0)    
            S_Event_list_array=S_Event_list_array[S_Event_list_array[:, 1].argsort()]
            # update speed with Speed.function
            #Speed.append(self.V(n))
        
        vehicle_information['trip_len(m)']=trip_length
        vehicle_information['t_dep']=Event_list1_array[:,1]
        vehicle_information['t_arr']=vehicle_information['t_dep']+vehicle_information['t_exp']
        time_list=np.concatenate((vehicle_information['t_dep'], vehicle_information['t_arr']), axis=0)
        time_list=time_list=np.sort(time_list,axis=None)
        return vehicle_information, time_list, Accumulation, Speed, S_Event_list_array_copy

    # Speed function
    def V(self, x):
        if isinstance(x,list):
            return [np.square(1-i/self.capacity)*9.78 for i in x]
        else:
            return np.square(1-x/self.capacity)*9.78
    
    def set_capacity(self, cap):
        self.capacity=cap

    def rearrange(self, df, cols):
    
        all_values=[]
        for idx,i in enumerate(cols):
        
            vals=df[i].T[idx].values
            all_values.append(vals)
    
        if len(cols[0])>1:
            return pd.DataFrame(data=np.vstack(all_values),columns=self.time_slots_names)
        else:
            return pd.DataFrame(data=np.vstack(all_values))
    
    def schedule_delay(self, T_diff):
        SD=np.empty((self.Number_of_user,2*self.tao+1))
        for i in range(2*self.tao+1):
            SD[:,i]=self.L*T_diff[:,i]*(1-np.array(T_diff[:,i]<0).astype(int))-\
            self.E*T_diff[:,i]*np.array(T_diff[:,i]<0).astype(int)
        return SD
    
    def get_day(self):
        return self.day

class CommuteEnv(Env):
    # creating a custom gym environment for training and testing models
    #def __init__(self,reward_type="tt",days_per_eps=30):
    def __init__(self, render_mode=""):
        super().__init__()
        
        self.params = {'alpha':1.1, 'omega':0.9, 'theta':5*10**(-1), 'tao':90, 'Number_of_user':3700} # alpha is unused
        # define action space for each actionable value
        # mu, sigma, A
        #self.action_space = Box(low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), shape=(3,), dtype=np.float64)
        #self.action_space = Box(low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), shape=(3,), dtype=np.float32) #used for absolute actions
        self.action_space = Box(low=np.array([-1.0]), high=np.array([1.0]), shape=(1,), dtype=np.float32) #used for absolute actions
        #self.action_space = Box(low=np.array([-1.0, -0.1]), high=np.array([1.0, 0.1]), shape=(2,), dtype=np.float32) #discount

        # define observation space for each observed value
        #self.observation_space = Box(low=(np.full((2,15), -np.inf)),\
        #                             high=(np.full((2,15), np.inf)),\
        #                             shape=(2,15), dtype=np.float32) # include more observations for a broader observation space

        self.observation_space = Box(low= -np.inf,\
                                     high=np.inf,\
                                    shape=(30,), dtype=np.float32) 
                                     #shape=(45,), dtype=np.float32) # include more observations for a broader observation space

        # reward space
        self.reward_dim = 2
        self.reward_space = Box(low=-np.inf,\
                                     high=np.inf,\
                                     shape=(self.reward_dim,), dtype=np.float32)

        # initialising a simulation object
        self.sim = Simulation(self.params)
        
        # sampling initial toll profile parameters
        #self.mu = random.random()*90.0 #0.0 to 90.0
        #self.sigma = random.random()*50.0 #0.0 to 50.0
        self.A = random.random()*20.0 #0.0 to 20.0
        self.discount = random.random()
        
        self.mu = 71.5
        self.sigma = 32.6
        #self.A = 0

        self.eps_no = 0
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
        self.discount_eps = np.array([])

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
        self.discount_all_eps = np.array([])

        self.first_ep = True
        self.day = 0

        self.eps1 = False
        self.eps2 = False
        self.eps3 = False
        
        self.number_of_days = 30
        #self.reward_type = reward_type

        
        

    def step(self, action):

        # if self.day == 0:
        #     prev_action = {"mu": np.zeros(15), "sigma": np.ones(15), "A": np.zeros(15)}
        #     self.action_eps = np.append(self.action_eps,{"mu": 0, "sigma": 1, "A": 0})
        # else:
        #     #prev_action = {"mu": np.full(15,self.action_eps[-1]["mu"]), "sigma": np.full(15,self.action_eps[-1]["sigma"]), "A": np.full(15,self.action_eps[-1]["sigma"])}
        #     prev_action = {"mu": np.full(15,self.mu), "sigma": np.full(15,self.sigma), "A": np.full(15,self.A)}
        #     self.action_eps = np.append(self.action_eps,{"mu": action[0], "sigma": action[1], "A": action[2]})

        if self.day == 0:
            prev_action = {"A": np.zeros(15)}
            self.action_eps = np.append(self.action_eps,{"mu": 0, "sigma": 1, "A": 0})
            #self.action_eps = np.append(self.action_eps,{"A": 0, "discount": 0})
        else:
            #prev_action = {"A": np.full(15,self.action_eps[-1]["A"])}
            prev_action = {"A": np.full(15,self.A)}
            #self.action_eps = np.append(self.action_eps,{"A": action[0], "discount": action[1]})
            self.action_eps = np.append(self.action_eps,{"A": action[0]})

        # ensuring profile parameters are within the set boundaries
        # if self.mu+action[0]*3.0 < 0.0:
        #     self.mu = 0.0
        # elif self.mu+action[0]*3.0 > 90.0:
        #     self.mu = 90.0
        # else:
        #     self.mu+=action[0]*3.0

        # if self.sigma+action[1]*2.0 < 0.0:
        #     self.sigma = 0.0
        # elif self.sigma+action[1]*2.0 > 50.0:
        #     self.sigma = 50.0
        # else:
        #     self.sigma+=action[1]*2.0

        # if self.A+action[2] < 0.0:
        #     self.A = 0.0
        # elif self.A+action[2] > 20.0:
        #     self.A = 20.0
        # else:
        #     self.A+=action[2]

        if self.A+action[0] < 0.0:
            self.A = 0.0
        elif self.A+action[0] > 20.0:
            self.A = 20.0
        else:
            self.A+=action[0]

        # Discount percentage
        # if self.discount+action[1] < 0.0:
        #    self.discount = 0.0
        # elif self.discount+action[1] > 1.0:
        #    self.discount = 1.0
        # else:
        #    self.discount+=action[1]
        
        self.day+=1
        # code for random action testing
        """ random_action = self.action_space.sample()
        if self.mu+random_action[0]*3.0 < 0.0:
            self.mu = 0.0
        elif self.mu+random_action[0]*3.0 > 90.0:
            self.mu = 90.0
        else:
            self.mu+=random_action[0]*3.0

        if self.sigma+random_action[1]*2.0 < 0.0:
            self.sigma = 0.0
        elif self.sigma+random_action[1]*2.0 > 50.0:
            self.sigma = 50.0
        else:
            self.sigma+=random_action[1]*2.0
        if self.A+random_action[2] < 0.0:
            self.A = 0.0
        elif self.A+random_action[2] > 20.0:
            self.A = 20.0
        else:
            self.A+=random_action[2] """
        
        # creating a dictionary of the toll profile parameters after having applied the action
        action = {'mu': self.mu, 'sigma': self.sigma, 'A': self.A}
        #action = {'mu': self.mu, 'sigma': self.sigma, 'A': self.A, 'discount': self.discount}
        #action1 = {'mu': action[0], 'sigma': action[1], 'A': action[2]} # absolute action
        #action = {'mu': 0.0, 'sigma': 0.0, 'A': 0.0} # no action

        # simulating one day on current simulation object
        tao_interval_information, sw, cs, gc, tc, ttcs, individual_cs, individual_sw, income_class, toll_paid, Accumulation, Event_list = self.sim.step(action) # day idx, number of users, toll paid, travel time
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
        gini_eps = gini(np.array(individual_cs))
        self.gini_eps = np.append(self.gini_eps, gini_eps)
        #self.discount_eps = np.append(self.discount_eps, action["discount"])
        
        
        #if self.reward_type == "tt":
        #    reward = 100000-np.sum(tao_interval_information[3]) # computing daily reward
        #elif self.reward_type == "sw":
        #    reward = sw
        reward = np.zeros(2)
        #reward[0] = 100000-np.sum(tao_interval_information[3]) # computing daily reward
        #print(individual_cs)
        reward[0] = 1 - gini_eps
        #reward[0] = (100 - (np.percentile(individual_cs, 95) - np.percentile(individual_cs, 5)))*0.01
        #reward[0] = (sw + 150000)*0.00001 #SW
        reward[1] = (sw + 150000)*0.00001 #SW

        #reward[0] = (sw + 150000)*0.0001 #SW
        #reward[1] = (sw + 150000)*0.0001 #SW
        
        # creating observation to pass to the model
        #observation = np.vstack([tao_interval_information[0],tao_interval_information[1],\
        # tao_interval_information[2],tao_interval_information[3]]) # day idx, number of users, toll paid, travel time
        #observation = np.vstack([tao_interval_information[0],tao_interval_information[1],tao_interval_information[3]]) # day idx, number of users, travel time
        #observation = np.vstack([tao_interval_information[0],tao_interval_information[1],tao_interval_information[2]]) # day idx, number of users, toll paid
        #observation = np.vstack([tao_interval_information[0],tao_interval_information[1]]) # day idx, number of users
        #observation = np.vstack([tao_interval_information[0],tao_interval_information[1]])
        #observation = np.vstack([tao_interval_information[0],tao_interval_information[1],prev_action["mu"],prev_action["sigma"],prev_action["A"]]) # day idx, number of users + action from the previous day, (try tt instead of no users)
        #observation = np.vstack([tao_interval_information[0],tao_interval_information[1],np.full(15,self.A)]) # day idx, number of users + action from the previous day, (try tt instead of no users)
        observation = np.vstack([tao_interval_information[1],np.full(15,self.A)]) #number of users + action from the previous day, (try tt instead of no users)
        #observation = np.vstack([tao_interval_information[1],np.full(15,self.A),np.full(15,self.discount)]) # No users, current policy


        # needed for rl function
        info = {}
        
        # finishing episode after 30 days
        if self.sim.get_day()==self.number_of_days:
            terminated = True
        else:
            terminated = False

        truncated = False #For now
        return observation.ravel(), np.array(reward), terminated, truncated, info
        
    # reset environment    
    def reset(self,seed=None, options=""):

        #self._np_random, seed = seeding.np_random(seed)
        #np.random.seed(seed=59) # tried 2
        print("Episode: " + str(self.eps_no))
        self.eps_no +=1
        # reset toll
        #self.mu = random.random()*90.0 #0.0 to 90.0
        #self.sigma = random.random()*50.0 #0.0 to 50.0
        self.A = random.random()*20.0 #0.0 to 20.0
        self.discount = random.random()

        #self.A = 0
        
        
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
            self.discount_all_eps = np.append(self.discount_all_eps, self.discount_eps)

        
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
            self.discount_eps = np.array([])
        
        self.first_ep = False
        self.sim = Simulation(self.params)
        #observation = np.zeros((2,15))
        #observation = np.zeros((15,3))
        observation = np.zeros((15,2))
        #observation = np.zeros((2,15))
        #observation = [0, 0,]
        info = {}
        self.day = 0

        return observation.ravel(), info


    # defining functions to get statistics from environment
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

    def get_accumulation(self):
        return np.array(self.accumulation_all_eps)
    
    def get_income(self):
        return np.array(self.income_all_eps)
    
    def get_toll(self):
        return np.array(self.toll_all_eps)

    def get_params(self):
        params = {'A': self.A, 'mu': self.mu, 'sigma': self.sigma}
        return params

    def get_gini(self):
        return np.array(self.gini_all_eps)

    def get_discount(self):
        return np.array(self.discount_all_eps)

    def set_params(self, params):
        self.A = params['A']
        self.mu = params['mu']
        self.sigma = params['sigma']

    def set_capacity(self, cap):
        self.sim.set_capacity(cap)

    def render(self, mode):
        pass

