import sys
sys.path.append("C:/modpods")
import modpods
import pystorms
import pyswmm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import control as ct
import dill as pickle
import datetime


np.set_printoptions(precision=3,suppress=True)

# options are: 'correct','granger','transfer-entropy','ccm','uncontrolled'
topo_infer_method = 'uncontrolled' 
verbose = False
plot = False

# defining the rainfall forcing only needs to be done once. comment this out once the files are generated for each rainfall condition
print("evaluating", topo_infer_method)
# read in the csv file "G:\My Drive\EECS_563_Hybrid_Systems\project\provably-safe-hybrid-controller\PF_Depth_English_PDS.csv" and skip the first 13 lines
precip_freq_estimates = pd.read_csv("C:/provably-safe-hybrid-controller/PF_Depth_English_PDS.csv", skiprows=13,index_col=0)
'''
#print(precip_freq_estimates)
# record how many 5 minute intervals are in each duration
duration_intervals = [1, 2, 3, 6, 12, 12*2, 12*3, 12*6, 12*12, 12*24, 12*24*2, 12*24*3, 12*24*4, 12*24*7, 12*24*10, 12*24*20, 12*24*30, 12*24*45, 12*24*60]
#print(duration_intervals)

for duration_idx, duration in enumerate(precip_freq_estimates.index):
    for recurrence_intervals in precip_freq_estimates.columns:
        
        rainfall_data = pd.DataFrame(index = [f"00:{5*i:02d}" for i in range(duration_intervals[duration_idx])], columns= ['in/hr'])

        if duration_intervals[duration_idx] < 2:
            # if the duration is 5 minutes, just use the single value
            rainfall_data['in/hr'] = precip_freq_estimates.loc[duration, recurrence_intervals]
        elif duration_intervals[duration_idx] < 3: # ten minutes, just split it in half
            rainfall_data['in/hr'] = [precip_freq_estimates.loc[duration, recurrence_intervals]/2, precip_freq_estimates.loc[duration, recurrence_intervals]/2]
        elif duration_intervals[duration_idx] < 4: # fifteen minutes, split it quarter, half, quarter
            rainfall_data['in/hr'] = [precip_freq_estimates.loc[duration, recurrence_intervals]/4, precip_freq_estimates.loc[duration, recurrence_intervals]/2, precip_freq_estimates.loc[duration, recurrence_intervals]/4]
        else:
            # create a triangle with its peak at the middle of rainfall_data's index and total area under the curve of 1
            rainfall_data['in/hr'] = np.concatenate((np.linspace(0.1,1,int(duration_intervals[duration_idx]/2)), np.linspace(1,0,int(duration_intervals[duration_idx]/2))))
        # normalize
        rainfall_data['in/hr'] = rainfall_data['in/hr']/sum(rainfall_data['in/hr'])
        # confirm sum is 1 now
        print(sum(rainfall_data['in/hr']))
        # now multiply by the rainfall depth indicated in the dataframe
        rainfall_data['in/hr'] = rainfall_data['in/hr']*precip_freq_estimates.loc[duration, recurrence_intervals]
        print(rainfall_data)
        print("total rainfall (inches)")
        print(sum(rainfall_data['in/hr']))
        print(precip_freq_estimates.loc[duration, recurrence_intervals])
        # save the rainfall data to a file
        rainfall_data.to_csv(f'forcing/rainfall_data_{duration}_{recurrence_intervals}.dat', sep='\t', header=False)

'''


# for debugging
#precip_freq_estimates = precip_freq_estimates.iloc[0:2,0:2]



# Specify the maximum depths for each basin we are controlling
# these are set to 0.5 multiplied by the surcharge depth in the swmm model so that we can accurately record exceeding the thresholds.
basin_max_depths = [5., 5., 10. , 13.72/2 ] # feet
valve_max_flows = np.ones(4)*3.9 # cfs

# these are scored as max(value/threshold). so closer to 0 is better, anything more than 1 indicates a violation
max_flow_exceedance = pd.DataFrame(index = precip_freq_estimates.index, columns=precip_freq_estimates.columns,data=-1.0)
max_filling_degree = pd.DataFrame(index = precip_freq_estimates.index, columns=precip_freq_estimates.columns,data=-1.0)
outfall_TSS_loading = pd.DataFrame(index = precip_freq_estimates.index, columns=precip_freq_estimates.columns,data=-1.0)


# all of the controllers are linear feedback with the same state and input penalties Q and R
# the observers will be the same for all controllers as well
# so the only place these will differ is loading a different plant approximation
# tuning will be based on the correct topology controller

if topo_infer_method == 'uncontrolled': # load the granger approximation (we won't actually use the commands)
    with open("C:/blind_swmm_controller_gamma/plant_approx_granger.pickle", 'rb') as handle:
        lti_plant_approx = pickle.load(handle)
else:
    # load the plant approximation
    with open("C:/blind_swmm_controller_gamma/plant_approx_"+str(topo_infer_method) + ".pickle", 'rb') as handle:
        lti_plant_approx = pickle.load(handle)


# define the observer-based-compensator gains

# check the eigenvalues of the plant approximation to verify that A is internally stable
plant_eigenvalues,_ = np.linalg.eig(lti_plant_approx['A'].values)

# go through all the plant approximation matrices and cast them all to floats
for letter in ["A","B","C"]:
    lti_plant_approx[letter] = lti_plant_approx[letter].astype(float) / 12
    # divided by 12 because of the scaling of "dt" in the plant approximation
        
# define the cost function
Q = np.eye(len(lti_plant_approx['A'].columns))*0 # we don't want to penalize the transition states as their magnitude doesn't have directly tractable physical meaning
# bryson's rule based on the maxiumum depth of each basin

for asset_index in range(len(lti_plant_approx['C'].index)):
    Q[lti_plant_approx['A'].columns.get_loc(lti_plant_approx['C'].index[asset_index]),lti_plant_approx['A'].columns.get_loc(lti_plant_approx['C'].index[asset_index])] = 1 / ((basin_max_depths[asset_index])**2 )
    
    if asset_index == 0: # node 1
        Q[lti_plant_approx['A'].columns.get_loc(lti_plant_approx['C'].index[asset_index])
            ,lti_plant_approx['A'].columns.get_loc(lti_plant_approx['C'].index[asset_index])] = Q[lti_plant_approx['A'].columns.get_loc(lti_plant_approx['C'].index[asset_index])
            ,lti_plant_approx['A'].columns.get_loc(lti_plant_approx['C'].index[asset_index])]*0.3
    if asset_index == 1: # node 4
        Q[lti_plant_approx['A'].columns.get_loc(lti_plant_approx['C'].index[asset_index])
            ,lti_plant_approx['A'].columns.get_loc(lti_plant_approx['C'].index[asset_index])] = Q[lti_plant_approx['A'].columns.get_loc(lti_plant_approx['C'].index[asset_index])
            ,lti_plant_approx['A'].columns.get_loc(lti_plant_approx['C'].index[asset_index])]*3
    if asset_index == 2: # node 6
        Q[lti_plant_approx['A'].columns.get_loc(lti_plant_approx['C'].index[asset_index])
            ,lti_plant_approx['A'].columns.get_loc(lti_plant_approx['C'].index[asset_index])] = Q[lti_plant_approx['A'].columns.get_loc(lti_plant_approx['C'].index[asset_index])
            ,lti_plant_approx['A'].columns.get_loc(lti_plant_approx['C'].index[asset_index])]*0.001
    if asset_index == 3: # node 10
        Q[lti_plant_approx['A'].columns.get_loc(lti_plant_approx['C'].index[asset_index])
            ,lti_plant_approx['A'].columns.get_loc(lti_plant_approx['C'].index[asset_index])] = Q[lti_plant_approx['A'].columns.get_loc(lti_plant_approx['C'].index[asset_index])
            ,lti_plant_approx['A'].columns.get_loc(lti_plant_approx['C'].index[asset_index])]*1.5

# threshold on flows at 0.11 m^3 / s which is 3.9 cfs
R = np.eye(4) / (3.9**2) # bryson's rule on maximum allowable flow
#R_flood = np.eye(4) / (3.9**2) # bryson's rule on maximum allowable flow
#R_treat = np.eye(4) / (3.9**2) # want zero flows through orifice 1 (leaving subcatchment)
#R_treat[0,0] = 1 / (0.1**2) # greatly reduce flows out of the catchment through orifice 1

state_weighting = 0.007 # weight of state penalties (flooding) vs input penalties (flows)
# this weighting is necessary because control violations are transient while state violations are persistent

Q = Q * state_weighting 

# define the system
# find the state feedback gain for the linear quadratic regulator
print("defining controller")
K,S,E = ct.lqr(lti_plant_approx['A'],lti_plant_approx['B'].values,Q,R) # the remaining columsn are distrubances, only valves 1-4 are controlled

feedback_poles,_ = np.linalg.eig(lti_plant_approx['A'].values - lti_plant_approx['B'].values@K)
#print(feedback_poles)

# does Tyv have a right inverse? if not, we can't set a precompensation gain G to achieve steady state tracking (up to model error)
A_full_rank = np.linalg.matrix_rank(lti_plant_approx['A'] ) == len(lti_plant_approx['A'].columns)
# already know that C has rank 4, so we only need to check the rank of A
# abbreviate just for legibility
A = lti_plant_approx['A'].values
B = lti_plant_approx['B'].values
C = lti_plant_approx['C'].values

# reference commands
G = np.linalg.inv( C @ np.linalg.inv(-A + B@K) @ B )    
  
r = 0.2*np.array(basin_max_depths).reshape(-1,1) # desire empty
# 20% of the maximum allowable depth is 10% of the flooding depth

precompensation = G@r

# define the observer gain

print("defining observer")

# observer is the same for the flooding and quality controllers
    
# had a ton of trouble with control.lqe giving nonsensical errors / results. try using LQR in the dual form to solve for the observer gain
measurement_noise = 1
process_noise = 200 # we trust the measurements a lot more than the model
L, S, E = ct.lqr(np.transpose(lti_plant_approx['A'].values),
                    np.transpose(lti_plant_approx['C'].values),
                    process_noise*np.eye(len(lti_plant_approx['A'].columns)),
                    measurement_noise*np.eye(len(lti_plant_approx['C'].index)))
L = np.transpose(L)
    
observer_poles,_ =  np.linalg.eig(lti_plant_approx['A'].values - L@lti_plant_approx['C'].values)
#print(observer_poles)
    
# is the observer internally stable? 

obc_poles,_ = np.linalg.eig(lti_plant_approx['A'].values - lti_plant_approx['B'].values@K - L@lti_plant_approx['C'].values)
   
# convert control command (flow) into orifice open percentage
# per the EPA-SWMM user manual volume ii hydraulics, orifices (section 6.2, page 107) - https://nepis.epa.gov/Exe/ZyPDF.cgi/P100S9AS.PDF?Dockey=P100S9AS.PDF 
# all orifices in gamma are "bottom"
Cd = 0.65 # happens to be the same for all of them
Ao = 1 # area is one square foot. again, happens to be the same for all of them. 
g = 32.2 # ft / s^2
# the expression for discharge is found using Torricelli's equation: Q = Cd * (Ao*open_pct) sqrt(2*g*H_e)
# H_e is the effective head in feet, which is just the depth in the basin as the orifices are "bottom"
# to get the action command as a percent open, we solve as: open_pct = Q_desired / (Cd * Ao * sqrt(2*g*H_e))

# load the training dt from the text file
with open("C:/provably-safe-hybrid-controller/swmm_training_dt.txt", 'r') as f:
    training_dt = float(f.read())
    
# ensuring flood avoidance during the most extreme storms
# drop every column except for '1000' in precip_freq_estimates
#precip_freq_estimates = precip_freq_estimates.drop(columns=['1','2','5','10','25','50','100','200','500'])
# drop the very short storms which won't cause flooding
#precip_freq_estimates = precip_freq_estimates.drop(index=['5-min','10-min','15-min','30-min','60-min','2-hr'])
for duration_idx, duration in enumerate(precip_freq_estimates.index):
    for recurrence_idx, recurrence_intervals in enumerate(precip_freq_estimates.columns):
        #duration = '2-day'
        #recurrence_intervals = '1000'
        print(duration, recurrence_intervals)  
        # overwrite "G:\My Drive\EECS_563_Hybrid_Systems\project\provably-safe-hybrid-controller\rainfall_data.dat" with the current rainfall data
        rainfall_data = pd.read_csv(f'forcing/rainfall_data_{duration}_{recurrence_intervals}.dat', sep='\t', header=None)

        rainfall_data.to_csv('rainfall_data.dat', sep='\t', header=False,index=False)

            
        env = pystorms.scenarios.gamma()
        env.env.sim = pyswmm.simulation.Simulation(r"C:\\blind_swmm_controller_gamma\\gamma.inp")
        # only observing and controlling 1, 4, 6, and 10
        env.config['states'] = [env.config['states'][0], env.config['states'][3], env.config['states'][5], env.config['states'][9]]
        env.config['action_space'] = [env.config['action_space'][0], env.config['action_space'][3], env.config['action_space'][5], env.config['action_space'][9]]

            
        if duration_idx < 6: # durations from 5 minutes to 2 hours
            env.env.sim.end_time = env.env.sim.start_time + datetime.timedelta(days = 7) # simulate a week
        elif duration_idx < 12: # durations from 3 hours to 3 days
            env.env.sim.end_time = env.env.sim.start_time + datetime.timedelta(days = 30) # simulate a month
        else: # storms longer than four days
            pass # use default simulation length (two and a half months)
        env.env.sim.start()
        done = False
            
        u = np.zeros((4,1) ) # start with all orifices completely closed
        u_open_pct = np.zeros((4,1)) # start with all orifices completely closed
        xhat = np.zeros((len(lti_plant_approx['A'].columns),1)) # initial state estimate

        if topo_infer_method == 'uncontrolled':
            u_open_pct = np.ones((4,1)) # everything is open
            
        steps = 0 # make sure the estimator and controller operate at the frequency the approxiation was trained at
            
        last_eval = datetime.datetime(2000,1,1) # initialize to a date in the past
        while not done:
            # the training frequency of the model was five minutes, this is also more realistic
            if env.env.sim.current_time.minute % 5 == 0 and (env.env.sim.current_time > last_eval + datetime.timedelta(minutes=2)):
                last_eval = env.env.sim.current_time
                # Query the current state of the simulation
                observables = env.state()
                y_measured = observables.reshape(-1,1) # depths at 1, 4, 6, and 10
                
                # for updating the plant, calculate the "u" that is actually applied to the plant, not the desired control input
                for idx in range(len(u)):
                    u[idx,0] = Cd*Ao*u_open_pct[idx,0]*np.sqrt(2*g*observables[idx]) # calculate the actual flow through the orifice

                # update the observer based on these measurements -> xhat_dot = A xhat + B u + L (y_m - C xhat)
                state_evolution = lti_plant_approx['A'].values @ xhat 
                impact_of_control = lti_plant_approx['B'].values @ u 
                yhat = lti_plant_approx['C'].values @ xhat # just for reference, could be useful for plotting later
                y_error =  y_measured - yhat # cast observables to be 2 dimensional
                output_updating = L @ y_error 
                xhat_dot =  state_evolution + impact_of_control  + L@y_measured - L@yhat # xhat_dot = A xhat + B u + L (y_m - C xhat)
                yhat_dot = lti_plant_approx['C'].values @ xhat_dot
                xhat += xhat_dot 

                u = -K @ xhat  # state feedback
                u = u + G @ r # precompensation

                u_open_pct = u*-1
    
                for idx in range(len(u)):
                    head = 2*g*observables[idx]
        
                    if head < 0.01: # if the head is less than 0.01 ft, the basin is empty, so close the orifice
                        u_open_pct[idx,0] = 0
                    else:
                        u_open_pct[idx,0] = u[idx,0] / (Cd*Ao * np.sqrt(2*g*observables[idx])) # open percentage for desired flow rate
        
                    if u_open_pct[idx,0] > 1: # if the calculated open percentage is greater than 1, the orifice is fully open
                        u_open_pct[idx,0] = 1
                    elif u_open_pct[idx,0]< 0: # if the calculated open percentage is less than 0, the orifice is fully closed
                        u_open_pct[idx,0] = 0

                # retrieve the total pollutant loading for the link O1 leaving the subcatchment
                # this is our "treatment performance" metric
                total_TSS_loading = pyswmm.Links(env.env.sim)["O1"].total_loading['TSS']
                
                if topo_infer_method == 'uncontrolled':
                    u_open_pct = np.ones((4,1)) # everything is open
                    
                if verbose and env.env.sim.current_time.minute == 0 and env.env.sim.current_time.hour % 2 == 0:
                    print("u_open_pct, yhat, y_measured, y_error")
                    print(np.c_[u_open_pct,yhat, y_measured, y_error])
                    print("current time, end time")
                    print(env.env.sim.current_time, env.env.sim.end_time)
                    print("\n")
   
            
            # Set the actions and progress the simulation
            done = env.step(u_open_pct.flatten())
            steps += 1
      
        # calculate the robustness scores
        # this will be max(value/threshold) for all the depths and flows (even at the uncontrolled nodes)
        #flooding_depths = [10.0,10.0,10.0,10.0,10.0,20.0,10.0,10.0,10.0,13.72,14.96] # feet
        #flow_restriction = 3.9*np.ones(11) # cfs
        # decided not to do it for all eleven because there's some violations we can't control or observe
        # some storms cause flooding at 9 which is an uncontrolled headwater node
        # to keep the indexing easy, just make the uncontrolled nodes have super high thresholds
        flooding_depths = [10.0,1e3,1e3,10.0,1e3,20.0,1e3,1e3,1e3,13.72,14.96e3] # feet
        flow_restriction = 1e3*np.ones(11) # cfs
        for controlled in [0,3,5,9]:
            flow_restriction[controlled] = 3.9 # cfs

        for key_idx, key in enumerate(env.data_log['depthN'].keys()):
            if max(env.data_log['depthN'][key])/flooding_depths[key_idx] == 1.0:
                print("flooding at ", key)
            max_filling_degree.loc[duration, recurrence_intervals] = max(max_filling_degree.loc[duration, recurrence_intervals], max(env.data_log['depthN'][key])/flooding_depths[key_idx])
        for key_idx, key in enumerate(env.data_log['flow'].keys()):
            # print the max flow exceedance for each orifice
            if max(env.data_log['flow'][key])/flow_restriction[key_idx] > 1.0:
                print("flow exceedance at ", key, " = ", max(env.data_log['flow'][key])/flow_restriction[key_idx])
                
            max_flow_exceedance.loc[duration, recurrence_intervals] = max(max_flow_exceedance.loc[duration, recurrence_intervals], max(env.data_log['flow'][key])/flow_restriction[key_idx])
                    
        # print the robustness score for this storm
        #print("robustness")
        #print(robustness_scores.loc[duration, recurrence_intervals])
        print("max_filling_degree (1.0 indicates flooding)")
        print(max_filling_degree.loc[duration, recurrence_intervals])
        print("max_flow_exceedance")
        print(max_flow_exceedance.loc[duration, recurrence_intervals])
            
        print("total TSS loading")
        print(total_TSS_loading)
        outfall_TSS_loading.loc[duration, recurrence_intervals] = total_TSS_loading
            
        if plot:
            obc_data = env.data_log

            fig,axes = plt.subplots(4,2,figsize=(16,8))
            fig.suptitle(topo_infer_method)
            axes[0,0].set_title("Valves",fontsize='xx-large')
            axes[0,1].set_title("Storage Nodes",fontsize='xx-large')

            valves = ["O1","O4","O6","O10"]
            storage_nodes = ["1","4","6","10"]
            cfs2cms = 35.315
            ft2meters = 3.281
            # plot the valves
            for idx in range(4):
                axes[idx,0].plot(obc_data['simulation_time'],np.array(obc_data['flow'][valves[idx]])/cfs2cms,label='LTI Feedback',color='g',linewidth=2)
                # add a dotted red line indicating the flow threshold
                axes[idx,0].hlines(3.9/cfs2cms, obc_data['simulation_time'][0],obc_data['simulation_time'][-1],label='Threshold',colors='r',linestyles='dashed',linewidth=2)
                #axes[idx,0].set_ylabel( str(  str(valves[idx]) + " Flow" ),rotation='horizontal',labelpad=8)
                axes[idx,0].annotate(str(  str(valves[idx]) + " Flow" ),xy=(0.5,0.8),xycoords='axes fraction',fontsize='xx-large')
                if idx == 0:
                    axes[idx,0].legend(fontsize='xx-large')
                if idx != 3:
                    axes[idx,0].set_xticks([])
       

            # plot the storage nodes
            flooding_depth_idx = [0,3,5,9]
            for idx in range(4):
                axes[idx,1].plot(obc_data['simulation_time'],np.array(obc_data['depthN'][storage_nodes[idx]])/ft2meters,label='LTI Feedback',color='g',linewidth=2)
                #axes[idx,1].set_ylabel( str( str(storage_nodes[idx]) + " Depth"),rotation='horizontal',labelpad=8)
                axes[idx,1].annotate( str( str(storage_nodes[idx]) + " Depth"),xy=(0.5,0.8),xycoords='axes fraction',fontsize='xx-large')
    
                # add a dotted red line indicating the depth threshold
                axes[idx,1].hlines(flooding_depths[flooding_depth_idx[idx]]/ft2meters,obc_data['simulation_time'][0],obc_data['simulation_time'][-1],label='Threshold',colors='r',linestyles='dashed',linewidth=2)
                if idx != 3:
                    axes[idx,1].set_xticks([])

            plt.tight_layout()
            plt.savefig(str("C:/blind_swmm_controller_gamma/" + topo_infer_method + "_plots_debugging.png"),dpi=450)
            plt.savefig(str("C:/blind_swmm_controller_gamma/" + topo_infer_method + "_plots_debugging.svg"),dpi=450)
            plt.show()
            
           


        # pickle env.data_log to a file named for the rainfall and control condition
        with open(f'simulation_results/{topo_infer_method}_{duration}_{recurrence_intervals}.pkl', 'wb') as f:
            pickle.dump(env.data_log, f)
            
               
# save the robustness_scores to a csv named for the control condition
#robustness_scores.to_csv('robustness_scores_uncontrolled.csv', sep=',', header=True)
max_filling_degree.to_csv(str("filling_degree_" + topo_infer_method + ".csv"),sep=',', header=True)
max_flow_exceedance.to_csv(str("flow_exceedance_" + topo_infer_method + ".csv"),sep=',', header=True)
outfall_TSS_loading.to_csv(str("outfall_TSS_loading_" + topo_infer_method + ".csv"), sep=',', header=True)
    