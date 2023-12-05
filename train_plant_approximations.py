import sys
#from modpods import topo_from_pystorms
sys.path.append("C:/modpods")
import modpods
import pystorms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import control as ct
import dill as pickle
import pyswmm
import swmm
import datetime
import re
import networkx as nx

# print all columns of pandas dataframes
pd.options.display.max_columns = None

# do a training simulation such that the flows are actually independent of the depths
# in both the uncontrolled and efd scenarios the flows at the orifice are highly coupled to the depths
# overwrite "G:\My Drive\EECS_563_Hybrid_Systems\project\provably-safe-hybrid-controller\rainfall_data.dat" with the current rainfall data
duration = "15-min"
recurrence_interval = "1000"
rainfall_data = pd.read_csv(f'forcing/rainfall_data_{duration}_{recurrence_interval}.dat', sep='\t', header=None)
rainfall_data.to_csv('rainfall_data.dat', sep='\t', header=False,index=False)
            
times = rainfall_data.values[:,0]

df_rain = pd.DataFrame({'times': times})
df_rain['elapsed_minutes'] = df_rain['times'].str.extract(r':(\d+)').astype(int)
df_rain['hours'] = (df_rain['elapsed_minutes'] / 60).astype(int)
df_rain['minutes'] = (df_rain['elapsed_minutes'] % 60).astype(int)
df_rain.drop(['elapsed_minutes'], axis=1, inplace=True)
df_rain.index = pd.Timestamp('2020-6-01') + pd.to_timedelta(df_rain['hours'], unit='h') + pd.to_timedelta(df_rain['minutes'], unit='m')
df_rain.drop(['times', 'hours', 'minutes'], axis=1, inplace=True)
df_rain['rain']=rainfall_data.values[:,1]
print(df_rain)

env = pystorms.scenarios.gamma()
env.env.sim = pyswmm.simulation.Simulation(r"C:\\blind_swmm_controller_gamma\\gamma.inp")
env.env.sim.end_time = datetime.datetime(2020,6,20,0,0) # the file is set up to go much longer than we actually need to here

# just for debugging, give us a super short simulation
env.env.sim.end_time = datetime.datetime(2020,6,8,0,0) # the file is set up to go much longer than we actually need to here


env.env.sim.start()
done = False

# Specify the maximum depths for each basin we are controlling
# these are set to 0.5 multiplied by the surcharge depth in the swmm model so that we can accurately record exceeding the thresholds.
basin_max_depths = [5., 5., 5., 5., 5., 10. , 5., 5. , 5., 13.72/2 , 14.96/2] # feet

valve_max_flows = np.ones(11)*3.9 # cfs
controlled_valves = [0,3,5,9] # valves 1, 4, 6, and 10 are controlled]
actions_characterize = np.ones(11) # start all uncontroleld valves open
actions_characterize[controlled_valves] = 0.0 # close all the controlled valves
step = 0
print("running characterization simulation")
last_eval = datetime.datetime(2020,6,2,hour=0) # don't start opening valves until all the runoff has been stored
i = 0 # for iterating through the valves
while not done:

    if env.env.sim.current_time.hour % 12 == 0 and env.env.sim.current_time.minute == 0 and (env.env.sim.current_time > last_eval + datetime.timedelta(minutes=2)):
        last_eval = env.env.sim.current_time
        actions_characterize = np.ones(11) # nothing else is controlled
        # select i to be a random integer amongst the controllable valves which are 1, 4, 6, and 10
        i = np.random.randint(0,4)
        valve_to_open = controlled_valves[i]
        actions_characterize[valve_to_open] = 0.6/(i+1) # open one valve
        for j in range(4):
            if j != i:
                actions_characterize[controlled_valves[j]] = 0.0 # close the other valves
        
            
        
    elif env.env.sim.current_time.hour % 6 == 0 and env.env.sim.current_time.minute == 0 and (env.env.sim.current_time > last_eval + datetime.timedelta(minutes=2)):
        actions_characterize = np.ones(11) # start all uncontroleld valves open
        actions_characterize[controlled_valves] = 0.0 # close all the controlled valves

    if env.env.sim.current_time.day >= 15:
        actions_characterize = np.ones(11) # open everything to allow draining
        


    # Set the actions and progress the simulation
    done = env.step(actions_characterize)
    step += 1


random_perf = sum(env.data_log["performance_measure"])
print("performance of characterization:")
print("{:.4e}".format(random_perf))

obc_data = env.data_log

fig,axes = plt.subplots(4,2)
#fig.suptitle("Characterization experiment")
axes[0,0].set_title("Valves",fontsize='xx-large')
axes[0,1].set_title("Storage Nodes",fontsize='xx-large')

valves = ["O1","O4","O6","O10"]
storage_nodes = ["1","4","6","10"]
cfs2cms = 35.315
ft2meters = 3.281
# plot the valves
for idx in range(4):
    axes[idx,0].plot(obc_data['simulation_time'],np.array(obc_data['flow'][valves[idx]])/cfs2cms,color='k',linewidth=2)
    # add a dotted red line indicating the flow threshold
    #axes[idx,0].hlines(3.9/cfs2cms, obc_data['simulation_time'][0],obc_data['simulation_time'][-1],label='Threshold',colors='r',linestyles='dashed',linewidth=2)
    #axes[idx,0].set_ylabel( str(  str(valves[idx]) + " Flow" ),rotation='horizontal',labelpad=8)
    axes[idx,0].annotate(str(  str(valves[idx]) + " Flow" ),xy=(0.5,0.8),xycoords='axes fraction',fontsize='xx-large')
    #if idx == 0:
        #axes[idx,0].legend(fontsize='xx-large')
    if idx != 3:
        axes[idx,0].set_xticks([])
       

# plot the storage nodes
for idx in range(4):

    axes[idx,1].plot(obc_data['simulation_time'],np.array(obc_data['depthN'][storage_nodes[idx]])/ft2meters,color='k',linewidth=2)
    #axes[idx,1].set_ylabel( str( str(storage_nodes[idx]) + " Depth"),rotation='horizontal',labelpad=8)
    axes[idx,1].annotate( str( str(storage_nodes[idx]) + " Depth"),xy=(0.5,0.8),xycoords='axes fraction',fontsize='xx-large')
    
    # add a dotted red line indicating the depth threshold
    #axes[idx,1].hlines(basin_max_depths[idx]/ft2meters,obc_data['simulation_time'][0],obc_data['simulation_time'][-1],label='Threshold',colors='r',linestyles='dashed',linewidth=2)
    if idx != 3:
        axes[idx,1].set_xticks([])

plt.tight_layout()
plt.savefig("C:/blind_swmm_controller_gamma/characterization_experiment.png",dpi=450)
plt.savefig("C:/blind_swmm_controller_gamma/characterization_experiment.svg",dpi=450)
#plt.show()
plt.close('all')



training_data = env.data_log
training_flows = pd.DataFrame.from_dict(training_data['flow'])
training_flows.columns = env.config['action_space']
training_depthN = pd.DataFrame.from_dict(training_data['depthN'])
training_depthN.columns = env.config['states']
training_response = pd.concat([training_flows, training_depthN], axis=1)
training_response.index = env.data_log['simulation_time']
# put the rainfall in there


print(training_response)


# for the columns of training_response which do not contain the string "O" (i.e. the depths)
# if the current column name is "X", make it "(X, depthN)"
# this is because the modpods library expects the depths to be named "X, depthN"
# where X is the name of the corresponding flow
for col in training_response.columns:
    if "O" in col: # the orifices  
        # if the index on O isn't one of the controllable valves, drop it
        if int(col[1:]) - 1 not in controlled_valves:
            training_response.drop(columns=col,inplace=True)
        else:
            training_response.rename(columns={col: (col, "flow")}, inplace=True)
    
    else: # the depths
        # cast the tuple column name to a string and extract the interger from the string
        valve_num = int(re.findall(r'\d+', str(col))[0])
        # if this isn't the depth of an asset controlled by a valve, drop it
        if valve_num - 1 not in controlled_valves:
            training_response.drop(columns=col,inplace=True)
    

# for debugging resample to a coarser time step (native resolution is about one minute but not consistent)
# need a consistent time step for modpods
orig_index_length = len(training_response.index)
training_response = training_response.resample('5T',axis='index').min().copy(deep=True)

print(training_response)
#resampled_rainfall = df_rain.resample('5T',axis='index').mean().copy(deep=True)
# fill any na's in resampled rainfall with zeros
#resampled_rainfall.fillna(0.0,inplace=True)
training_dt =  orig_index_length / len(training_response.index)

# since rainfall is not provided, cut off the first day of the simulation
training_response = training_response[24*12:] # 24 hours * 12 5-minute intervals per hour

print(training_response)

#print(training_response)

# now how many rows have nan's in them?
print("number of rows with nan's in them:")
print(training_response.isna().sum().sum())
training_response.fillna(0.0,inplace=True)
print(training_response)

dependent_columns = training_response.columns[-4:] # the depths of the controlled basins
independent_columns = training_response.drop(columns = dependent_columns).columns # everything else (flows at the controlled basins)

training_response.plot(subplots=True,figsize=(10,10))
plt.show(block=False)
plt.pause(10)
plt.close('all')

# read the topology from the swmm file (this is much cheaper)
env.config['states'] = dependent_columns
env.config['action_space'] = independent_columns 
# the default is controlling all 11 orifices so we need to edit the environment
print("defining topology")

blind=True
topo_infer_method = 'granger'
# options are: 'granger', 'ccm', 'transfer-entropy'
if blind:
    print("inferring topology from data")
else:
    print("inferring topology from the swmm file, this is quick")

if not blind:
    swmm_topo = modpods.topo_from_pystorms(env)
    #swmm_topo['rain'] = 'd' 

    print(swmm_topo)

    # learn the dynamics now, or load a previously learned model

    # learn the dynamics from the trainingd response
    print("learning dynamics")
    lti_plant_approx_seeing = modpods.lti_system_gen(swmm_topo, training_response, 
                                                     independent_columns= independent_columns,
                                                     dependent_columns = dependent_columns, max_iter = 250,
                                                     swmm=True,bibo_stable=True,max_transition_state_dim=25)
    # pickle the plant approximation to load later
    with open("C:/blind_swmm_controller_gamma/plant_approx_correct.pickle", 'wb') as handle:
        pickle.dump(lti_plant_approx_seeing, handle)
    # save the training dt as a text file
    with open("C:/blind_swmm_controller_gamma/swmm_training_dt.txt", 'w') as handle:
        handle.write(str(training_dt))  

        '''
    # load the plant approximation
    with open("C:/blind_swmm_controller_gamma/plant_approx_correct.pickle", 'rb') as handle:
        lti_plant_approx_seeing = pickle.load(handle)
        '''
        
    lti_plant_approx = lti_plant_approx_seeing
    
if blind:
    print("learning topology")
    topo, total_graph = modpods.infer_causative_topology(training_response,dependent_columns=dependent_columns,
                                            independent_columns=independent_columns,graph_type='Weak-Conn',
                                            verbose=True, swmm=True, max_iter = 10,method=topo_infer_method)
    
  
    print(topo)

    # how does this compare to the correct topology derived from the swmm file?
    swmm_topo = modpods.topo_from_pystorms(env)
    print(swmm_topo)
    
    swmm_topo_graph = swmm_topo.copy(deep=True) 
    swmm_topo_graph.values[:,:] = 0 # no connection
    swmm_topo_graph = swmm_topo_graph.astype('int64')
    
    topo_graph = topo.copy(deep=True)
    topo_graph.values[:,:] = 0
    topo_graph = topo_graph.astype('int64')
    
    
    # wherever there is an "i" in either graph, make it a 1
    swmm_topo_graph[swmm_topo == 'i'] = 1
    topo_graph[topo == 'i'] = 1
    # wherever there's a "d" in either graph, make it a 2
    swmm_topo_graph[swmm_topo == 'd'] = 2
    topo_graph[topo == 'd'] = 2

    # transposing the graphs reverses them and generates an influecne diagram
    swmm_topo_graph = swmm_topo_graph.transpose()
    topo_graph = topo_graph.transpose()

    # for every entry in the index which isn't in the columns, add it as a new column filled with zeros
    for idx in swmm_topo_graph.index:
        if idx not in swmm_topo_graph.columns:
            swmm_topo_graph[idx] = 0
    for idx in topo_graph.index:
        if idx not in topo_graph.columns:
            topo_graph[idx] = 0
    
    # cast the columns and indices of the dataframes to strings
    swmm_topo_graph.columns = swmm_topo_graph.columns.astype(str)
    swmm_topo_graph.index = swmm_topo_graph.index.astype(str)
    topo_graph.columns = topo_graph.columns.astype(str)
    topo_graph.index = topo_graph.index.astype(str)


    print(swmm_topo_graph)
    print(topo_graph)
    
    # use networkx to draw the networks identified by the different methods
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].set_title("Correct Topology",fontsize='xx-large')
    axes[1].set_title("Inferred Topology",fontsize='xx-large')
    # draw the correct topology
    G = nx.from_pandas_adjacency(swmm_topo_graph, create_using=nx.DiGraph)
    pos = nx.planar_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=1000, ax=axes[0])
    nx.draw_networkx_labels(G, pos, font_size=20, ax=axes[0])
    nx.draw_networkx_edges(G, pos, arrows=True, ax=axes[0],edge_cmap='viridis', edge_vmin = 1,edge_vmax = 2,arrowsize=50)
    # draw the inferred topology
    G = nx.from_pandas_adjacency(topo_graph, create_using=nx.DiGraph)
    #pos = nx.planar_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=1000,  ax=axes[1])
    nx.draw_networkx_labels(G, pos, font_size=20, ax=axes[1])
    nx.draw_networkx_edges(G, pos, arrows=True, ax=axes[1],edge_cmap='viridis', edge_vmin = 1,edge_vmax = 2,arrowsize=50)
    plt.tight_layout()
    plt.savefig(str("C:/blind_swmm_controller_gamma/topo_comparison_" + str(topo_infer_method) + ".png"))
    plt.savefig(str("C:/blind_swmm_controller_gamma/topo_comparison_" + str(topo_infer_method) + ".svg"))
    plt.show()
    plt.close()
    
    
    
    # learn the dynamics from the trainingd response
    print("learning dynamics")
    lti_plant_approx_blind = modpods.lti_system_gen(topo, training_response, 
                                                     independent_columns= independent_columns,
                                                     dependent_columns = dependent_columns, max_iter = 0,
                                                     swmm=True,bibo_stable=True,max_transition_state_dim=25)
    # pickle the plant approximation to load later
    with open("C:/swmm_blind_controller_gamma/plant_approx_"+str(topo_infer_method) + ".pickle", 'wb') as handle:
        pickle.dump(lti_plant_approx_blind, handle)
    # save the training dt as a text file
    with open("C:/swmm_blind_controller_gamma/swmm_training_dt.txt", 'w') as handle:
        handle.write(str(training_dt))  

        '''
    # load the plant approximation
    with open("C:/swmm_blind_controller_gamma/plant_approx_"+str(topo_infer_method) + ".pickle", 'rb') as handle:
        lti_plant_approx_seeing = pickle.load(handle)
        '''
        
    lti_plant_approx = lti_plant_approx_blind



# evaluate the plant approximation accuracy
# only plot the depths at 1, 2, 3, and 4
# the forcing is the flows at O1, O2, O3, and O4
# is the plant approximation internally stable?
plant_eigenvalues,_ = np.linalg.eig(lti_plant_approx['A'].values)

# cast the columns of dataframes to strings for easier indexing
training_response.columns = training_response.columns.astype(str)
dependent_columns = [str(col) for col in dependent_columns]
independent_columns = [str(col) for col in independent_columns]
# reindex the training_response to an integer step
training_response.index = np.arange(0,len(training_response),1)

approx_response = ct.forced_response(lti_plant_approx['system'], U=np.transpose(training_response[independent_columns].values), T=training_response.index.values)
approx_data = pd.DataFrame(index=training_response.index.values)
for idx in range(4):
    approx_data[dependent_columns[idx]] = approx_response.outputs[controlled_valves[idx]-1][:]

output_columns = dependent_columns[0:4] # depths at storage nodes


fig, axes = plt.subplots(4, 1, figsize=(10, 40))

for idx in range(len(output_columns)):
    axes[idx].plot(training_response[output_columns[idx]],label='actual')
    axes[idx].plot(approx_data[output_columns[idx]],label='approx')
    if idx == 0:
        axes[idx].legend(fontsize='x-large',loc='best')
    axes[idx].set_ylabel(output_columns[idx],fontsize='large')
    if idx == len(output_columns)-1:
        axes[idx].set_xlabel("time",fontsize='x-large')
# label the left column of plots "training"
axes[0].set_title("outputs",fontsize='xx-large')
plt.tight_layout()
if not blind:
    plt.savefig("C:/swmm_blind_controller_gamma/swmm_plant_approx_seeing.png")
    plt.savefig("C:/swmm_blind_controller_gamma/swmm_plant_approx_seeing.svg")
if blind:
    plt.savefig(str("C:/swmm_blind_controller_gamma/swmm_plant_approx_" + str(topo_infer_method) + ".png"))
    plt.savefig(str("C:/swmm_blind_controller_gamma/swmm_plant_approx_" + str(topo_infer_method) + ".svg"))

plt.show()
plt.close()

print("plant estimation complete")
