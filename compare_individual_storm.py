from matplotlib.lines import lineStyles
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

from seaborn._core.properties import LineStyle

# define the duration and recurrence interval of the storm
duration = '_24-hr_' # include the underscores to avoid including durations with prefixes or suffixes
recurrence_intervals = '_100.' # include the dot to avoid including recurrence intervals that append zeros to this (i.e., 1 would include 10, 100, and 1000)
cartoon=False
#"G:\My Drive\EECS_563_Hybrid_Systems\project\provably-safe-hybrid-controller\simulation_results\equal_filling_2-day_1.pkl"

# define the path to the simulation results
path = "C:/blind_swmm_controller_gamma/simulation_results/"

# find all the files within the path that include the duration and recurrence interval
files = []
for file in os.listdir(path):
    if duration in file and recurrence_intervals in file:
        files.append(file)
        
print("found", len(files), " simulation results files")

# for each file, load the data 
correct = False
granger = False
transfer_entropy = False
ccm = False
uncontrolled = False
for file in files:
    if "correct" in file:
        print(file)
        correct_data = pd.read_pickle(path + file)
        correct = True
    elif "granger" in file:
        print(file)
        granger_data = pd.read_pickle(path + file)
        granger = True
    elif "transfer-entropy" in file:
        print(file)
        transfer_entropy_data = pd.read_pickle(path + file)
        transfer_entropy = True
    elif "ccm" in file:
        print(file)
        ccm_data = pd.read_pickle(path + file)
        ccm = True
    elif "uncontrolled" in file:
        print(file)
        uncontrolled_data = pd.read_pickle(path + file)
        uncontrolled = True
    else:
        print("error: file not recognized")
        print(file)


fig,axes = plt.subplots(4,2,figsize=(16,8))
title = "Comparison for duration " + str(duration[1:-1]) + " and recurrence interval " + str(recurrence_intervals[1:-1]) + " years"
#fig.suptitle(title)
axes[0,0].set_title("Valves",fontsize='xx-large')
axes[0,1].set_title("Storage Nodes",fontsize='xx-large')

basin_max_depths = [5., 5., 5., 5.] # feet
valve_max_flows = np.ones(4)*3.9 # cfs
valves = ["O1","O4","O6","O10"]
storage_nodes = ["1","4","6","10"]
cfs2cms = 35.315
ft2meters = 3.281

flooding_depths = [10.0,1e3,1e3,10.0,1e3,20.0,1e3,1e3,1e3,13.72,14.96e3] # feet
flooding_depth_idx = [0,3,5,9]

# plot the valves
for idx in range(4):
    if uncontrolled:
        axes[idx,0].plot(uncontrolled_data['simulation_time'],np.array(uncontrolled_data['flow'][valves[idx]])/cfs2cms,label='Uncontrolled',color='k',linewidth=2,alpha=0.7)
    if correct:
        axes[idx,0].plot(correct_data['simulation_time'],np.array(correct_data['flow'][valves[idx]])/cfs2cms,label='Correct',color='b',linewidth=2,alpha=0.7)
    if granger:
        axes[idx,0].plot(granger_data['simulation_time'],np.array(granger_data['flow'][valves[idx]])/cfs2cms,label='Granger',color='g',linewidth=2,alpha=0.7,linestyle='dotted')
    if transfer_entropy:
        axes[idx,0].plot(transfer_entropy_data['simulation_time'],np.array(transfer_entropy_data['flow'][valves[idx]])/cfs2cms,label='Transfer Entropy',color='y',linewidth=2,alpha=0.7,linestyle='dashed')
    if ccm:
        axes[idx,0].plot(ccm_data['simulation_time'],np.array(ccm_data['flow'][valves[idx]])/cfs2cms,label='CCM',color='m',linewidth=2,alpha=0.7,linestyle='dashdot')

    # add a dotted red line indicating the flow threshold
    axes[idx,0].hlines(3.9/cfs2cms, correct_data['simulation_time'][0],correct_data['simulation_time'][-1],label='Threshold',colors='r',linestyles='solid',linewidth=2,alpha=0.5)
    #axes[idx,0].set_ylabel( str(  str(valves[idx]) + " Flow" ),rotation='horizontal',labelpad=8)
    axes[idx,0].annotate(str( "V" +   str(valves[idx][1:]) + " Flow" ),xy=(0.5,0.8),xycoords='axes fraction',fontsize='xx-large')
    if idx == 2:
        axes[idx,0].legend(fontsize='medium',loc='upper right')
    if idx != 3:
        axes[idx,0].set_xticks([])
       

# plot the storage nodes
for idx in range(4):
    if uncontrolled:
        axes[idx,1].plot(uncontrolled_data['simulation_time'],np.array(uncontrolled_data['depthN'][storage_nodes[idx]])/ft2meters,label='Uncontrolled',color='k',linewidth=2,alpha=0.7)
    if correct:
        axes[idx,1].plot(correct_data['simulation_time'],np.array(correct_data['depthN'][storage_nodes[idx]])/ft2meters,label='Correct',color='b',linewidth=2,alpha=0.7)
    if granger:
        axes[idx,1].plot(granger_data['simulation_time'],np.array(granger_data['depthN'][storage_nodes[idx]])/ft2meters,label='Granger',color='g',linewidth=2,alpha=0.7,linestyle='dotted')
    if transfer_entropy:
        axes[idx,1].plot(transfer_entropy_data['simulation_time'],np.array(transfer_entropy_data['depthN'][storage_nodes[idx]])/ft2meters,label='Transfer Entropy',color='y',linewidth=2,alpha=0.7,linestyle='dashed')
    if ccm:
        axes[idx,1].plot(ccm_data['simulation_time'],np.array(ccm_data['depthN'][storage_nodes[idx]])/ft2meters,label='CCM',color='m',linewidth=2,alpha=0.7,linestyle='dashdot')

    #axes[idx,1].set_ylabel( str( str(storage_nodes[idx]) + " Depth"),rotation='horizontal',labelpad=8)
    axes[idx,1].annotate( str( str(storage_nodes[idx]) + " Depth"),xy=(0.5,0.8),xycoords='axes fraction',fontsize='xx-large')
    
    # add a dotted red line indicating the depth threshold
    axes[idx,1].hlines(flooding_depths[flooding_depth_idx[idx]]/ft2meters,correct_data['simulation_time'][0],correct_data['simulation_time'][-1],label='Threshold',colors='r',linestyles='solid',linewidth=2,alpha=0.5)
    if idx != 3:
        axes[idx,1].set_xticks([])
        
if cartoon:
    # get rid of all the tick labels
    for ax in axes.flatten():
        ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    # make all the lines twice as thick
    for ax in axes.flatten():
        for line in ax.lines:
            line.set_linewidth(5)
    

plt.tight_layout()
filename = "C:/blind_swmm_controller_gamma/comparisons/" + str(duration[1:]) + str(recurrence_intervals[1:-1])
if cartoon:
    filename += "_cartoon"
plt.savefig(str(filename + ".png"), dpi=450,transparent=True)
plt.savefig(str(filename + ".svg"), dpi=450,transparent=True)
plt.show()
        

