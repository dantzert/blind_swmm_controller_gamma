import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# load the inferred topologies
correct = False
granger = False
transfer_entropy = False
ccm = False

try:
    correct_data = pd.read_csv("C:/blind_swmm_controller_gamma/topo_correct.csv",index_col=0)
    correct = True
    # cast the columns and indices of the dataframes to strings
    correct_data.columns = correct_data.columns.astype(str)
    correct_data.index = correct_data.index.astype(str)
    #print("correct influence diagram")
    #print(correct_data)
except:
    print("no correct topology found")

try:
    granger_data = pd.read_csv("C:/blind_swmm_controller_gamma/topo_granger.csv",index_col=0)
    granger = True
    granger_data.columns = granger_data.columns.astype(str)
    granger_data.index = granger_data.index.astype(str)
except:
    print("no granger topology found")

try:
    transfer_entropy_data = pd.read_csv("C:/blind_swmm_controller_gamma/topo_transfer-entropy.csv",index_col=0)
    transfer_entropy = True
    transfer_entropy_data.columns = transfer_entropy_data.columns.astype(str)
    transfer_entropy_data.index = transfer_entropy_data.index.astype(str)
except:
    print("no transfer entropy topology found")

try:
    ccm_data = pd.read_csv("C:/blind_swmm_controller_gamma/topo_ccm.csv",index_col=0)
    ccm = True
    ccm_data.columns = ccm_data.columns.astype(str)
    ccm_data.index = ccm_data.index.astype(str)
except:
    print("no ccm topology found")


if correct:
    correct_graph = correct_data.copy(deep=True)
    correct_graph.values[:,:] = 0 # no connection
    correct_graph = correct_graph.astype('int64')
    # wherever there is an "i" in either graph, make it a 1
    correct_graph[correct_data == 'i'] = 1
    # wherever there's a "d" in either graph, make it a 2
    correct_graph[correct_data == 'd'] = 2
    # transposing the graphs reverses them and generates an influecne diagram
    correct_graph = correct_graph.transpose()
    # for every entry in the index which isn't in the columns, add it as a new column filled with zeros
    for idx in correct_graph.index:
        if idx not in correct_graph.columns:
            correct_graph[idx] = 0
    # cast the columns and indices of the dataframes to strings
    correct_graph.columns = correct_graph.columns.astype(str)
    correct_graph.index = correct_graph.index.astype(str)
    # omit self-loops when plotting (just cleaner this way)
    for idx in correct_graph.index:
        correct_graph.loc[idx,idx] = 0
    #print(correct_graph)

if granger:
    granger_graph = granger_data.copy(deep=True)
    granger_graph.values[:,:] = 0 # no connection
    granger_graph = granger_graph.astype('int64')
    granger_graph[granger_data == 'i'] = 1
    granger_graph[granger_data == 'd'] = 2
    granger_graph = granger_graph.transpose()
    for idx in granger_graph.index:
        if idx not in granger_graph.columns:
            granger_graph[idx] = 0
    granger_graph.columns = granger_graph.columns.astype(str)
    granger_graph.index = granger_graph.index.astype(str)
    for idx in granger_graph.index:
        granger_graph.loc[idx,idx] = 0

if transfer_entropy:
    transfer_entropy_graph = transfer_entropy_data.copy(deep=True)
    transfer_entropy_graph.values[:,:] = 0 # no connection
    transfer_entropy_graph = transfer_entropy_graph.astype('int64')
    transfer_entropy_graph[transfer_entropy_data == 'i'] = 1
    transfer_entropy_graph[transfer_entropy_data == 'd'] = 2
    transfer_entropy_graph = transfer_entropy_graph.transpose()
    for idx in transfer_entropy_graph.index:
        if idx not in transfer_entropy_graph.columns:
            transfer_entropy_graph[idx] = 0
    transfer_entropy_graph.columns = transfer_entropy_graph.columns.astype(str)
    transfer_entropy_graph.index = transfer_entropy_graph.index.astype(str)
    for idx in transfer_entropy_graph.index:
        transfer_entropy_graph.loc[idx,idx] = 0

if ccm:
    ccm_graph = ccm_data.copy(deep=True)
    ccm_graph.values[:,:] = 0 # no connection
    ccm_graph = ccm_graph.astype('int64')
    ccm_graph[ccm_data == 'i'] = 1
    ccm_graph[ccm_data == 'd'] = 2
    ccm_graph = ccm_graph.transpose()
    for idx in ccm_graph.index:
        if idx not in ccm_graph.columns:
            ccm_graph[idx] = 0
    ccm_graph.columns = ccm_graph.columns.astype(str)
    ccm_graph.index = ccm_graph.index.astype(str)
    for idx in ccm_graph.index:
        ccm_graph.loc[idx,idx] = 0

    
# use networkx to draw the networks identified by the different methods
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].set_title("Correct",fontsize='xx-large')
axes[1].set_title("Granger",fontsize='xx-large')
axes[2].set_title("Transfer Entropy",fontsize='xx-large')
axes[3].set_title("CCM",fontsize='xx-large')
# draw the correct topology
G = nx.from_pandas_adjacency(correct_graph, create_using=nx.DiGraph)
pos = nx.spectral_layout(G) # this position will be used for all the other graphs as well
nx.draw_networkx_nodes(G, pos, node_size=600, ax=axes[0])
nx.draw_networkx_labels(G, pos, font_size=12, ax=axes[0])
nx.draw_networkx_edges(G, pos, arrows=True, ax=axes[0],edge_cmap='viridis', edge_vmin = 1,edge_vmax = 2,arrowsize=30,style='dashed',alpha=0.4,connectionstyle="arc3,rad=0.3")
# draw granger
if granger:
    G = nx.from_pandas_adjacency(granger_graph, create_using=nx.DiGraph)
    nx.draw_networkx_nodes(G, pos, node_size=600,  ax=axes[1])
    nx.draw_networkx_labels(G, pos, font_size=12, ax=axes[1])
    nx.draw_networkx_edges(G, pos, arrows=True, ax=axes[1],edge_cmap='viridis', edge_vmin = 1,edge_vmax = 2,arrowsize=30,style='dashed',alpha=0.4,connectionstyle="arc3,rad=0.3")
# draw transfer entropy
if transfer_entropy:
    G = nx.from_pandas_adjacency(transfer_entropy_graph, create_using=nx.DiGraph)
    nx.draw_networkx_nodes(G, pos, node_size=600,  ax=axes[2])
    nx.draw_networkx_labels(G, pos, font_size=12, ax=axes[2])
    nx.draw_networkx_edges(G, pos, arrows=True, ax=axes[2],edge_cmap='viridis', edge_vmin = 1,edge_vmax = 2,arrowsize=30,style='dashed',alpha=0.4,connectionstyle="arc3,rad=0.3")
# draw ccm
if ccm:
    G = nx.from_pandas_adjacency(ccm_graph, create_using=nx.DiGraph)
    nx.draw_networkx_nodes(G, pos, node_size=600,  ax=axes[3])
    nx.draw_networkx_labels(G, pos, font_size=12, ax=axes[3])
    nx.draw_networkx_edges(G, pos, arrows=True, ax=axes[3],edge_cmap='viridis', edge_vmin = 1,edge_vmax = 2,arrowsize=30,style='dashed',alpha=0.4,connectionstyle="arc3,rad=0.3")

# get rid of all the boudning boxes on all the axes
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')



plt.tight_layout()
plt.savefig(str("C:/blind_swmm_controller_gamma/all_influence_diagrams.png"))
plt.savefig(str("C:/blind_swmm_controller_gamma/all_influence_diagrams.svg"))
plt.show()
#plt.close()

# make consensus_data a dataframe which includes a 3 wherever all three of granger, transfer entropy, and ccm have something other than "n"
# and a "2" wherever two of them have something other than "n"
# and a "1" wherever only one of them has something other than "n"
# don't include "correct" in this because it's not an inference method
consensus_data = correct_data.copy(deep=True)
consensus_data.values[:,:] = 0
consensus_data = consensus_data.astype('int64')
if granger:
    consensus_data[granger_data != 'n'] += 1
if transfer_entropy:
    consensus_data[transfer_entropy_data != 'n'] += 1
if ccm:
    consensus_data[ccm_data != 'n'] += 1

print(consensus_data)



