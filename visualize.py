#from sqlite3 import Row
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy._lib.deprecation import deprecate_cython_api
import seaborn as sns

correct = False
granger = False
transfer_entropy = False
ccm = False
uncontrolled = False

# define tss_max as the maximum value of all the tss_loading data
tss_max = 0.0
flow_exceedance_max = 0.0
filling_degree_max = 0.0

try:
    tss_loading_uncontrolled = pd.read_csv("C:/blind_swmm_controller_gamma/outfall_TSS_loading_uncontrolled.csv",
                                       index_col=0,header=0)
    uncontrolled=True
    # change sediment loading from pounds to kilograms
    tss_loading_uncontrolled = tss_loading_uncontrolled / 2.2
    tss_max = max(tss_max, tss_loading_uncontrolled.max().max())
except:
    print("no uncontrolled data")
    
try:
    tss_loading_correct = pd.read_csv("C:/blind_swmm_controller_gamma/outfall_TSS_loading_correct.csv",
                                      index_col=0,header=0)
    correct=True
    tss_loading_correct = tss_loading_correct / 2.2
    tss_max = max(tss_max, tss_loading_correct.max().max())
except:
    print("no correct data")
    
try:
    tss_loading_granger = pd.read_csv("C:/blind_swmm_controller_gamma/outfall_TSS_loading_granger.csv",
                                      index_col=0,header=0)
    granger=True
    tss_loading_granger = tss_loading_granger / 2.2
    tss_max = max(tss_max, tss_loading_granger.max().max())
except:
    print("no granger data")
    
try:    
    tss_loading_transfer_entropy = pd.read_csv("C:/blind_swmm_controller_gamma/outfall_TSS_loading_transfer-entropy.csv",
                                               index_col=0,header=0)
    transfer_entropy=True
    tss_loading_transfer_entropy = tss_loading_transfer_entropy / 2.2
    tss_max = max(tss_max, tss_loading_transfer_entropy.max().max())
except:
    print("no transfer entropy data")

try:   
    tss_loading_ccm = pd.read_csv("C:/blind_swmm_controller_gamma/outfall_TSS_loading_ccm.csv",
                                  index_col=0,header=0)
    ccm=True
    tss_loading_ccm = tss_loading_ccm / 2.2
    tss_max = max(tss_max, tss_loading_ccm.max().max())
except:
    print("no ccm data")
    
if uncontrolled:
    flow_exceedance_uncontrolled = pd.read_csv("C:/blind_swmm_controller_gamma/flow_exceedance_uncontrolled.csv",
                                       index_col=0,header=0)
    flow_exceedance_max = max(flow_exceedance_max, flow_exceedance_uncontrolled.max().max())
if correct:
    flow_exceedance_correct = pd.read_csv("C:/blind_swmm_controller_gamma/flow_exceedance_correct.csv",
                                        index_col=0,header=0)
    flow_exceedance_max = max(flow_exceedance_max, flow_exceedance_correct.max().max())
if granger:
    flow_exceedance_granger = pd.read_csv("C:/blind_swmm_controller_gamma/flow_exceedance_granger.csv",
                                        index_col=0,header=0)
    flow_exceedance_max = max(flow_exceedance_max, flow_exceedance_granger.max().max())
if transfer_entropy:
    flow_exceedance_transfer_entropy = pd.read_csv("C:/blind_swmm_controller_gamma/flow_exceedance_transfer-entropy.csv",
                                        index_col=0,header=0)
    flow_exceedance_max = max(flow_exceedance_max, flow_exceedance_transfer_entropy.max().max())
if ccm:
    flow_exceedance_ccm = pd.read_csv("C:/blind_swmm_controller_gamma/flow_exceedance_ccm.csv",
                                        index_col=0,header=0)
    flow_exceedance_max = max(flow_exceedance_max, flow_exceedance_ccm.max().max())
    
if uncontrolled:
    filling_degree_uncontrolled = pd.read_csv("C:/blind_swmm_controller_gamma/filling_degree_uncontrolled.csv",
                                       index_col=0,header=0)
    filling_degree_max = max(filling_degree_max, filling_degree_uncontrolled.max().max())
if correct:
    filling_degree_correct = pd.read_csv("C:/blind_swmm_controller_gamma/filling_degree_correct.csv",
                                        index_col=0,header=0)
    filling_degree_max = max(filling_degree_max, filling_degree_correct.max().max())
if granger:
    filling_degree_granger = pd.read_csv("C:/blind_swmm_controller_gamma/filling_degree_granger.csv",
                                        index_col=0,header=0)
    filling_degree_max = max(filling_degree_max, filling_degree_granger.max().max())
if transfer_entropy:
    filling_degree_transfer_entropy = pd.read_csv("C:/blind_swmm_controller_gamma/filling_degree_transfer-entropy.csv",
                                        index_col=0,header=0)
    filling_degree_max = max(filling_degree_max, filling_degree_transfer_entropy.max().max())
if ccm:
    filling_degree_ccm = pd.read_csv("C:/blind_swmm_controller_gamma/filling_degree_ccm.csv",
                                        index_col=0,header=0)
    filling_degree_max = max(filling_degree_max, filling_degree_ccm.max().max())

# print summary statistics for each of the five dataframes
if uncontrolled:
    print("tss_loading_uncontrolled")
    print(tss_loading_uncontrolled.describe())
if correct:
    print("tss_loading_correct")
    print(tss_loading_correct.describe())
if granger:
    print("tss_loading_granger")
    print(tss_loading_granger.describe())
if transfer_entropy:
    print("tss_loading_transfer_entropy")
    print(tss_loading_transfer_entropy.describe())
if ccm:
    print("tss_loading_ccm")
    print(tss_loading_ccm.describe())

if uncontrolled:
    print("flow_exceedance_uncontrolled")
    print(flow_exceedance_uncontrolled.describe())
if correct:
    print("flow_exceedance_correct")
    print(flow_exceedance_correct.describe())
if granger:
    print("flow_exceedance_granger")
    print(flow_exceedance_granger.describe())
if transfer_entropy:
    print("flow_exceedance_transfer_entropy")
    print(flow_exceedance_transfer_entropy.describe())
if ccm:
    print("flow_exceedance_ccm")
    print(flow_exceedance_ccm.describe())
    
if uncontrolled:
    print("filling_degree_uncontrolled")
    print(filling_degree_uncontrolled.describe())
if correct:
    print("filling_degree_correct")
    print(filling_degree_correct.describe())
if granger:
    print("filling_degree_granger")
    print(filling_degree_granger.describe())
if transfer_entropy:
    print("filling_degree_transfer_entropy")
    print(filling_degree_transfer_entropy.describe())
if ccm:
    print("filling_degree_ccm")
    print(filling_degree_ccm.describe())

# display all those as heatmaps which have the same legend
if uncontrolled:
    sns.heatmap(tss_loading_uncontrolled, vmin=0, vmax=tss_max)
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    plt.title("TSS Loading Uncontrolled")
    plt.savefig("C:/blind_swmm_controller_gamma/tss_loading_uncontrolled.png")
    plt.close()
    
    sns.heatmap(flow_exceedance_uncontrolled, vmin=0, vmax=flow_exceedance_max)
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    plt.title("Flow Exceedance Uncontrolled")
    plt.savefig("C:/blind_swmm_controller_gamma/flow_exceedance_uncontrolled.png")
    plt.close()
    
    sns.heatmap(filling_degree_uncontrolled, vmin=0, vmax=filling_degree_max)
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    plt.title("Filling Degree Uncontrolled")
    plt.savefig("C:/blind_swmm_controller_gamma/filling_degree_uncontrolled.png")
    plt.close()
    
if correct:
    sns.heatmap(tss_loading_correct, vmin=0, vmax=tss_max)
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    plt.title("TSS Loading Correct")
    plt.savefig("C:/blind_swmm_controller_gamma/tss_loading_correct.png")
    plt.close()
    
    sns.heatmap(flow_exceedance_correct, vmin=0, vmax=flow_exceedance_max)
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    plt.title("Flow Exceedance Correct")
    plt.savefig("C:/blind_swmm_controller_gamma/flow_exceedance_correct.png")
    plt.close()
    
    sns.heatmap(filling_degree_correct, vmin=0, vmax=filling_degree_max)
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    plt.title("Filling Degree Correct")
    plt.savefig("C:/blind_swmm_controller_gamma/filling_degree_correct.png")
    plt.close()
    
if granger:
    sns.heatmap(tss_loading_granger, vmin=0, vmax=tss_max)
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    plt.title("TSS Loading Granger")
    plt.savefig("C:/blind_swmm_controller_gamma/tss_loading_granger.png")
    plt.close()
    
    sns.heatmap(flow_exceedance_granger, vmin=0, vmax=flow_exceedance_max)
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    plt.title("Flow Exceedance Granger")
    plt.savefig("C:/blind_swmm_controller_gamma/flow_exceedance_granger.png")
    plt.close()
    
    sns.heatmap(filling_degree_granger, vmin=0, vmax=filling_degree_max)
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    plt.title("Filling Degree Granger")
    plt.savefig("C:/blind_swmm_controller_gamma/filling_degree_granger.png")
    plt.close()
    
if transfer_entropy:
    sns.heatmap(tss_loading_transfer_entropy, vmin=0, vmax=tss_max)
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    plt.title("TSS Loading Transfer Entropy")
    plt.savefig("C:/blind_swmm_controller_gamma/tss_loading_transfer_entropy.png")
    plt.close()
    
    sns.heatmap(flow_exceedance_transfer_entropy, vmin=0, vmax=flow_exceedance_max)
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    plt.title("Flow Exceedance Transfer Entropy")
    plt.savefig("C:/blind_swmm_controller_gamma/flow_exceedance_transfer_entropy.png")
    plt.close()
    
    sns.heatmap(filling_degree_transfer_entropy, vmin=0, vmax=filling_degree_max)
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    plt.title("Filling Degree Transfer Entropy")
    plt.savefig("C:/blind_swmm_controller_gamma/filling_degree_transfer_entropy.png")
    plt.close()
    
if ccm:
    sns.heatmap(tss_loading_ccm, vmin=0, vmax=tss_max)
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    plt.title("TSS Loading CCM")
    plt.savefig("C:/blind_swmm_controller_gamma/tss_loading_ccm.png")
    plt.close()
    
    sns.heatmap(flow_exceedance_ccm, vmin=0, vmax=flow_exceedance_max)
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    plt.title("Flow Exceedance CCM")
    plt.savefig("C:/blind_swmm_controller_gamma/flow_exceedance_ccm.png")
    plt.close()
    
    sns.heatmap(filling_degree_ccm, vmin=0, vmax=filling_degree_max)
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    plt.title("Filling Degree CCM")
    plt.savefig("C:/blind_swmm_controller_gamma/filling_degree_ccm.png")
    plt.close()
    

# make mega-figure (leave out flow exceedance for now)
fig,axes = plt.subplots(2,6)
# make the figure 3 times as wide as it is tall
fig.set_size_inches(18,6)
axes[0,0].set_title("Uncontrolled",fontsize='x-large')
axes[0,1].set_title("Correct",fontsize='x-large')
axes[0,2].set_title("Granger",fontsize='x-large')
axes[0,3].set_title("Transfer Entropy",fontsize='x-large')
axes[0,4].set_title("CCM",fontsize='x-large')

# filling degree
sns.heatmap(filling_degree_uncontrolled,ax=axes[0,0],cbar=False,vmin=0,vmax=filling_degree_max,yticklabels='auto',xticklabels=False)
sns.heatmap(filling_degree_correct,ax=axes[0,1],cbar=False,vmin=0,vmax=filling_degree_max,yticklabels=False,xticklabels=False)
sns.heatmap(filling_degree_granger,ax=axes[0,2],cbar=False,vmin=0,vmax=filling_degree_max,yticklabels=False,xticklabels=False)
sns.heatmap(filling_degree_transfer_entropy,ax=axes[0,3],cbar=False,vmin=0,vmax=filling_degree_max,yticklabels=False,xticklabels=False)
sns.heatmap(filling_degree_ccm,ax=axes[0,4],cbar=True,cbar_ax = axes[0,5],vmin=0,vmax=filling_degree_max,yticklabels=False,xticklabels=False)

axes[0,0].contour(filling_degree_uncontrolled, levels=[0.5,0.99], colors=['white','black'])
axes[0,1].contour(filling_degree_correct, levels=[0.5,0.99], colors=['white','black'])
axes[0,2].contour(filling_degree_granger, levels=[0.5,0.99], colors=['white','black'])
axes[0,3].contour(filling_degree_transfer_entropy, levels=[0.5,0.99], colors=['white','black'])
axes[0,4].contour(filling_degree_ccm, levels=[0.5,0.99], colors=['white','black'])
axes[0,5].axhline(0.5, color='white')
axes[0,5].axhline(0.99, color='black')

# tss loading in the second row, make the contour lines at 100 and 250
sns.heatmap(tss_loading_uncontrolled,ax = axes[1,0],cbar=False,vmin=0,vmax=tss_max,yticklabels='auto',xticklabels='auto')
sns.heatmap(tss_loading_correct,ax = axes[1,1],cbar=False,vmin=0,vmax=tss_max,yticklabels=False,xticklabels='auto')
sns.heatmap(tss_loading_granger,ax = axes[1,2],cbar=False,vmin=0,vmax=tss_max,yticklabels=False,xticklabels='auto')
sns.heatmap(tss_loading_transfer_entropy,ax = axes[1,3],cbar=False,vmin=0,vmax=tss_max,yticklabels=False,xticklabels='auto')
sns.heatmap(tss_loading_ccm,ax = axes[1,4],cbar=True,cbar_ax = axes[1,5],vmin=0,vmax=tss_max,yticklabels=False,xticklabels='auto')

axes[1,0].contour(tss_loading_uncontrolled, levels=[50,250], colors=['white','black'])
axes[1,1].contour(tss_loading_correct, levels=[50,250], colors=['white','black'])
axes[1,2].contour(tss_loading_granger, levels=[50,250], colors=['white','black'])
axes[1,3].contour(tss_loading_transfer_entropy, levels=[50,250], colors=['white','black'])
axes[1,4].contour(tss_loading_ccm, levels=[50,250], colors=['white','black'])
axes[1,5].axhline(50, color='white')
axes[1,5].axhline(250, color='black')

axes[0,0].set_ylabel("Max\nFilling\nDegree",fontsize='x-large',rotation='horizontal',labelpad=20.0)
axes[1,0].set_ylabel("TSS\nLoading\n(kg)",fontsize='x-large',rotation='horizontal',labelpad=20.0)


plt.tight_layout()
plt.savefig("C:/blind_swmm_controller_gamma/total_results.png",dpi=600,transparent=True)
plt.savefig("C:/blind_swmm_controller_gamma/total_results.svg",dpi=600,transparent=True)
plt.show()
plt.close()