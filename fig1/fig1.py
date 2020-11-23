import networkx as nx
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle
import time
import random
import pandas as pd
import os
import datetime as dt
import glob
from tqdm import tqdm

alphab=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q',
        'R','S','T','U','V','W','X','Y','Z','AA','AB','AC','AD','AE','AF','AG']
comtodraw=alphab[:15]
com2type={'A':'Other','B':'International expert','C':'Political',
         'D':'National elite','E':'Other','F':'Political',
          'G':'International expert','H':'Political','I':'National elite',
          'J':'National elite','K':'National elite',
          'L':'Political','M':'National elite',
          'N':'Other', 'O':'Other','XX':'Other'
         }    

palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
type2col = {'International expert': palette[4], 'National elite': palette[1], 'Political': palette[2], 'Other': '.5'} 

com2col={'A':'black','B':'red','C':'orchid','D':'deepskyblue','E':'orange',
        'F':'darkseagreen','G':'deeppink','H':'blue','I':'yellow','J':'brown','K':'cyan',
        'L':'lime','M':'green','N':'slateblue'}
for com in comtodraw:
    if com not in list(com2col.keys()):
         com2col[com]='grey'
com2col['XX']='grey'

#Load dataframe with positions computed with Gephi
df_nodes=pd.read_parquet('../data/fig1/df_gephi_positions.parquet')

colori=[com2col[com] for com in df_nodes.community]
s_t=time.time()
f=plt.figure(dpi=600,figsize=(12,12))
plt.scatter(df_nodes['x'],df_nodes['y'],c=colori,s=df_nodes['size'],alpha=1,
            marker='.',edgecolors=None,linewidths=0)
markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for com,color in com2col.items() if com!='XX']
scritte=[l for l in alphab[:15]]
plt.legend(markers, scritte, numpoints=1,loc=(.95,.3),#'upper right',
          fontsize=16)
plt.axis('off')

plt.savefig('fig1.png', bbox_inches='tight')
# Saving to .pdf would be too heavy

e_t=time.time()-s_t
print('Elapsed time: {} min'.format(e_t/60))