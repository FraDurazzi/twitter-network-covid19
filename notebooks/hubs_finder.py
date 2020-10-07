#Use this script to identify the most retweeted users for each community
# and save them to a .csv
#Needs raw data

import networkx as nx
import networkit as nk
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle
import multiprocessing as mp
import time
import random
from collections import Counter
import pandas as pd
import matplotlib.dates as mdates
from scipy import stats
from functools import partial
import os
import psutil
import datetime as dt
import glob
import joblib
from tqdm import tqdm
import sys
sys.path.insert(1,'../helpers')
import destino



def read_data(f_name):
    """Reads single parquet file"""
    return pd.read_parquet(f_name,columns=col_names)

def load_threads(f_names):
    """Load data with threads"""
    ts = time.time()
    parallel = joblib.Parallel(n_jobs=30, prefer='threads')
    read_data_delayed = joblib.delayed(read_data)
    res = parallel(read_data_delayed(f_name) for f_name in tqdm(f_names))
    df = pd.concat(res)
    te = time.time()
    print(f'Load threads took {te-ts:.5f} sec')
    return df

s_tot=time.time()
col_names=['created_at','id','user.id','user.screen_name',
           'user.num_followers','lang']
directory='data/parq/tweets'
lst=os.listdir(directory)
lst.sort()
print('Loading data...')
s_t=time.time()
lst_path=[directory+'/'+fi for fi in lst]
df=load_threads(lst_path)
e_t=time.time()-s_t
print('Elapsed time: {} min'.format(e_t/60))
df=df[df['lang']=='en']
df.drop(['lang'],axis=1,inplace=True)
print(df.shape)

directory='data/parq/merged_predictions'
col_names=['user.id', 'type_label','category_label','userbio_lang']
lst=os.listdir(directory)
lst.sort()
print('Loading predictions...')
s_t=time.time()
lst_path=[directory+'/'+fi for fi in lst]
df_cat=load_threads(lst_path)
e_t=time.time()-s_t
print('Elapsed time: {} min'.format(e_t/60))
print(df_cat.shape)
df_cat=df_cat[df_cat['userbio_lang']=='en']
df_cat.drop(['userbio_lang'],axis=1,inplace=True)
print(df_cat.shape)

s_t=time.time()
u2type=pd.Series(df_cat['type_label'].values,index=df_cat['user.id']).to_dict()
u2cat=pd.Series(df_cat['category_label'].values,index=df_cat['user.id']).to_dict()
df['type_label']=df['user.id'].map(u2type)
df['category_label']=df['user.id'].map(u2cat)
print(df.shape)
e_t=time.time()-s_t
print('Elapsed time: {} min'.format(e_t/60))

print('\nLoading communities...')
s_t=time.time()
alphab=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q',
        'R','S','T','U','V','W','X','Y','Z','AA','AB','AC','AD','AE','AF','AG']
com_of_user,how_many_best=destino.load_communities('data/com_of_user_letters_1_30.pickle',c_ids=False)
print('Mapping communities...')
df['community']=df['user.id'].map(com_of_user)
e_t=time.time()-s_t
print('Elapsed time: {} min'.format(e_t/60))

print('\nLoading network...')
s_t=time.time()
Gx=nx.read_weighted_edgelist('data/edgelist_may.txt',delimiter='\t',create_using=nx.DiGraph,nodetype=str)
e_t=time.time()-s_t
print('Elapsed time: {} min'.format(e_t/60))

show=30
best_df={}
s_t=time.time()
for i,c in enumerate(alphab[:how_many_best]):
    #if len(com_ids[c])<100000 or i>=2:
    #    continue
    df_part=df[df.community==c]
    print('Analyzing community ',c)
    Gxs=Gx.subgraph(set(df_part['user.id']))
    DXs=Gxs.out_degree(weight='weight')
    DXs=sorted(DXs,key=lambda x:x[1],reverse=True)
    max_retweeted=DXs[:show]
    best_list=[]
    for u in max_retweeted:
        urow=df_part[df_part['user.id']==u[0]][['user.screen_name','user.num_followers','type_label','category_label']].iloc[-1]
        #print(urow[0])
        best_list.append([urow[0],u[1],urow[1],urow[2],urow[3]])
    best_df[c]=pd.DataFrame(best_list)
e_t=time.time()-s_t
print('Elapsed time: {} min'.format(e_t/60))

print('\nWriting sheets...')
s_t=time.time()
writer = pd.ExcelWriter('csvs/hubs_1_30_cat.xlsx', engine='xlsxwriter')
for c in tqdm(list(best_df.keys())):
    best_df[c].to_excel(writer, sheet_name='Commmunity '+c)
writer.save()
e_t=time.time()-s_t
print('Elapsed time: {} min'.format(e_t/60))

e_tot=time.time()-s_tot
print('\nElapsed time: {} hours'.format(e_tot/60/60))

    