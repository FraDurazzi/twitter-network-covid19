from collections import Counter
from itertools import chain
import networkit as nk
import numpy as np
import pandas as pd
import pickle

#Disease Echoes and Sentiments on Twitter INformation Over-time

def best_communities(coms_saved,how_many_best):
    #Sorts communities in descending size (# of users)
    #Set communities of the first trial as reference for the others
    trials=len(coms_saved)
    best_coms=[]
    for i in range(trials):
        temp=sorted([(c,len(v)) for c,v in coms_saved[i].items()],key=lambda x:x[1],reverse=True)
        best_coms.append(tuple([c[0] for c in temp][:how_many_best]))
        
        #print(best_coms)
    print('Communities ordered with descending size')
    count=Counter(chain(best_coms))
    print(count.most_common(1))
    best_com=list(count.most_common(1)[0][0])
    com_trials=[]
    for i in range(trials):
        com_trials.append({j: coms_saved[i][j] for j in best_coms[i]})
    return best_com,com_trials

def superposition(com_trials,trial1,com1,trial2,com2):
    #NOT USED
    #Computes overlap between com1 and com2 respect to the total of com1 users
    try:
        return len(set(com_trials[trial1][com1]).intersection(set(com_trials[trial2][com2])))/len(com_trials[trial1][com1])
    except ZeroDivisionError:
        return 0
    
def superposition2(com_trials,trial1,com1,trial2,com2):
    #Computes overlap fraction between com1 and com2 respect to the total of com2 users

    try:
        return len(set(com_trials[trial2][com2]).intersection(set(com_trials[trial1][com1])))/len(com_trials[trial2][com2])
    except ZeroDivisionError:
        return 0


def superp_matrix(com_trials,how_many_best,best_trial=0):
    #Computes the overlap fraction between communities in different trials,
    # respect to a reference trial (default: 1st)
    #Returns: list of 'how_many_best' matrices,  trials x how_many_best
    # com_superp[i][j][k] is the overlap fraction of trial 0 community i 
    #with trial j community k
    trials=len(com_trials)
    print('Communities superposition on {} trials'.format(trials))
    com_superp=[]
    for com1 in range(how_many_best):
        print('Computing superpositions of Com {}...'.format(com1+1))
        com_superp.append(np.empty((trials,how_many_best),dtype=float))
        for trial2 in range(trials):
            for com2 in range(how_many_best):
                #pass
                
                
                if com1>=len(com_trials[best_trial].keys()) or com2>=len(com_trials[trial2].keys()):
                    com_superp[com1][trial2][com2]=0
                    
                else:
                    c1=list(com_trials[best_trial].keys())[com1]#il nome/chiave in quel trial
                    c2=list(com_trials[trial2].keys())[com2]
                    com_superp[com1][trial2][com2]=superposition2(com_trials,
                                                             best_trial,
                                                             c1,
                                                             trial2,
                                                             c2)
    return com_superp


def compute_translation(com_superp,best_com):
    com_trans=[]
    trials=len(com_superp[0])
    for t in range(trials):
        
        com_trans.append({i:(list(best_com)[np.random.choice(
            np.flatnonzero(com_superp[i][t]==com_superp[i][t].max()))]) for i in range(len(best_com))})
    for i in range(len(best_com)):
        s=0
        for trial in range(trials):
            #if np.argmax(com_superp[i][trial])==i:
            if np.random.choice(np.flatnonzero(com_superp[i][trial]==com_superp[i][trial].max()))==i:
                s=s+1
        print('Community {} remains in position: {} times on {}'.format(list(best_com)[i],
                                                                        s,trials))
    return com_trans

def load_communities(path,node_map=None,c_ids=False,verbose=True):
    #path: path of the user_id:community dict (.pickle)
    #node_map: NOT USED
    #c_ids: True if you want to return the community:set_of_user_ids dict
    
    with open(path,'rb') as f:
        com_of_user=pickle.load(f)
    how_many_best=len(set(com_of_user.values()))
    user_set=set(com_of_user.keys())
    com_ids={}
    if c_ids:
        for i,c in enumerate(sorted(list(set(com_of_user.values())))):
            com_ids[c]=set([u for u in user_set if com_of_user[u]==c])
            if verbose:
                print('Com {} length: {}'.format(c,len(com_ids[c])))
        #sets=[set(com_ids[i]) for i in range(how_many_best)]
    com_nodes=[]
    if node_map:
        for i in range(how_many_best):
            com_nodes.append([node_map[u] for u in com_ids[i]])
        return com_of_user, how_many_best,com_ids,sets,com_nodes
    elif c_ids:
        return com_of_user,how_many_best,com_ids
    else:
        return com_of_user,how_many_best

def print_locations(df,n=10):
    #how_many_best=len(sets)
    for c in sorted(list(set(df['community']))):
        if c:
            print('\nCommon locations in {}'.format(c))
            print(df[df['community']==c]['user.location'].value_counts()[:n])
        
def k2date(k):
    form='%H_%M_%d_%m %Y'
    return pd.to_datetime(k+' 2020',origin='unix',format=form)