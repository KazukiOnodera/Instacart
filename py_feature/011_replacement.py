#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 09:26:45 2017

@author: konodera

nohup python -u 011_replacement.py &


t-3 -> t-2 -> t-1
 a      a      a
 b      d      c
 c      e      d
               f

pids_3notin2: b,c
pids_2notin3: d,e
pids_1notin2: c,f
pids_skip:    c

c -> e -> c

ratio: freq(c -> d -> c)/freq(c -> d)

merge t-1: c->d

"""

import pandas as pd
import gc
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from itertools import product
import utils
utils.start(__file__)

#==============================================================================
# load
#==============================================================================

usecols = ['user_id', 'order_number', 'product_id', 'product_name', 'order_id', 'order_number_rev']
log = utils.read_pickles('../input/mk/log', usecols).sort_values(usecols[:3])

order_tbl = log[['order_id', 'user_id', 'order_number', 'order_number_rev']].drop_duplicates().reset_index(drop=True)
for i in range(1, 4):
    order_tbl['t-{}_order_id'.format(i)] = order_tbl.groupby('user_id')['order_id'].shift(i)
order_tbl.dropna(inplace=True)

#order_pids = log.head(999999).groupby('order_id').product_id.apply(set).reset_index()
order_pids = log.groupby('order_id').product_id.apply(set).reset_index()

order_tbl = pd.merge(order_tbl, 
                     order_pids.add_prefix('t-1_'), 
                     on='t-1_order_id', how='inner')
order_tbl = pd.merge(order_tbl, 
                     order_pids.add_prefix('t-2_'), 
                     on='t-2_order_id', how='inner')
order_tbl = pd.merge(order_tbl, 
                     order_pids.add_prefix('t-3_'), 
                     on='t-3_order_id', how='inner')

#==============================================================================
# def
#==============================================================================

def make(T):
    """
    T = 0
    folder = 'trainT-0'
    """
    if T==-1:
        folder = 'test'
    else:
        folder = 'trainT-'+str(T)
    
    order_tbl_ = order_tbl[order_tbl.order_number_rev>T]
    
    pid_cnt    = defaultdict(int)
    pid_chance = defaultdict(int)
    
#    for pids_bk3, pids_bk2, pids_bk1 in tqdm(order_tbl_[['t-3_product_id', 't-2_product_id', 't-1_product_id']].values):
#    for uid, onb, pid in tqdm(log_[['user_id', 'order_number', 'product_name']].head(1999999).values):
#    for uid, onb, pid in tqdm(log_[['user_id', 'order_number', 'product_id']].values, miniters=99999):
    for pids_bk3, pids_bk2, pids_bk1 in tqdm(order_tbl_[['t-3_product_id', 't-2_product_id', 't-1_product_id']].values, miniters=99999):
        
        pids_3notin2 = pids_bk3 - pids_bk2
        pids_2notin3 = pids_bk2 - pids_bk3
        pids_hub  = pids_bk2 - pids_bk3 - pids_bk1
        pids_skip = (pids_bk3 & pids_bk1) - pids_bk2
        
        li = []
        for i1, i2 in list(product(pids_3notin2, pids_2notin3)):
            key = str(i1)+' -> '+str(i2)
            li.append(key)
            pid_chance[key] +=1
            
        li = []
        for i1, i2 in list(product(pids_skip, pids_hub)):
            key = str(i1)+' -> '+str(i2)
            li.append(key)
            pid_cnt[key] +=1
        
        
    
    pid_chance = pd.DataFrame.from_dict(pid_chance, orient='index').reset_index()
    pid_chance.columns = ['pids', 'chance']
    
    pid_cnt = pd.DataFrame.from_dict(pid_cnt, orient='index').reset_index()
    pid_cnt.columns = ['pids', 'back']
    
    df = pd.merge(pid_chance, pid_cnt, on='pids', how='left').fillna(0)
    
    df['ratio'] = df.back/df.chance
    df.sort_values('ratio', ascending=False, inplace=True)
    
    df.reset_index(drop=True, inplace=True)
    df['pid1'] = df.pids.map(lambda x: x.split(' -> ')[0]).astype(int)
    df['pid2'] = df.pids.map(lambda x: x.split(' -> ')[1]).astype(int)
    df[['pid1', 'pid2', 'back', 'chance', 'ratio']].to_pickle('../input/mk/replacement.p')
    
#==============================================================================
# main
#==============================================================================

make(2)

#==============================================================================
utils.end(__file__)

