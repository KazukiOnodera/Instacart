#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 22:36:10 2017

@author: konodera


nohup python -u 212_withinN.py &


"""

import pandas as pd
import gc
import numpy as np
from collections import defaultdict
import multiprocessing as mp
total_proc = 3
import utils
utils.start(__file__)

#==============================================================================
# load
#==============================================================================

usecols = ['product_id', 'user_id', 'order_number', 'order_id', 'order_number_rev']
log = utils.read_pickles('../input/mk/log', usecols).sort_values(usecols[:3])

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
    
    log_ = log[log.order_number_rev>T]
    log_['user_max_onb'] = log_.groupby('user_id').order_number.transform(np.max)
    
    item_N2_cnt    = defaultdict(int)
    item_N2_chance = defaultdict(int)
    item_N3_cnt    = defaultdict(int)
    item_N3_chance = defaultdict(int)
    item_N4_cnt    = defaultdict(int)
    item_N4_chance = defaultdict(int)
    item_N5_cnt    = defaultdict(int)
    item_N5_chance = defaultdict(int)
    pid_bk = uid_bk = onb_bk = None
#    for pid, uid, onb, max_onb in tqdm(log_[['product_id', 'user_id', 'order_number','user_max_onb']].values):
    for pid, uid, onb, max_onb in log_[['product_id', 'user_id', 'order_number','user_max_onb']].values:
        
        if pid==pid_bk and uid==uid_bk and (onb-onb_bk)<=2 and (max_onb-onb) >=2:
            item_N2_cnt[pid] +=1
        if pid==pid_bk and uid==uid_bk and (max_onb-onb) >=2:
            item_N2_chance[pid] +=1
        
        if pid==pid_bk and uid==uid_bk and (onb-onb_bk)<=3 and (max_onb-onb) >=3:
            item_N3_cnt[pid] +=1
        if pid==pid_bk and uid==uid_bk and (max_onb-onb) >=3:
            item_N3_chance[pid] +=1
        
        if pid==pid_bk and uid==uid_bk and (onb-onb_bk)<=4 and (max_onb-onb) >=4:
            item_N4_cnt[pid] +=1
        if pid==pid_bk and uid==uid_bk and (max_onb-onb) >=4:
            item_N4_chance[pid] +=1
        
        if pid==pid_bk and uid==uid_bk and (onb-onb_bk)<=5 and (max_onb-onb) >=5:
            item_N5_cnt[pid] +=1
        if pid==pid_bk and uid==uid_bk and (max_onb-onb) >=5:
            item_N5_chance[pid] +=1
        
        pid_bk = pid
        uid_bk = uid
        onb_bk = onb
    
    item_N2_cnt = pd.DataFrame.from_dict(item_N2_cnt, orient='index').reset_index()
    item_N2_cnt.columns = ['product_id', 'item_N2_cnt']
    item_N2_chance = pd.DataFrame.from_dict(item_N2_chance, orient='index').reset_index()
    item_N2_chance.columns = ['product_id', 'item_N2_chance']
    
    item_N3_cnt = pd.DataFrame.from_dict(item_N3_cnt, orient='index').reset_index()
    item_N3_cnt.columns = ['product_id', 'item_N3_cnt']
    item_N3_chance = pd.DataFrame.from_dict(item_N3_chance, orient='index').reset_index()
    item_N3_chance.columns = ['product_id', 'item_N3_chance']
    
    item_N4_cnt = pd.DataFrame.from_dict(item_N4_cnt, orient='index').reset_index()
    item_N4_cnt.columns = ['product_id', 'item_N4_cnt']
    item_N4_chance = pd.DataFrame.from_dict(item_N4_chance, orient='index').reset_index()
    item_N4_chance.columns = ['product_id', 'item_N4_chance']
    
    item_N5_cnt = pd.DataFrame.from_dict(item_N5_cnt, orient='index').reset_index()
    item_N5_cnt.columns = ['product_id', 'item_N5_cnt']
    item_N5_chance = pd.DataFrame.from_dict(item_N5_chance, orient='index').reset_index()
    item_N5_chance.columns = ['product_id', 'item_N5_chance']
    
    df2 = pd.merge(item_N2_cnt, item_N2_chance, on='product_id', how='outer')
    df3 = pd.merge(item_N3_cnt, item_N3_chance, on='product_id', how='outer')
    df4 = pd.merge(item_N4_cnt, item_N4_chance, on='product_id', how='outer')
    df5 = pd.merge(item_N5_cnt, item_N5_chance, on='product_id', how='outer')
    
    df = pd.merge(pd.merge(df2, df3, on='product_id', how='outer'),
                  pd.merge(df4, df5, on='product_id', how='outer'), 
                  on='product_id', how='outer').fillna(0)
    
    df['item_N2_ratio'] = df['item_N2_cnt']/df['item_N2_chance']
    df['item_N3_ratio'] = df['item_N3_cnt']/df['item_N3_chance']
    df['item_N4_ratio'] = df['item_N4_cnt']/df['item_N4_chance']
    df['item_N5_ratio'] = df['item_N5_cnt']/df['item_N5_chance']
    
    df.fillna(0, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_pickle('../feature/{}/f212_product.p'.format(folder))
    
#==============================================================================
# main
#==============================================================================

mp_pool = mp.Pool(total_proc)
mp_pool.map(make, range(-1,3))

#==============================================================================
utils.end(__file__)

