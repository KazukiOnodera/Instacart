#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 15:52:10 2017

@author: konodera

if t-1 == first buy, what's the ratio of reorderd?

"""

import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import utils
utils.start(__file__)

#==============================================================================
# load
#==============================================================================

usecols = ['user_id', 'product_id', 'order_number', 'reordered', 'order_number_rev']
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
    log_ = log_.groupby(['user_id', 'product_id']).head(2)
    
    item_cnt    = defaultdict(int)
    item_chance = defaultdict(int)
    pid_bk = uid_bk = onb_bk = None
    
    for uid, pid, onb, max_onb in log_[['user_id', 'product_id', 'order_number', 'user_max_onb']].values:
        
        if uid==uid_bk and pid==pid_bk and (onb-onb_bk==1):
            item_cnt[pid] +=1
        if onb!=max_onb:
            item_chance[pid] +=1
    
        pid_bk = pid
        uid_bk = uid
        onb_bk = onb
    
    item_cnt = pd.DataFrame.from_dict(item_cnt, orient='index').reset_index()
    item_cnt.columns = ['product_id', 'item_first_cnt']
    item_chance = pd.DataFrame.from_dict(item_chance, orient='index').reset_index()
    item_chance.columns = ['product_id', 'item_first_chance']
    
    df = pd.merge(item_cnt, item_chance, on='product_id', how='outer').fillna(0)
    df['item_first_ratio'] = df.item_first_cnt/df.item_first_chance
    
    df.to_pickle('../feature/{}/f214_product.p'.format(folder))


#==============================================================================
# main
#==============================================================================

make(0)
make(1)
make(2)

make(-1)

#==============================================================================

utils.end(__file__)

