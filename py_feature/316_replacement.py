#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 22:36:10 2017

@author: konodera


nohup python -u 316_replacement.py &


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
order_pids = log.groupby('order_id').product_id.apply(set).reset_index()

#item = pd.read_pickle('../input/mk/replacement2.p').head(999)
item = pd.read_pickle('../input/mk/replacement.p')
item = item[item.back>9]

# parse
item_di = defaultdict(int)
for pid1,pid2,ratio in item[['pid1', 'pid2', 'ratio']].values:
    item_di['{} {}'.format(int(pid1),int(pid2))] = ratio
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
    
    X_base = pd.read_pickle('../feature/X_base_t3.p')
    label = pd.read_pickle('../feature/{}/label_reordered.p'.format(folder))
    
    # 'inner' for removing t-n_order_id == NaN
    if 'train' in folder:
        df = pd.merge(X_base[X_base.is_train==1], label, on='order_id', how='inner')
    elif folder == 'test':
        df = pd.merge(X_base[X_base.is_train==0], label, on='order_id', how='inner')
    
    df = pd.merge(df, 
                  order_pids.add_prefix('t-1_'), 
                  on='t-1_order_id', how='left')
    df = pd.merge(df, 
                  order_pids.add_prefix('t-2_'), 
                  on='t-2_order_id', how='left')
    
    ratio_min  = []
    ratio_mean = []
    ratio_max  = []
    ratio_sum  = []
    ratio_len  = []
    for t_2,t_1,pid in tqdm(df[['t-2_product_id', 't-1_product_id', 'product_id']].values, miniters=99999):
        rep = t_1 - t_2
        if pid not in t_1 and pid in t_2 and len(rep)>0:
            ratios = [item_di['{} {}'.format(i1,i2)] for i1,i2 in  list(product([pid], rep))]
            ratio_min.append(np.min(ratios))
            ratio_mean.append(np.mean(ratios))
            ratio_max.append(np.max(ratios))
            ratio_sum.append(np.sum(ratios))
            ratio_len.append(len(ratios))
        else:
            ratio_min.append(-1)
            ratio_mean.append(-1)
            ratio_max.append(-1)
            ratio_sum.append(-1)
            ratio_len.append(-1)
    
    df['comeback_ratio_min']  = ratio_min
    df['comeback_ratio_mean'] = ratio_mean
    df['comeback_ratio_max']  = ratio_max
    df['comeback_ratio_sum']  = ratio_sum
    df['comeback_ratio_len']  = ratio_len
    
    col = ['order_id', 'product_id', 'comeback_ratio_min', 'comeback_ratio_mean',
           'comeback_ratio_max', 'comeback_ratio_sum', 'comeback_ratio_len']
    df[col].to_pickle('../feature/{}/f316_order_product.p'.format(folder))
    del df
    gc.collect()
    
#==============================================================================
# main
#==============================================================================
make(0)
make(1)
make(2)

make(-1)

#==============================================================================
utils.end(__file__)

