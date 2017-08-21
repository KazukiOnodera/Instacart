#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 12:55:38 2017

@author: konodera

item order ratio divide by chance


ex1:
onb_buy = [5,8,9]
onb_visit = [1,2,5,8,9]
return: 3/3

ex2:
onb_buy = [5,9]
onb_visit = [1,2,5,8,9]
return: 2/3

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import utils
utils.start(__file__)


#==============================================================================
# load
#==============================================================================
col = ['order_id', 'user_id', 'product_id', 'order_number', 'order_number_rev']
log = utils.read_pickles('../input/mk/log', col).sort_values(['user_id', 'product_id', 'order_number'])


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

    cnt = log_.groupby(['user_id', 'product_id']).size()
    cnt.name = 'cnt'
    cnt = cnt.reset_index()
    
    # chance
    user_onb_max = log_.groupby('user_id').order_number.max().reset_index()
    user_onb_max.columns = ['user_id', 'onb_max']
    
    user_item_min = log_.groupby(['user_id', 'product_id']).order_number.min().reset_index()
    user_item_min.columns = ['user_id', 'product_id', 'onb_min']
    
    chance = pd.merge(user_item_min, user_onb_max, on='user_id', how='left')
    chance['chance'] = chance.onb_max - chance.onb_min +1
    
    df = pd.merge(cnt, chance, on=['user_id', 'product_id'], how='left')
    
    df['order_ratio_bychance'] = df.cnt / df.chance
    
    col = ['user_id', 'product_id', 'chance', 'order_ratio_bychance']
    df[col].to_pickle('../feature/{}/f309_user-product.p'.format(folder))
    
    # === near5 ===
    log_ = log[log.order_number_rev>T][log.order_number_rev<=(T+5)]

    cnt = log_.groupby(['user_id', 'product_id']).size()
    cnt.name = 'cnt'
    cnt = cnt.reset_index()
    
    # chance
    user_onb_max = log_.groupby('user_id').order_number.max().reset_index()
    user_onb_max.columns = ['user_id', 'onb_max']
    
    user_item_min = log_.groupby(['user_id', 'product_id']).order_number.min().reset_index()
    user_item_min.columns = ['user_id', 'product_id', 'onb_min']
    
    chance = pd.merge(user_item_min, user_onb_max, on='user_id', how='left')
    chance['chance_n5'] = chance.onb_max - chance.onb_min +1
    
    df = pd.merge(cnt, chance, on=['user_id', 'product_id'], how='left')
    
    df['order_ratio_bychance_n5'] = df.cnt / df.chance_n5
    
    col = ['user_id', 'product_id', 'chance_n5', 'order_ratio_bychance_n5']
    df[col].to_pickle('../feature/{}/f309_user-product_n5.p'.format(folder))
    
    
#==============================================================================
# main
#==============================================================================
make(0)
make(1)
make(2)

make(-1)



#==============================================================================
utils.end(__file__)

