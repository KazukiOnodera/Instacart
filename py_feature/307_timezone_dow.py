#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 15:50:03 2017

@author: konodera

そのユーザーがそのアイテムを買う時間帯の割合

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import utils
utils.start(__file__)

#==============================================================================
# load
#==============================================================================

col = ['order_id', 'user_id', 'product_id', 'order_dow', 'order_hour_of_day', 'order_number_rev']
log = utils.read_pickles('../input/mk/log', col).sort_values('user_id')
log = pd.merge(log, pd.read_pickle('../input/mk/timezone.p'), 
               on='order_hour_of_day', how='left')

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

    cnt = log_.groupby(['user_id', 'product_id', 'timezone']).size()
    cnt.name = 'useritem_buy_timezone_cnt'
    cnt = cnt.reset_index()
    
    sum_ = log_.groupby(['user_id', 'product_id']).size()
    sum_.name = 'total'
    sum_ = sum_.reset_index()
    
    df = pd.merge(cnt, sum_, on=['user_id', 'product_id'], how='left')
    
    df['useritem_buy_timezone_ratio'] = df.useritem_buy_timezone_cnt / df.total
    
    col = ['user_id', 'product_id', 'timezone', 
           'useritem_buy_timezone_cnt', 'useritem_buy_timezone_ratio']
    
    df[col].to_pickle('../feature/{}/f307_user-product-timezone.p'.format(folder))
    
    #==============================================================================
    
    
    cnt = log_.groupby(['user_id', 'product_id', 'order_dow']).size()
    cnt.name = 'useritem_buy_dow_cnt'
    cnt = cnt.reset_index()
    
    sum_ = log_.groupby(['user_id', 'product_id']).size()
    sum_.name = 'total'
    sum_ = sum_.reset_index()
    
    df = pd.merge(cnt, sum_, on=['user_id', 'product_id'], how='left')
    
    df['useritem_buy_dow_ratio'] = df.useritem_buy_dow_cnt / df.total
    
    col = ['user_id', 'product_id', 'order_dow', 
           'useritem_buy_dow_cnt', 'useritem_buy_dow_ratio']
    
    df[col].to_pickle('../feature/{}/f307_user-product-dow.p'.format(folder))

#==============================================================================
# main
#==============================================================================
make(0)
make(1)
make(2)

make(-1)






#==============================================================================
utils.end(__file__)

