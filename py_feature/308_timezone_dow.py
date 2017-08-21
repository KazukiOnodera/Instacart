#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 23:28:15 2017

@author: konodera

そのユーザーがその時間にそのアイテムを買う率

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
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
    
    # timezone
    cnt = log_.groupby(['user_id', 'product_id', 'timezone']).size()
    cnt.name = 'useritem_buy_timezone_cnt'
    cnt = cnt.reset_index()
    
    chance = log_.drop_duplicates('order_id').groupby(['user_id', 'timezone']).size()
    chance.name = 'total'
    chance = chance.reset_index()
    
    df = pd.merge(cnt, chance, on=['user_id', 'timezone'], how='left')
    df['useritem_buy_timezone_ratio2'] = df.useritem_buy_timezone_cnt / df.total
    
    col = ['user_id', 'product_id', 'timezone', 'useritem_buy_timezone_ratio2']
    
    df[col].to_pickle('../feature/{}/f308_user-product-timezone.p'.format(folder))
    
    # dow
    cnt = log_.groupby(['user_id', 'product_id', 'order_dow']).size()
    cnt.name = 'useritem_buy_dow_cnt'
    cnt = cnt.reset_index()
    
    chance = log_.drop_duplicates('order_id').groupby(['user_id', 'order_dow']).size()
    chance.name = 'total'
    chance = chance.reset_index()
    
    df = pd.merge(cnt, chance, on=['user_id', 'order_dow'], how='left')
    df['useritem_buy_dow_ratio2'] = df.useritem_buy_dow_cnt / df.total
    
    col = ['user_id', 'product_id', 'order_dow', 'useritem_buy_dow_ratio2']
    
    df[col].to_pickle('../feature/{}/f308_user-product-dow.p'.format(folder))

#==============================================================================
# main
#==============================================================================
make(0)
make(1)
make(2)

make(-1)





#==============================================================================
utils.end(__file__)

