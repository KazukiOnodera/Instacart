#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 03:58:09 2017

@author: konodera

一回しか買わないユーザーの数

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
    
    item = log_.groupby(['product_id', 'user_id']).size().reset_index()
    item.columns = ['product_id', 'user_id', 'cnt']
    
    item_one = item[item.cnt==1].groupby('product_id').size().reset_index()
    item_one.columns = ['product_id', 'item_only_one_user_cnt']
    
    item_size = item.groupby('product_id').size().reset_index()
    item_size.columns = ['product_id', 'item_unique_user']
    
    item = pd.merge(item_one, item_size, on='product_id', how='left')
    item['item_only_one_user_cnt_ratio'] = item['item_only_one_user_cnt']/item['item_unique_user']
    
    col = ['product_id', 'item_only_one_user_cnt', 'item_only_one_user_cnt_ratio']
    item[col].to_pickle('../feature/{}/f208_product.p'.format(folder))

#==============================================================================
# main
#==============================================================================
make(0)
make(1)
make(2)

make(-1)













utils.end(__file__)

