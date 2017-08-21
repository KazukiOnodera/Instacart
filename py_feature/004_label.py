#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 17:07:09 2017

@author: konodera
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import utils
utils.start(__file__)

col = ['order_id', 'user_id', 'product_id', 'order_number', 'reordered', 'order_number_rev']
log = utils.read_pickles('../input/mk/log', col).rename(columns={'reordered':'y'})

test_order = pd.read_pickle('../input/mk/test_user.p')

#==============================================================================
# train
#==============================================================================
def make(T):
    label_t1 = log[log.order_number_rev>T]
    label_t1.drop_duplicates(['user_id','product_id'], keep='last', inplace=True)
    label_t1.sort_values(['user_id','product_id'], inplace=True)
    
    label_t0_y1 = log.loc[log.order_number_rev==T].loc[log.y==1]
    label_t0_y1.sort_values(['user_id','product_id'], inplace=True)
    
    label_t1['key'] = label_t1.user_id.map(str) + ' ' + label_t1.product_id.map(str)
    label_t0_y1['key'] = label_t0_y1.user_id.map(str) + ' ' + label_t0_y1.product_id.map(str)
    label_t0_y0 = label_t1[~label_t1.key.isin(label_t0_y1.key)]
    
    label_t0_y0.drop('order_id', axis=1 ,inplace=True)
    label_t0_y0 = pd.merge(label_t0_y0, log.loc[log.order_number_rev==T, ['user_id','order_id']].drop_duplicates(), 
                           on='user_id', how='left')
    label_t0_y0.y = 0
    
    label_train = pd.concat([label_t0_y1, label_t0_y0], ignore_index=1)
    label_train.sort_values(['user_id','product_id'], inplace=True)
    label_train.reset_index(drop=1, inplace=True)
    
    col = ['order_id', 'product_id', 'y']
    
    print(label_train[col].isnull().sum())
    label_train[col].to_pickle('../feature/trainT-{}/label_reordered.p'.format(T))

make(0) # basically train is T=0, for validation, train;T=1 valid;T=0
make(1)
make(2)

#==============================================================================
# test
#==============================================================================
log_test = log.drop_duplicates(['user_id','product_id'])[['user_id','product_id']]
log_test = log_test[log_test.user_id.isin(test_order.user_id)]

log_test.sort_values(['user_id','product_id'],inplace=True)
log_test.reset_index(drop=1, inplace=True)

test_order = pd.merge(test_order, log_test, on='user_id', how='left')

print(test_order[['order_id', 'product_id']].isnull().sum())
test_order[['order_id', 'product_id']].to_pickle('../feature/test/label_reordered.p')




#==============================================================================
utils.end(__file__)

