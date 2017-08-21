#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 00:00:43 2017

@author: konodera


oder_num - last_order_num
"""


import pandas as pd
import numpy as np
from tqdm import tqdm
import utils
utils.start(__file__)


col = ['order_id', 'user_id', 'product_id', 'order_number', 'order_number_rev']
log = utils.read_pickles('../input/mk/log', col).sort_values(['user_id', 'order_number'])

orders = pd.read_csv('../input/orders.csv.gz', usecols=['order_id', 'order_number'])

X_base = pd.read_pickle('../feature/X_base_t3.p')
X_base = pd.merge(X_base, orders, on='order_id', how='left')


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
        
    label = pd.read_pickle('../feature/{}/label_reordered.p'.format(folder))
    label = pd.merge(label, X_base, on='order_id', how='left')
    
    log_ = log[log.order_number_rev>T]
    log_.drop_duplicates(['user_id', 'product_id'], keep='last', inplace=True)
    log_.drop(['order_id','order_number_rev'], axis=1, inplace=1)
    log_.columns = ['user_id', 'product_id', 'last_order_number']
    
    df = pd.merge(label, log_, on=['user_id', 'product_id'], how='left')
    df['order_number_diff'] = df.order_number - df.last_order_number

    col = ['order_id', 'product_id', 'last_order_number', 'order_number_diff']
    df[col].to_pickle('../feature/{}/f305_order-product.p'.format(folder))

#==============================================================================
# main
#==============================================================================
make(0)
make(1)
make(2)

make(-1)




utils.end(__file__)

