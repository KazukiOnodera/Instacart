#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 16:02:01 2017

@author: konodera
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import utils
utils.start(__file__)


col = ['order_id', 'user_id', 'product_id', 'order_number', 'order_number_rev']
log = utils.read_pickles('../input/mk/log', col)

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
    
    order_tbl = log_.groupby('order_id').size().to_frame()
    order_tbl.columns = ['order_size']
    order_tbl.reset_index(inplace=True)
    
    order_tbl = pd.merge(order_tbl, log_[['order_id', 'user_id', 'product_id']])
    
    col = ['user_id', 'product_id']
    tbl = log_.sort_values(col).drop_duplicates(col)[col]
    tbl = tbl.set_index(col)
    
    gr = order_tbl.groupby(['user_id', 'product_id'])
    
    tbl['useritem_cooccur-min'] = gr.order_size.min()
    tbl['useritem_cooccur-max'] = gr.order_size.max()
    tbl['useritem_cooccur-mean'] = gr.order_size.mean()
    tbl['useritem_cooccur-median'] = gr.order_size.median()
    tbl['useritem_cooccur-std'] = gr.order_size.std()
    tbl.reset_index(inplace=True)
    
    user_osz = order_tbl.groupby(['user_id']).order_size.min().to_frame()
    user_osz.columns = ['user_order_size-min']
    user_osz['user_order_size-max'] = order_tbl.groupby(['user_id']).order_size.max()
    user_osz.reset_index(inplace=True)
    
    tbl = pd.merge(tbl, user_osz, on='user_id', how='left')
    
    tbl['useritem_cooccur-min-min'] = tbl['user_order_size-min']  - tbl['useritem_cooccur-min']
    tbl['useritem_cooccur-max-min'] = tbl['useritem_cooccur-max'] - tbl['useritem_cooccur-min']
    tbl['useritem_cooccur-max-max'] = tbl['user_order_size-max'] - tbl['useritem_cooccur-max']
    tbl.drop(['user_order_size-min', 'user_order_size-max'], axis=1, inplace=True)
    
    tbl.to_pickle('../feature/{}/f314_user-product.p'.format(folder))

#==============================================================================
# main
#==============================================================================
make(0)
make(1)
make(2)

make(-1)


utils.end(__file__)

