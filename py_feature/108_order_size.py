#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 21:04:09 2017

@author: konodera


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
log = utils.read_pickles('../input/mk/log', col).sort_values('user_id')

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
    
    order_tbl = pd.merge(order_tbl, log_[['order_id', 'user_id']].drop_duplicates())
    
    user_osz = order_tbl.groupby(['user_id']).order_size.min().to_frame()
    user_osz.columns = ['user_order_size-min']
    user_osz['user_order_size-max'] = order_tbl.groupby(['user_id']).order_size.max()
    user_osz['user_order_size-median'] = order_tbl.groupby(['user_id']).order_size.median()
    user_osz['user_order_size-mean'] = order_tbl.groupby(['user_id']).order_size.mean()
    user_osz['user_order_size-std'] = order_tbl.groupby(['user_id']).order_size.std()
    user_osz.reset_index(inplace=True)
    
    user_osz.to_pickle('../feature/{}/f108_user.p'.format(folder))

#==============================================================================
# main
#==============================================================================
make(0)
make(1)
make(2)

make(-1)
















utils.end(__file__)

