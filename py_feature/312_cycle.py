#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 10:35:09 2017

@author: konodera

userのそのitemのcycle

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import utils
utils.start(__file__)


#==============================================================================
# load
#==============================================================================
usecols = [ 'order_id', 'user_id', 'product_id', 'order_number', 'reordered', 'order_number_rev']
log = pd.merge(utils.read_pickles('../input/mk/log', usecols),
               utils.read_pickles('../input/mk/days_since_last_order'),
               on=['order_id','product_id'], how='left')


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
    
    key = ['user_id', 'product_id']
    tbl = log_.groupby(key).days_since_last_order_this_item.mean().to_frame()
    tbl.columns = ['useritem_order_days_mean']
    tbl['useritem_order_days_min'] = log_.groupby(key).days_since_last_order_this_item.min()
    tbl['useritem_order_days_max'] = log_.groupby(key).days_since_last_order_this_item.max()
    tbl['useritem_order_days_median'] = log_.groupby(key).days_since_last_order_this_item.median()
    
    tbl.reset_index().to_pickle('../feature/{}/f312_user_product.p'.format(folder))

    # === near5 ===
    log_ = log[log.order_number_rev>T][log.order_number_rev<=(T+5)]
    
    key = ['user_id', 'product_id']
    tbl = log_.groupby(key).days_since_last_order_this_item.mean().to_frame()
    tbl.columns = ['useritem_order_days_mean_n5']
    tbl['useritem_order_days_min_n5'] = log_.groupby(key).days_since_last_order_this_item.min()
    tbl['useritem_order_days_max_n5'] = log_.groupby(key).days_since_last_order_this_item.max()
    tbl['useritem_order_days_median_n5'] = log_.groupby(key).days_since_last_order_this_item.median()
    
    tbl.reset_index().to_pickle('../feature/{}/f312_user_product_n5.p'.format(folder))

#==============================================================================
# main
#==============================================================================
make(0)
make(1)
make(2)

make(-1)



#==============================================================================
utils.end(__file__)

