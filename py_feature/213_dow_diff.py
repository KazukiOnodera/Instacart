#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 13:58:58 2017

@author: konodera
"""

import pandas as pd
import numpy as np
import utils
utils.start(__file__)

#==============================================================================
# load
#==============================================================================

usecols = ['product_id', 'order_dow', 'order_number_rev']
log = utils.read_pickles('../input/mk/log', usecols)

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
    
    all_item_dist = log_.order_dow.value_counts(normalize=True).reset_index()
    all_item_dist.columns = ['order_dow', 'dow_dist_ratio']
    
    tbl = log_.groupby(['product_id', 'order_dow']).size().reset_index()
    tbl.columns = ['product_id', 'order_dow', 'item_dow_cnt']
    tbl['item_dow_ratio'] = tbl.item_dow_cnt / tbl.groupby('product_id').transform(np.sum).item_dow_cnt
    
    tbl = pd.merge(tbl, all_item_dist, on='order_dow', how='left')
    
    tbl['item_dow_ratio_diff'] = tbl.item_dow_ratio - tbl.dow_dist_ratio
    
    tbl[['product_id','order_dow', 'item_dow_ratio_diff']].to_pickle('../feature/{}/f213_product-dow.p'.format(folder))

#==============================================================================
# main
#==============================================================================
make(0)
make(1)
make(2)

make(-1)

#==============================================================================
utils.end(__file__)















utils.end(__file__)

