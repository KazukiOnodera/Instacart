#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 07:11:23 2017

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

col = ['order_id', 'product_id', 'add_to_cart_order', 'order_number_rev']
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
    
    gr = log_.groupby('product_id')
    
    items = gr.add_to_cart_order.mean().to_frame()
    items.columns = ['item_mean_pos_cart']
    items['item_sum_pos_cart'] = gr.add_to_cart_order.sum()
    items['item_min_pos_cart'] = gr.add_to_cart_order.min()
    items['item_median_pos_cart'] = gr.add_to_cart_order.median()
    items['item_max_pos_cart'] = gr.add_to_cart_order.max()
    items['item_std_pos_cart'] = gr.add_to_cart_order.std()
    items.reset_index(inplace=True)
    
    items.to_pickle('../feature/{}/f207_product.p'.format(folder))

#==============================================================================
# main
#==============================================================================
make(0)
make(1)
make(2)

make(-1)

#==============================================================================
utils.end(__file__)

