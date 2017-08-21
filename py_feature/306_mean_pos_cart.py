#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 06:42:38 2017

@author: konodera

平均pos_cart

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import utils
utils.start(__file__)


col = ['order_id', 'user_id', 'product_id', 'add_to_cart_order', 'order_number_rev']
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
        
    gr = log_.groupby(['user_id', 'product_id'])
    
    user = gr.add_to_cart_order.mean().to_frame()
    user.columns = ['useritem_mean_pos_cart']
    user['useritem_sum_pos_cart'] = gr.add_to_cart_order.sum()
    user['useritem_min_pos_cart'] = gr.add_to_cart_order.min()
    user['useritem_median_pos_cart'] = gr.add_to_cart_order.median()
    user['useritem_max_pos_cart'] = gr.add_to_cart_order.max()
    user['useritem_std_pos_cart'] = gr.add_to_cart_order.std()
    user.reset_index(inplace=True)
    
    user.to_pickle('../feature/{}/f306_user-product.p'.format(folder))
    
    # === near5 ===
    log_ = log[log.order_number_rev>T][log.order_number_rev<=(T+5)]
        
    gr = log_.groupby(['user_id', 'product_id'])
    
    user = gr.add_to_cart_order.mean().to_frame()
    user.columns = ['useritem_mean_pos_cart_n5']
    user['useritem_sum_pos_cart_n5'] = gr.add_to_cart_order.sum()
    user['useritem_min_pos_cart_n5'] = gr.add_to_cart_order.min()
    user['useritem_median_pos_cart_n5'] = gr.add_to_cart_order.median()
    user['useritem_max_pos_cart_n5'] = gr.add_to_cart_order.max()
    user['useritem_std_pos_cart_n5'] = gr.add_to_cart_order.std()
    user.reset_index(inplace=True)
    
    user.to_pickle('../feature/{}/f306_user-product_n5.p'.format(folder))
    
    
#==============================================================================
# main
#==============================================================================
make(0)
make(1)
make(2)

make(-1)




#==============================================================================
utils.end(__file__)

