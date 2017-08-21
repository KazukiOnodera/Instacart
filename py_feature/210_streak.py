#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 22:36:10 2017

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

col = ['order_id', 'user_id', 'product_id', 'order_number_rev']
log = utils.read_pickles('../input/mk/log', col).sort_values('user_id')

streak = pd.read_pickle('../input/mk/streak_order-product.p')
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
        
    log_ = pd.merge(log[log.order_number_rev>T], streak,
                    on=['order_id', 'product_id'], how='left')
    
    gr = log_.groupby('product_id')
    item = gr.streak.mean().to_frame()
    item.columns = ['item_streak_mean']
    
    item['item_streak_min'] = gr.streak.min()
    item['item_streak_max'] = gr.streak.max()
    item['item_streak_std'] = gr.streak.std()
    
    item.reset_index().to_pickle('../feature/{}/f210_product.p'.format(folder))

#==============================================================================
# main
#==============================================================================
make(0)
make(1)
make(2)

make(-1)














utils.end(__file__)

