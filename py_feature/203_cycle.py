#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 06:46:06 2017

@author: konodera

Item buy cycle

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
#import gc
import utils
utils.start(__file__)

usecols = [ 'order_id', 'user_id', 'product_id', 'order_number', 'reordered', 'order_number_rev']
log = pd.merge(utils.read_pickles('../input/mk/log', usecols),
               utils.read_pickles('../input/mk/days_since_last_order'),
               on=['order_id','product_id'], how='left')

def make(log, folder):

    tbl = log.groupby('product_id').days_since_last_order_this_item.mean().to_frame()
    tbl.columns = ['item_order_days_mean']
    tbl['item_order_days_min'] = log.groupby('product_id').days_since_last_order_this_item.min()
    tbl['item_order_days_max'] = log.groupby('product_id').days_since_last_order_this_item.max()
    tbl['item_order_days_median'] = log.groupby('product_id').days_since_last_order_this_item.median()
    
    tbl['item_order_freq'] = log.groupby('product_id').size()
    
    tbl['item_reorderd_freq'] = log.groupby('product_id').reordered.sum()
    tbl['item_reorder_ratio'] = (tbl.item_reorderd_freq / tbl.item_order_freq).astype(np.float32)
    
    tbl['item_unique_user'] = log.drop_duplicates(['user_id', 'product_id']).groupby('product_id').size()
    tbl['item_order_per-user'] = tbl['item_order_freq'] / tbl['item_unique_user']
    
    tbl.reset_index(inplace=1)
    
    
    tbl.to_pickle('../feature/{}/f203_product.p'.format(folder))
#==============================================================================
# main
#==============================================================================
make(log[log.order_number_rev>0], 'trainT-0')
make(log[log.order_number_rev>1], 'trainT-1')
make(log[log.order_number_rev>2], 'trainT-2')

make(log, 'test')



#==============================================================================
utils.end(__file__)

