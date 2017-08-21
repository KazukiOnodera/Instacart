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
    
    order_size = log_.groupby('order_id').size().reset_index()
    order_size.columns = ['order_id', 'total']
    
    log_ = pd.merge(log_, order_size, on='order_id', how='left')
    
    item = log_.groupby('product_id').total.mean().to_frame()
    item.columns = ['item_together_mean']
    
    item['item_together_min'] = log_.groupby('product_id').total.min()
    item['item_together_max'] = log_.groupby('product_id').total.max()
    item['item_together_std'] = log_.groupby('product_id').total.std()
    
    item.reset_index().to_pickle('../feature/{}/f209_product.p'.format(folder))

#==============================================================================
# main
#==============================================================================
make(0)
make(1)
make(2)

make(-1)














utils.end(__file__)

