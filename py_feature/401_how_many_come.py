#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 01:09:41 2017

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
col = ['order_id', 'user_id', 'product_id', 'order_number', 'order_dow', 'order_hour_of_day', 'order_number_rev']
log = utils.read_pickles('../input/mk/log', col).sort_values(['user_id', 'product_id', 'order_number'])


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
    
    # dow
    dow = log_.drop_duplicates('order_id').groupby('order_dow').size()
    dow.name = 'dow_order_cnt'
    dow = dow.to_frame()
    
    dow['dow_item_cnt'] = log_.groupby('order_dow').size()
    
    dow /= dow.sum()
    
    dow['dow_rank_diff'] = dow.dow_order_cnt.rank() - dow.dow_item_cnt.rank()
    
    dow.reset_index().to_pickle('../feature/{}/f401_dow.p'.format(folder))
    
    
    # hour
    hour = log_.drop_duplicates('order_id').groupby('order_hour_of_day').size()
    hour.name = 'hour_order_cnt'
    hour = hour.to_frame()
    
    hour['hour_item_cnt'] = log_.groupby('order_hour_of_day').size()
    
    hour /= hour.sum()
    
    hour['hour_rank_diff'] = hour.hour_order_cnt.rank() - hour.hour_item_cnt.rank()
    
    hour.reset_index().to_pickle('../feature/{}/f401_hour.p'.format(folder))

#==============================================================================
# main
#==============================================================================
make(0)
make(1)
make(2)

make(-1)







#==============================================================================
utils.end(__file__)

