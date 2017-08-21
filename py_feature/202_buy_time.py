#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 19:58:46 2017

@author: konodera

アイテムが買われる時間帯

"""

import pandas as pd
import numpy as np
import gc
from tqdm import tqdm
import utils
utils.start(__file__)

col = ['order_id', 'user_id', 'product_id', 'order_dow', 'order_hour_of_day', 'order_number_rev']
log = utils.read_pickles('../input/mk/log', col)
log = pd.merge(log, pd.read_pickle('../input/mk/timezone.p'), on='order_hour_of_day', how='left')
log['dow_tz'] = log.order_dow.map(str) + '_' + log.timezone


# TODO: rolling mean
def make(log, folder):
    #==============================================================================
    # hour
    #==============================================================================
    gc.collect()
    tbl = log.groupby(['product_id', 'order_hour_of_day']).size().reset_index()
    tbl.columns = ['product_id', 'order_hour_of_day', 'item_hour_cnt']
    
    tbl['item_hour_ratio'] = tbl.item_hour_cnt / tbl.groupby('product_id').transform(np.sum).item_hour_cnt
    
    tbl.to_pickle('../feature/{}/f202_product_hour.p'.format(folder))
    
    # unique
    tbl = log.drop_duplicates(['user_id', 'product_id', 'order_hour_of_day']).groupby(['product_id', 'order_hour_of_day']).size().reset_index()
    tbl.columns = ['product_id', 'order_hour_of_day', 'item_hour_cnt_unq']
    
    tbl['item_hour_ratio_unq'] = tbl.item_hour_cnt_unq / tbl.groupby('product_id').transform(np.sum).item_hour_cnt_unq
    
    tbl.to_pickle('../feature/{}/f202_uniq_product_hour.p'.format(folder))
    
    
    #==============================================================================
    # dow
    #==============================================================================
    gc.collect()
    tbl = log.groupby(['product_id', 'order_dow']).size().reset_index()
    tbl.columns = ['product_id', 'order_dow', 'item_dow_cnt']
    
    tbl['item_dow_ratio'] = tbl.item_dow_cnt / tbl.groupby('product_id').transform(np.sum).item_dow_cnt
    
    tbl.to_pickle('../feature/{}/f202_product_dow.p'.format(folder))
    
    # unique
    tbl = log.drop_duplicates(['user_id', 'product_id', 'order_dow']).groupby(['product_id', 'order_dow']).size().reset_index()
    tbl.columns = ['product_id', 'order_dow', 'item_dow_cnt_unq']
    
    tbl['item_dow_ratio_unq'] = tbl.item_dow_cnt_unq / tbl.groupby('product_id').transform(np.sum).item_dow_cnt_unq
    
    tbl.to_pickle('../feature/{}/f202_uniq_product_dow.p'.format(folder))
    
    
    #==============================================================================
    # timezone
    #==============================================================================
    gc.collect()
    tbl = log.groupby(['product_id', 'timezone']).size().reset_index()
    tbl.columns = ['product_id', 'timezone', 'item_timezone_cnt']
    
    tbl['item_timezone_ratio'] = (tbl.item_timezone_cnt / tbl.groupby('product_id').transform(np.sum).item_timezone_cnt).map(float)
    
    tbl.to_pickle('../feature/{}/f202_product_timezone.p'.format(folder))
    
    # unique
    tbl = log.drop_duplicates(['user_id', 'product_id', 'timezone']).groupby(['product_id', 'timezone']).size().reset_index()
    tbl.columns = ['product_id', 'timezone', 'item_timezone_cnt_uniq']
    
    tbl['item_timezone_ratio_uniq'] = (tbl.item_timezone_cnt_uniq / tbl.groupby('product_id').transform(np.sum).item_timezone_cnt_uniq).map(float)
    
    tbl.to_pickle('../feature/{}/f202_uniq_product_timezone.p'.format(folder))
    
    #==============================================================================
    # timezone * dow
    #==============================================================================
    gc.collect()
    
    tbl = log.groupby(['product_id', 'order_dow', 'timezone']).size().reset_index()
    tbl.columns = ['product_id', 'order_dow', 'timezone', 'item_dow-tz_cnt']
    
    tbl['item_dow-tz_ratio'] = (tbl['item_dow-tz_cnt'] / tbl.groupby('product_id').transform(np.sum)['item_dow-tz_cnt']).map(float)
    
    tbl.to_pickle('../feature/{}/f202_product_dow-timezone.p'.format(folder))
    
    # unique
    tbl = log.drop_duplicates(['user_id', 'product_id', 'order_dow', 'timezone']).groupby(['product_id', 'order_dow', 'timezone']).size().reset_index()
    tbl.columns = ['product_id', 'order_dow', 'timezone', 'item_dow-tz_cnt_uniq']
    
    tbl['item_dow-tz_ratio_uniq'] = (tbl['item_dow-tz_cnt_uniq'] / tbl.groupby('product_id').transform(np.sum)['item_dow-tz_cnt_uniq']).map(float)
    
    tbl.to_pickle('../feature/{}/f202_uniq_product_dow-timezone.p'.format(folder))
    
    
    #==============================================================================
    # flat
    #==============================================================================
    gc.collect()
    tbl = pd.crosstab(log.product_id, log.dow_tz, normalize='index').add_prefix('item_flat_dow-tz_')
    
    tbl.reset_index().to_pickle('../feature/{}/f202_flat_product.p'.format(folder))
#==============================================================================
# main
#==============================================================================
make(log[log.order_number_rev>0], 'trainT-0')
make(log[log.order_number_rev>1], 'trainT-1')
make(log[log.order_number_rev>2], 'trainT-2')

make(log, 'test')












#==============================================================================
utils.end(__file__)

