#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 22:45:03 2017

@author: konodera

リークしてるのでshiftして使うこと！
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
import utils
utils.start(__file__)

# setting T
T = 3


#==============================================================================
# load base
#==============================================================================
X_base = pd.read_pickle('../feature/X_base_t{}.p'.format(T))
all_order = pd.concat([X_base[c] for c in X_base.columns if 't-' in c]).unique()

order_tbl = pd.read_pickle('../input/mk/order_tbl.p')

col = ['order_id', 'order_number', 'order_dow', 'order_hour_of_day', 
       'days_since_prior_order', 'days_since_first_order']
X = pd.merge(X_base, order_tbl[col], on='order_id', how='left')

col_feature = []

#==============================================================================
# repeat_previous_ratio
#==============================================================================
order_tbl['t-2_product_name'] = order_tbl.groupby('user_id')['product_name'].shift(2)
order_tbl['t-3_product_name'] = order_tbl.groupby('user_id')['product_name'].shift(3)
order_tbl['t-4_product_name'] = order_tbl.groupby('user_id')['product_name'].shift(4)
order_tbl['t-5_product_name'] = order_tbl.groupby('user_id')['product_name'].shift(5)

order_tbl = order_tbl[order_tbl.order_id.isin(all_order)]

# fill list
col = ['product_name'] + [c for c in order_tbl.columns if 't-' in c]

def fill_list(s):
    if isinstance(s, float):
        return []
    return s

for c in col:
    order_tbl[c] = order_tbl[c].map(fill_list)

def ratio(list1, list2):
    """
    list1: previous
    list2: current
    
    return: intersection(previous & current) / current
    """
    if len(list1)==0 or len(list2)==0:
        return 
    ret = sum([1 for i in list2 if i in list1]) / len(list2)
    
    return ret

# w means window size
order_tbl['repeat_previous_ratio-w1'] = order_tbl.apply(\
         lambda x: ratio(x['t-1_product_name'], x['product_name']), axis=1)

order_tbl['repeat_previous_ratio-w2'] = order_tbl.apply(\
         lambda x: ratio(x['t-1_product_name']+x['t-2_product_name'], 
                         x['product_name']), axis=1)

order_tbl['repeat_previous_ratio-w3'] = order_tbl.apply(\
         lambda x: ratio(x['t-1_product_name']+x['t-2_product_name']+x['t-3_product_name'], 
                         x['product_name']), axis=1)

order_tbl['repeat_previous_ratio-w4'] = order_tbl.apply(\
         lambda x: ratio(x['t-1_product_name']+x['t-2_product_name']+x['t-3_product_name']+\
                         x['t-4_product_name'], x['product_name']), axis=1)

order_tbl['repeat_previous_ratio-w5'] = order_tbl.apply(\
         lambda x: ratio(x['t-1_product_name']+x['t-2_product_name']+x['t-3_product_name']+\
                         x['t-4_product_name']+x['t-5_product_name'], x['product_name']), axis=1)

col_feature += ['repeat_previous_ratio-w1','repeat_previous_ratio-w2',
                'repeat_previous_ratio-w3','repeat_previous_ratio-w4',
                'repeat_previous_ratio-w5']

#==============================================================================
# reordered_ratio
#==============================================================================
log = utils.read_pickles('../input/mk/log')
reordered_ratio = log.groupby(['order_id']).reordered.mean().reset_index()
reordered_ratio.columns = ['order_id', 'reordered_ratio']
order_tbl = pd.merge(order_tbl, reordered_ratio, on='order_id', how='left')

log['unreordered'] = 1-log.reordered
unreordered_ratio = log.groupby(['order_id']).unreordered.mean().reset_index()
unreordered_ratio.columns = ['order_id', 'unreordered_ratio']

order_tbl = pd.merge(order_tbl, unreordered_ratio, on='order_id', how='left')


del reordered_ratio, unreordered_ratio; gc.collect()

col_feature += ['reordered_ratio']

#==============================================================================
# total_unique_item
#==============================================================================

order_unique_item = log.groupby('order_id').unreordered.sum().reset_index()
order_unique_item.columns = ['order_id', 'unreordered_sum']

order_tbl = pd.merge(order_tbl, order_unique_item, on='order_id', how='left')

order_tbl['total_unique_item'] = order_tbl.groupby('user_id').unreordered_sum.cumsum()
order_tbl['total_unique_item_ratio'] = order_tbl['total_unique_item']/order_tbl['order_number']

del order_unique_item; gc.collect()

col_feature += ['unreordered_sum','total_unique_item', 'total_unique_item_ratio']

#==============================================================================
# ordered item
#==============================================================================

ordered_item = log.groupby('order_id').size().reset_index()
ordered_item.columns = ['order_id', 'ordered_item']

order_tbl = pd.merge(order_tbl, ordered_item, on='order_id', how='left')

order_tbl['total_ordered_item'] = order_tbl.groupby('user_id').ordered_item.cumsum()
order_tbl['total_ordered_item_ratio'] = order_tbl['total_ordered_item']/order_tbl['order_number']

del ordered_item; gc.collect()

col_feature += ['ordered_item','total_ordered_item', 'total_ordered_item_ratio']



#==============================================================================
# merge & split
#==============================================================================
col = ['order_id', 'order_dow', 'order_hour_of_day', 
       'days_since_prior_order', 'days_since_first_order']
for i in range(1, 1+T):
    X = pd.merge(X, order_tbl[col+col_feature].add_prefix('t-{}_'.format(i)), 
                 on='t-{}_order_id'.format(i), how='left')


train = X[X.is_train==1].drop(['user_id','is_train'], axis=1).reset_index(drop=1)
test  = X[X.is_train==0].drop(['user_id','is_train'], axis=1).reset_index(drop=1)

#==============================================================================
# write
#==============================================================================
col = [c for c in train.columns if not ('t-' in c and '_id' in c)]
train[col].to_pickle('../feature/trainT-0/f101_order.p')
test[col].to_pickle('../feature/test/f101_order.p')


#==============================================================================
utils.end(__file__)

