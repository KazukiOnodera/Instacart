#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 07:51:30 2017

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
X_base = pd.read_pickle('../feature/X_base_t3.p')


col = ['order_id', 'user_id', 'product_id', 'order_dow', 'order_hour_of_day', 'order_number_rev']
log = utils.read_pickles('../input/mk/log', col)
log = pd.merge(log, pd.read_pickle('../input/mk/timezone.p'), 
               on='order_hour_of_day', how='left')
log['dow_tz'] = log.order_dow.map(str) + '_' + log.timezone

log = pd.merge(log, pd.read_pickle('../input/mk/products_feature.p'), 
               on='product_id', how='left')

#==============================================================================
# train
#==============================================================================
def make(T):
    log_tr = log[log.order_number_rev>T]
    
    user = log_tr.groupby(['user_id']).size().to_frame()
    user.columns = ['total']
    user['organic_cnt'] = log_tr.groupby(['user_id'])['item_is_Organic'].sum()
    user['glutenfree_cnt'] = log_tr.groupby(['user_id'])['item_is_Gluten-Free'].sum()
    user['Asian_cnt'] = log_tr.groupby(['user_id'])['item_is_Asian'].sum()
    
    user['organic_ratio'] = user['organic_cnt'] / user.total
    user['glutenfree_ratio'] = user['glutenfree_cnt'] / user.total
    user['Asian_ratio'] = user['Asian_cnt'] / user.total
    
    user.drop('total', axis=1, inplace=True)
    user.reset_index().to_pickle('../feature/trainT-{}/f104_user.p'.format(T))

make(0)
make(1)
make(2)

#==============================================================================
# test
#==============================================================================

user = log.groupby(['user_id']).size().to_frame()
user.columns = ['total']
user['organic_cnt'] = log.groupby(['user_id'])['item_is_Organic'].sum()
user['glutenfree_cnt'] = log.groupby(['user_id'])['item_is_Gluten-Free'].sum()
user['Asian_cnt'] = log.groupby(['user_id'])['item_is_Asian'].sum()

user['organic_ratio'] = user['organic_cnt'] / user.total
user['glutenfree_ratio'] = user['glutenfree_cnt'] / user.total
user['Asian_ratio'] = user['Asian_cnt'] / user.total

user.drop('total', axis=1, inplace=True)
user.reset_index().to_pickle('../feature/test/f104_user.p')


#==============================================================================
utils.end(__file__)

