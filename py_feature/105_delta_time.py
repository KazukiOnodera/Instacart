#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 09:41:08 2017

@author: konodera

delta order time

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import utils
utils.start(__file__)

#==============================================================================
# load
#==============================================================================

col = ['order_id', 'user_id','order_number', 'order_dow', 'order_hour_of_day', 
       'days_since_prior_order', 'eval_set']
order_tbl = pd.read_pickle('../input/mk/order_tbl.p')[col]
order_tbl.sort_values(['user_id', 'order_number'], inplace=True)
#order_tbl = order_tbl[order_tbl.eval_set!='test']


#==============================================================================
# main
#==============================================================================
order_tbl['t-1_order_id'] = order_tbl.groupby('user_id')['order_id'].shift(1)
order_tbl['t-2_order_id'] = order_tbl.groupby('user_id')['order_id'].shift(2)
order_tbl['t-3_order_id'] = order_tbl.groupby('user_id')['order_id'].shift(3)

col = ['order_id', 'order_dow', 'order_hour_of_day']
order_tbl = pd.merge(order_tbl, order_tbl[col].add_prefix('t-1_'), on='t-1_order_id', how='left')
order_tbl = pd.merge(order_tbl, order_tbl[col].add_prefix('t-2_'), on='t-2_order_id', how='left')
order_tbl = pd.merge(order_tbl, order_tbl[col].add_prefix('t-3_'), on='t-3_order_id', how='left')

order_tbl['delta_hour_t-1'] = order_tbl['order_hour_of_day'] - order_tbl['t-1_order_hour_of_day']
order_tbl['delta_hour_t-2'] = order_tbl['order_hour_of_day'] - order_tbl['t-2_order_hour_of_day']
order_tbl['delta_hour_t-3'] = order_tbl['order_hour_of_day'] - order_tbl['t-3_order_hour_of_day']


col = ['order_id', 'delta_hour_t-1', 'delta_hour_t-2',
       'delta_hour_t-3']
order_tbl[col].to_pickle('../feature/trainT-0/f105_order.p')
order_tbl[col].to_pickle('../feature/test/f105_order.p')



utils.end(__file__)

