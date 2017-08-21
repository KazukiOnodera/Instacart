#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 15:58:46 2017

@author: konodera

order span

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import utils
utils.start(__file__)


X_base = pd.read_pickle('../feature/X_base_t3.p')
col = ['order_id', 'user_id', 'days_since_prior_order', 'eval_set', 'order_number_rev']
order_tbl = pd.read_pickle('../input/mk/order_tbl.p')[col]

#==============================================================================
# train
#==============================================================================
def make(T):
    order_tbl_tr = order_tbl[order_tbl.order_number_rev>T]
    
    user = order_tbl_tr.groupby('user_id')['days_since_prior_order'].mean().reset_index()
    user.columns = ['user_id', 'days_order_mean']
    
    user.to_pickle('../feature/trainT-{}/f102_user.p'.format(T))

make(0)
make(1)
make(2)



#==============================================================================
# test
#==============================================================================
order_tbl_te = order_tbl[order_tbl.eval_set != 'test']

user = order_tbl_te.groupby('user_id')['days_since_prior_order'].mean().reset_index()
user.columns = ['user_id', 'days_order_mean']

user.to_pickle('../feature/test/f102_user.p')











#==============================================================================
utils.end(__file__)

