#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 17:05:46 2017

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
order_tbl = pd.read_pickle('../input/mk/order_tbl.p')
order_tbl = order_tbl[['order_id', 'user_id', 'order_number', 'order_number_rev']]
order_tbl.sort_values(['user_id', 'order_number', 'order_id'], inplace=True)

test_order = pd.read_pickle('../input/mk/test_user.p')

#==============================================================================
# def
#==============================================================================
def main(T):
    for i in range(1, 1+T):
        order_tbl['t-{}_order_id'.format(i)] = order_tbl.groupby('user_id')['order_id'].shift(i)
    
    order_tbl.dropna(inplace=True)
    
    col = [c for c in order_tbl.columns if 'order_id' in c]
    for c in col:
        order_tbl[c] = order_tbl[c].map(int)
    
    order_tbl.reset_index(drop=1, inplace=True)
    
    order_tbl['is_train'] = 1-order_tbl.order_id.isin(test_order.order_id)*1
    
    order_tbl[col+['user_id','is_train']].to_pickle('../feature/X_base_t{}.p'.format(T))


main(3)
main(5)

#==============================================================================
utils.end(__file__)

"""
206209 rows
"""