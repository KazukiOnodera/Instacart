#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 23:59:01 2017

@author: konodera

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import utils
utils.start(__file__)

LOOP = 20
#==============================================================================
# load
#==============================================================================
order_tbl = pd.read_pickle('../input/mk/order_tbl.p')[['order_id', 'user_id', 'order_number']].sort_values(['user_id', 'order_number', 'order_id'])
for i in range(1, LOOP):
    order_tbl['t-{}_order_id'.format(i)] = order_tbl.groupby('user_id')['order_id'].shift(i)

col = [c for c in order_tbl.columns if 'order_id' in c]
order_tbl = order_tbl[col]

order_None = pd.read_pickle('../input/mk/order_None.p')


#==============================================================================
# main
#==============================================================================        
df = order_tbl.copy()

for i in tqdm(range(1, LOOP)):
    df = pd.merge(df, order_None.add_prefix('t-{}_'.format(i)), 
                on='t-{}_order_id'.format(i), how='left')
    
col = [c for c in df.columns if c.endswith('_order_id')]
df.drop(col, axis=1, inplace=True)

df.fillna(-1, inplace=True)

df.to_pickle('../feature/trainT-0/f110_order.p')
df.to_pickle('../feature/test/f110_order.p')


#==============================================================================

utils.end(__file__)

