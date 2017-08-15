#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 09:42:55 2017

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

col = ['order_id', 'user_id', 'order_number', 'product_id', 'reordered']
log = utils.read_pickles('../input/mk/log', col)
log.sort_values(['user_id', 'order_number', 'product_id'], inplace=True)


#==============================================================================
# def
#==============================================================================
def multi(T):
    """
    T = 0
    folder = 'trainT-0'
    """
    if T==-1:
        folder = 'test'
    else:
        folder = 'trainT-'+str(T)
        
    label = pd.read_pickle('../feature/{}/label_reordered.p'.format(folder))
    df = pd.merge(label, order_tbl, on='order_id', how='left') 
    
    for i in tqdm(range(1, LOOP)):
        oid = 't-{}_order_id'.format(i)
        v = 't-{}_reordered'.format(i)
        log_ = log.rename(columns={'order_id':oid, 
                                   'reordered':v})[[oid, 'product_id', v]]
        df = pd.merge(df, log_, on=[oid, 'product_id'], how='left')
    
    col = ['order_id', 'product_id'] + [c for c in df.columns if '_reordered' in c]
    
    df[col].fillna(-1).to_pickle('../feature/{}/f302_order-product_all.p'.format(folder))
#==============================================================================
# main
#==============================================================================
mp_pool = mp.Pool(7)
mp_pool.map(multi, [0, 1, 2, #3, 4, 5, 
                    -1])



#==============================================================================

utils.end(__file__)

