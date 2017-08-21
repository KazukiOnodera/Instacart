#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 02:10:45 2017

@author: konodera

現時点の連続購入記録
*リーク

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import utils
utils.start(__file__)


streak = pd.read_pickle('../input/mk/streak_order-product.p')
X_base = pd.read_pickle('../feature/X_base_t3.p')

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
    label = pd.merge(label, X_base, on='order_id', how='inner')
    
    # ======== T-1~3 ========
    for t in range(1,4):
        
        df = pd.merge(label, streak.rename(columns={'order_id':'t-{}_order_id'.format(t),
                                                    'streak':'t-{}_streak'.format(t)}),
                      on=['t-{}_order_id'.format(t),'product_id'], how='left')
        
        print(df.isnull().sum())
        df.fillna(-99, inplace=1)
        df.reset_index(drop=1, inplace=1)
        
        col = ['order_id', 'product_id', 't-{}_streak'.format(t)]
        df[col].to_pickle('../feature/{}/f315-{}_order-product.p'.format(folder, t))
    
#==============================================================================
# main
#==============================================================================
mp_pool = mp.Pool(3)
callback = mp_pool.map(multi, list(range(-1,3)))

#==============================================================================
utils.end(__file__)

