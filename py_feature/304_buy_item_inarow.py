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
import utils
utils.start(__file__)


log = pd.read_pickle('../input/mk/log_inarow.p')
X_base = pd.read_pickle('../feature/X_base_t3.p')

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
        
    label = pd.read_pickle('../feature/{}/label_reordered.p'.format(folder))
    label = pd.merge(label, X_base, on='order_id', how='left') # TODO: change to inner
    
    # ======== T-1~3 ========
    for t in range(1,4):
        col = ['order_id', 'product_id', 'buy_item_inarow']
        df = pd.merge(label, log[col].rename(columns={'order_id':'t-{}_order_id'.format(t)}),
                      on=['t-{}_order_id'.format(t),'product_id'], how='left')
        
        col = ['order_id', 'order_number']
        df = pd.merge(df, log[col].rename(columns={'order_id':'t-{}_order_id'.format(t)}).drop_duplicates(),
                      on=['t-{}_order_id'.format(t)], how='left')
        
        df['buy_item_inarow_ratio'] = df['buy_item_inarow']/df['order_number']
        df = df.rename(columns={'buy_item_inarow':'t-{}_buy_item_inarow'.format(t),
                                'buy_item_inarow_ratio':'t-{}_buy_item_inarow_ratio'.format(t)})
        print(df.isnull().sum())
        df.fillna(0, inplace=1)
        df.reset_index(drop=1, inplace=1)
        
        col = ['order_id', 'product_id', 't-{}_buy_item_inarow'.format(t),'t-{}_buy_item_inarow_ratio'.format(t)]
        df[col].to_pickle('../feature/{}/f304-{}_order-product.p'.format(folder, t))
    
#==============================================================================
# main
#==============================================================================
make(0)
make(1)
make(2)

make(-1)

#==============================================================================
utils.end(__file__)

