#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 22:12:32 2017

@author: konodera

そのユーザーがそのアイテムをいくつ買ったか
*リークしてない

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import utils
utils.start(__file__)


col = ['order_id', 'user_id', 'product_id', 'order_number_rev']
log = utils.read_pickles('../input/mk/log', col)

orders = pd.read_csv('../input/orders.csv.gz',usecols=['order_id','user_id','order_number'])

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
    df = pd.merge(label, orders, on='order_id', how='left')
    
    total_buy = log[log.order_number_rev>T].groupby(['user_id', 'product_id']).size().reset_index()
    total_buy.columns = ['user_id', 'product_id','total_buy']
    
    df = pd.merge(df, total_buy, on=['user_id', 'product_id'], how='left')
    df['total_buy_ratio'] = df.total_buy / (df.order_number-1)

    col = ['order_id', 'product_id','total_buy', 'total_buy_ratio']
    df[col].to_pickle('../feature/{}/f301_order-product.p'.format(folder))
    
    # near5
    df = pd.merge(label, orders, on='order_id', how='left')
    total_buy = log[log.order_number_rev>T][log.order_number_rev<=(T+5)].groupby(['user_id', 'product_id']).size().reset_index()
    total_buy.columns = ['user_id', 'product_id','total_buy_n5']
    
    df = pd.merge(df, total_buy, on=['user_id', 'product_id'], how='left').fillna(0)
    df['total_buy_ratio_n5'] = df['total_buy_n5'] / df.order_number.map(lambda x: min(5, x))

    col = ['order_id', 'product_id','total_buy_n5', 'total_buy_ratio_n5']
    df[col].to_pickle('../feature/{}/f301_order-product_n5.p'.format(folder))
    

#==============================================================================
# main
#==============================================================================
make(0)
make(1)
make(2)

make(-1)

#==============================================================================
utils.end(__file__)

