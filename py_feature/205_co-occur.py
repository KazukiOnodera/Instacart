#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 22:00:22 2017

@author: konodera

nohup python -u 205_co-occur.py &


=== order_numberまたぎ共起 ===
t-1に何を買うと(自分以外)、t-0にreorderする？
exp:
    t-1にbananaを買った人の30%がt-0にstrawberryを買う
    
takes 3 hour
"""

import pandas as pd
import numpy as np
#from tqdm import tqdm
from collections import Counter
from itertools import product
from operator import itemgetter
import gc
import multiprocessing as mp
import utils
utils.start(__file__)

#==============================================================================
# load
#==============================================================================
col = ['order_id', 'user_id', 'product_name', 't-1_product_name', 'order_number', 'order_number_rev']
order_tbl = pd.read_pickle('../input/mk/order_tbl.p')[col]
order_tbl.sort_values(['user_id','order_number'], inplace=1)
order_tbl['t-1_order_id'] = order_tbl.groupby('user_id')['order_id'].shift(1)
order_tbl.reset_index(drop=True, inplace=True)

prods = pd.read_pickle('../input/mk/goods.p')[['product_id','product_name']]

log = utils.read_pickles('../input/mk/log', ['order_id', 'product_id', 'order_number_rev'])
order_item_array = log.groupby('order_id').product_id.apply(np.array).reset_index()
del log; gc.collect()
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
    print("start T:{} folder:{}".format(T, folder))
    order_tbl_ = order_tbl[order_tbl.order_number_rev>T].dropna() # drop first order
    
    item2item = []
    item_bunbo = Counter()
    for item_prior, item_now in order_tbl_[['t-1_product_name', 'product_name']].values:
        item2item  += [i1+' -> '+i2 for i1, i2 in list(product(item_prior, item_now))]
        item_bunbo += Counter(item_prior)
    item2item = Counter(item2item)

    df = pd.DataFrame.from_dict(item2item, orient='index').reset_index()
    df.columns = ['item', 'cnt']
    del item2item; gc.collect()

    df_ = pd.DataFrame.from_dict(item_bunbo, orient='index').reset_index()
    df_.columns = ['before', 'total_cnt']
    del item_bunbo; gc.collect()
    
    df.sort_values('cnt', ascending=False, inplace=True)
    
    df['before'] = df.item.map(lambda x: x.split(' -> ')[0])
    df['after'] = df.item.map(lambda x: x.split(' -> ')[1])
    df = df[df.before!=df.after]
    
    df = pd.merge(df, df_, on='before', how='left')
    
    df['before_to_after_ratio'] = df.cnt / df.total_cnt
    df = df[['before', 'after', 'before_to_after_ratio']]
    gc.collect()

    df = pd.merge(df, prods.rename(columns={'product_name':'before', 'product_id':'before_id'}), 
                   on='before', how='left')
    df = pd.merge(df, prods.rename(columns={'product_name':'after', 'product_id':'after_id'}), 
                   on='after', how='left')
    
    df = df[['before_id', 'after_id', 'before_to_after_ratio']]
    gc.collect()
    """
    df.head()
          before_id  after_id  before_to_after_ratio
    0      47209     13176               0.288618
    1      13176     47209               0.175736
    2      13176     21137               0.148974
    3      21137     13176               0.188769
    """
    #==============================================================================
    print('Merge', T)
    #==============================================================================
    label = pd.read_pickle('../feature/{}/label_reordered.p'.format(folder))
    label = pd.merge(label, order_tbl[['order_id', 't-1_order_id']], 
                     on='order_id', how='left')
    print('今まで買ったitem and t-1に買ったitem')
    order_b4after = pd.merge(label, order_item_array.add_prefix('t-1_'), 
                             on='t-1_order_id', how='left')
    gc.collect()
    
    col = ['order_id', 't-1_product_id', 'product_id']
    order_b4after = order_b4after[col]
    gc.collect()
    """
    order_b4after.head()
    Out[9]:
       order_id                                     t-1_product_id  product_id
    0   1187899  [46149, 39657, 38928, 25133, 10258, 35951, 130...         196
    1   1187899  [46149, 39657, 38928, 25133, 10258, 35951, 130...       10258
    2   1187899  [46149, 39657, 38928, 25133, 10258, 35951, 130...       10326
    3   1187899  [46149, 39657, 38928, 25133, 10258, 35951, 130...       12427
    4   1187899  [46149, 39657, 38928, 25133, 10258, 35951, 130...       13032
    """
    #==============================================================================
    print('search max ratio',T)
    #==============================================================================
    df['key'] = df.before_id.map(str) + 'to' + df.after_id.map(str)
    
    ratio_tbl = {}
    for k,v in df[['key','before_to_after_ratio']].values:
        ratio_tbl[k] = v
    
    del df; gc.collect()
    
    def get_ratio(key):
        try:
            return ratio_tbl[key]
        except:
            return -1
    
    def search_max_ratio(before_items, item):
        """
        before_items = order_tr.loc[0,'t-1_product_id']
        item    = order_tr.loc[0,'product_id']    
        """
        comb = list(product(before_items, [item]))
        comb = [str(x) + 'to' + str(y) for x,y in sorted(comb, key=itemgetter(1))]
        return np.max([get_ratio(k) for k in comb])


    print('== before_to_after_ratio ==', T)
    ret = []
    for before_items, item in order_b4after[['t-1_product_id', 'product_id']].values:
        ret.append(search_max_ratio(before_items, item))
    order_b4after['before_to_after_ratio'] = ret
    
    col = ['order_id', 'product_id', 'before_to_after_ratio']
    order_b4after[col].to_pickle('../feature/{}/f205_order_product.p'.format(folder))
    
#==============================================================================
# main
#==============================================================================

mp_pool = mp.Pool(3)
mp_pool.map(make, [-1, 0, 1, 2, #3,# 4, 5
                   ])




#==============================================================================
utils.end(__file__)











