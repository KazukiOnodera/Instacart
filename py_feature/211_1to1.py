#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 22:36:10 2017

@author: konodera



item_1to1_cnt
item_1to1_ratio
item_11to1_cnt
item_11to1_ratio
item_10to1_cnt
item_10to1_ratio

takes 4 hours

"""

import pandas as pd
import gc
from collections import defaultdict
import multiprocessing as mp
total_proc = 60
import utils
utils.start(__file__)

#==============================================================================
# load
#==============================================================================

col = ['order_id', 'user_id', 'product_id', 'order_number', 'order_number_rev']
log = utils.read_pickles('../input/mk/log', col)

#==============================================================================
# def
#==============================================================================
def multi(uid):
    
    item_1to1_cnt = defaultdict(int)
    item_1to1_chance = defaultdict(int)
    item_11to1_cnt = defaultdict(int)
    item_11to1_chance = defaultdict(int)
    item_10to1_cnt = defaultdict(int)
    item_10to1_chance = defaultdict(int)
    item_111to1_cnt = defaultdict(int)
    item_111to1_chance = defaultdict(int)
    item_110to1_cnt = defaultdict(int)
    item_110to1_chance = defaultdict(int)
    item_101to1_cnt = defaultdict(int)
    item_101to1_chance = defaultdict(int)
    item_100to1_cnt = defaultdict(int)
    item_100to1_chance = defaultdict(int)
    
    tmp = log_[log_.user_id==uid]
    onb_max = tmp.order_number.max()
    ct = pd.crosstab(tmp.order_number, tmp.product_id).reset_index().set_index('order_number')
    for pid in ct.columns:
        odr_bk1 = odr_bk2 = odr_bk3 = None
        for onb,odr in enumerate(ct[pid].values):
            onb+=1
            
            # 1 -> ?
            if odr_bk1==1:
                if onb != onb_max:
                    item_1to1_chance[pid] += 1
                if odr==1:
                    item_1to1_cnt[pid] += 1
                    
            # 11 -> ?
            if odr_bk2==1 and odr_bk1==1:
                if onb != onb_max:
                    item_11to1_chance[pid] += 1
                if odr==1:
                    item_11to1_cnt[pid] += 1
                    
            # 10 -> ?
            if odr_bk2==1 and odr_bk1==0:
                if onb != onb_max:
                    item_10to1_chance[pid] += 1
                if odr==1:
                    item_10to1_cnt[pid] += 1
            
            # 111 -> ?
            if odr_bk3==1 and odr_bk2==1 and odr_bk1==1:
                if onb != onb_max:
                    item_111to1_chance[pid] += 1
                if odr==1:
                    item_111to1_cnt[pid] += 1
            
            # 110 -> ?
            if odr_bk3==1 and odr_bk2==1 and odr_bk1==0:
                if onb != onb_max:
                    item_110to1_chance[pid] += 1
                if odr==1:
                    item_110to1_cnt[pid] += 1
            
            # 101 -> ?
            if odr_bk3==1 and odr_bk2==0 and odr_bk1==1:
                if onb != onb_max:
                    item_101to1_chance[pid] += 1
                if odr==1:
                    item_101to1_cnt[pid] += 1
                    
            # 100 -> ?
            if odr_bk3==1 and odr_bk2==0 and odr_bk1==0:
                if onb != onb_max:
                    item_100to1_chance[pid] += 1
                if odr==1:
                    item_100to1_cnt[pid] += 1
                    
            odr_bk3 = odr_bk2
            odr_bk2 = odr_bk1
            odr_bk1 = odr
    
    if len(item_1to1_cnt)==0:
        item_1to1_cnt[pid] = 0
    if len(item_1to1_chance)==0:
        item_1to1_chance[pid] = 0
        
    if len(item_11to1_cnt)==0:
        item_11to1_cnt[pid] = 0
    if len(item_11to1_chance)==0:
        item_11to1_chance[pid] = 0
        
    if len(item_10to1_cnt)==0:
        item_10to1_cnt[pid] = 0
    if len(item_10to1_chance)==0:
        item_10to1_chance[pid] = 0
        
    if len(item_111to1_cnt)==0:
        item_111to1_cnt[pid] = 0
    if len(item_111to1_chance)==0:
        item_111to1_chance[pid] = 0
        
    if len(item_110to1_cnt)==0:
        item_110to1_cnt[pid] = 0
    if len(item_110to1_chance)==0:
        item_110to1_chance[pid] = 0
        
    if len(item_101to1_cnt)==0:
        item_101to1_cnt[pid] = 0
    if len(item_101to1_chance)==0:
        item_101to1_chance[pid] = 0
        
    if len(item_100to1_cnt)==0:
        item_100to1_cnt[pid] = 0
    if len(item_100to1_chance)==0:
        item_100to1_chance[pid] = 0
        
    item_1to1_cnt = pd.DataFrame.from_dict(item_1to1_cnt, orient='index').reset_index()
    item_1to1_cnt.columns = ['product_id', 'item_1to1_cnt']
    item_1to1_chance = pd.DataFrame.from_dict(item_1to1_chance, orient='index').reset_index()
    item_1to1_chance.columns = ['product_id', 'item_1to1_chance']
    
    item_11to1_cnt = pd.DataFrame.from_dict(item_11to1_cnt, orient='index').reset_index()
    item_11to1_cnt.columns = ['product_id', 'item_11to1_cnt']
    item_11to1_chance = pd.DataFrame.from_dict(item_11to1_chance, orient='index').reset_index()
    item_11to1_chance.columns = ['product_id', 'item_11to1_chance']
    
    item_10to1_cnt = pd.DataFrame.from_dict(item_10to1_cnt, orient='index').reset_index()
    item_10to1_cnt.columns = ['product_id', 'item_10to1_cnt']
    item_10to1_chance = pd.DataFrame.from_dict(item_10to1_chance, orient='index').reset_index()
    item_10to1_chance.columns = ['product_id', 'item_10to1_chance']
    
    item_111to1_cnt = pd.DataFrame.from_dict(item_111to1_cnt, orient='index').reset_index()
    item_111to1_cnt.columns = ['product_id', 'item_111to1_cnt']
    item_111to1_chance = pd.DataFrame.from_dict(item_111to1_chance, orient='index').reset_index()
    item_111to1_chance.columns = ['product_id', 'item_111to1_chance']
    
    item_110to1_cnt = pd.DataFrame.from_dict(item_110to1_cnt, orient='index').reset_index()
    item_110to1_cnt.columns = ['product_id', 'item_110to1_cnt']
    item_110to1_chance = pd.DataFrame.from_dict(item_110to1_chance, orient='index').reset_index()
    item_110to1_chance.columns = ['product_id', 'item_110to1_chance']
    
    item_101to1_cnt = pd.DataFrame.from_dict(item_101to1_cnt, orient='index').reset_index()
    item_101to1_cnt.columns = ['product_id', 'item_101to1_cnt']
    item_101to1_chance = pd.DataFrame.from_dict(item_101to1_chance, orient='index').reset_index()
    item_101to1_chance.columns = ['product_id', 'item_101to1_chance']
    
    item_100to1_cnt = pd.DataFrame.from_dict(item_100to1_cnt, orient='index').reset_index()
    item_100to1_cnt.columns = ['product_id', 'item_100to1_cnt']
    item_100to1_chance = pd.DataFrame.from_dict(item_100to1_chance, orient='index').reset_index()
    item_100to1_chance.columns = ['product_id', 'item_100to1_chance']
    
    df1 = pd.merge(item_1to1_cnt, item_1to1_chance, on='product_id', how='outer')
    df2 = pd.merge(item_11to1_cnt, item_11to1_chance, on='product_id', how='outer')
    df3 = pd.merge(item_10to1_cnt, item_10to1_chance, on='product_id', how='outer')
    
    df4 = pd.merge(item_111to1_cnt, item_111to1_chance, on='product_id', how='outer')
    df5 = pd.merge(item_110to1_cnt, item_110to1_chance, on='product_id', how='outer')
    df6 = pd.merge(item_101to1_cnt, item_101to1_chance, on='product_id', how='outer')
    df7 = pd.merge(item_100to1_cnt, item_100to1_chance, on='product_id', how='outer')
    
    df123 = pd.merge(pd.merge(df1, df2, on='product_id', how='outer'),
                     df3, on='product_id', how='outer').fillna(0)
    df4567 = pd.merge(pd.merge(df4, df5, on='product_id', how='outer'),
                      pd.merge(df6, df7, on='product_id', how='outer'), 
                      on='product_id', how='outer').fillna(0)
    df = pd.merge(df123, df4567, on='product_id', how='outer')
    return df

def make(T):
    """
    T = 0
    folder = 'trainT-0'
    """
    if T==-1:
        folder = 'test'
    else:
        folder = 'trainT-'+str(T)
    
    global log_
    log_ = log[log.order_number_rev>T]
    
    user_id = log_.user_id.unique()
    mp_pool = mp.Pool(total_proc)
    callback = mp_pool.map(multi, user_id)
    callback = pd.concat(callback)
    gc.collect()
    
    gr = callback.groupby('product_id')
    
    df = gr['item_1to1_cnt'].sum().to_frame()
    df.columns = ['item_1to1_cnt']
    df['item_1to1_chance'] = gr['item_1to1_chance'].sum()
    
    col = ['11to1', '10to1', '111to1', '110to1', '101to1', '100to1']
    for c in col:
        c_cnt    = 'item_{}_cnt'.format(c)
        c_chance = 'item_{}_chance'.format(c)
        df[c_cnt]    = gr[c_cnt].sum()
        df[c_chance] = gr[c_chance].sum()
    
    col = ['1to1', '11to1', '10to1', '111to1', '110to1', '101to1', '100to1']
    for c in col:
        c_cnt    = 'item_{}_cnt'.format(c)
        c_chance = 'item_{}_chance'.format(c)
        df['item_{}_ratio'.format(c)] = df[c_cnt]/df[c_chance]
    
    df.fillna(0, inplace=True)
    df.reset_index(inplace=True)
    print('writing 211 T:',T)
    df.to_pickle('../feature/{}/f211_product.p'.format(folder))    
    

#==============================================================================
# main
#==============================================================================
make(0)
make(1)
make(2)

make(-1)


utils.end(__file__)

