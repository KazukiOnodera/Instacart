#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 04:11:27 2017

@author: konodera


nohup python -u 501_concat.py &

"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import gc
import utils
utils.start(__file__)

#==============================================================================
# def
#==============================================================================
def user_feature(df, name):
    
    if 'train' in name:
        name_ = 'trainT-0'
    elif name == 'test':
        name_ = 'test'
        
    df = pd.merge(df, pd.read_pickle('../feature/{}/f101_order.p'.format(name_)),# same
                  on='order_id', how='left')
    # timezone
    df = pd.merge(df, pd.read_pickle('../input/mk/timezone.p'), 
                  on='order_hour_of_day', how='left')
    df = pd.merge(df, pd.read_pickle('../feature/{}/f102_user.p'.format(name)),
                  on='user_id', how='left')
    df = pd.merge(df, pd.read_pickle('../feature/{}/f103_user.p'.format(name)),
                  on='user_id', how='left')
    df = pd.merge(df, pd.read_pickle('../feature/{}/f104_user.p'.format(name)),
                  on='user_id', how='left')
    df = pd.merge(df, pd.read_pickle('../feature/{}/f105_order.p'.format(name_)),# same 
                  on='order_id', how='left')
    df = pd.merge(df, pd.read_pickle('../feature/{}/f110_order.p'.format(name_)),# same 
                  on='order_id', how='left')
    gc.collect()
    
    return df

def item_feature(df, name):
    
#    aisle =  pd.read_pickle('../input/mk/goods.p')[['product_id', 'aisle_id']]
#    aisle = pd.get_dummies(aisle.rename(columns={'aisle_id':'item_aisle'}), columns=['item_aisle'])
#    df = pd.merge(df, aisle, on='product_id', how='left')
    
    organic = pd.read_pickle('../input/mk/products_feature.p')
    df = pd.merge(df, organic, on='product_id', how='left')
    
    # this could be worse
    df = pd.merge(df, pd.read_pickle('../feature//{}/f202_product_hour.p'.format(name)), 
                     on=['product_id','order_hour_of_day'], how='left')
    df = pd.merge(df, pd.read_pickle('../feature/{}/f202_uniq_product_hour.p'.format(name)), 
                     on=['product_id','order_hour_of_day'], how='left')
    
    df = pd.merge(df, pd.read_pickle('../feature/{}/f202_product_dow.p'.format(name)), 
                     on=['product_id','order_dow'], how='left')
    df = pd.merge(df, pd.read_pickle('../feature/{}/f202_uniq_product_dow.p'.format(name)), 
                     on=['product_id','order_dow'], how='left')
    gc.collect()
    # low importance
    df = pd.merge(df, pd.read_pickle('../feature/{}/f202_product_timezone.p'.format(name)), 
                     on=['product_id','timezone'], how='left')
    df = pd.merge(df, pd.read_pickle('../feature/{}/f202_uniq_product_timezone.p'.format(name)), 
                     on=['product_id','timezone'], how='left')
    
    # low importance
    df = pd.merge(df, pd.read_pickle('../feature/{}/f202_product_dow-timezone.p'.format(name)), 
                     on=['product_id', 'order_dow', 'timezone'], how='left')
    df = pd.merge(df, pd.read_pickle('../feature/{}/f202_uniq_product_dow-timezone.p'.format(name)), 
                     on=['product_id', 'order_dow', 'timezone'], how='left')
    
    # no boost
    df = pd.merge(df, pd.read_pickle('../feature/{}/f202_flat_product.p'.format(name)),
                     on=['product_id'], how='left')
    
    df = pd.merge(df, pd.read_pickle('../feature/{}/f203_product.p'.format(name)), 
                     on='product_id', how='left')
    
    df = pd.merge(df, pd.read_pickle('../feature/{}/f205_order_product.p'.format(name)), 
                     on=['order_id', 'product_id'], how='left')
    
    df = pd.merge(df, pd.read_pickle('../feature/{}/f207_product.p'.format(name)), 
                     on='product_id', how='left')
    df = pd.merge(df, pd.read_pickle('../feature/{}/f208_product.p'.format(name)), 
                     on='product_id', how='left')
    # low imp
    df = pd.merge(df, pd.read_pickle('../feature/{}/f209_product.p'.format(name)), 
                     on='product_id', how='left')
    
    df = pd.merge(df, pd.read_pickle('../feature/{}/f210_product.p'.format(name)), 
                     on='product_id', how='left')
    df = pd.merge(df, pd.read_pickle('../feature/{}/f211_product.p'.format(name)), 
                     on='product_id', how='left')
    df = pd.merge(df, pd.read_pickle('../feature/{}/f212_product.p'.format(name)), 
                     on='product_id', how='left')
    df = pd.merge(df, pd.read_pickle('../feature/{}/f213_product-dow.p'.format(name)), 
                     on=['product_id','order_dow'], how='left')
    df = pd.merge(df, pd.read_pickle('../feature/{}/f214_product.p'.format(name)), 
                     on='product_id', how='left')
    df = pd.merge(df, pd.read_pickle('../feature/{}/f215_product.p'.format(name)), 
                     on='product_id', how='left')
    gc.collect()
    return df

def user_item_feature(df, name):
    
    df = pd.merge(df, pd.read_pickle('../feature/{}/f301_order-product.p'.format(name)), 
                     on=['order_id', 'product_id'],how='left')
    df = pd.merge(df, pd.read_pickle('../feature/{}/f301_order-product_n5.p'.format(name)), 
                     on=['order_id', 'product_id'],how='left')
    
    df = pd.merge(df, pd.read_pickle('../feature/{}/f302_order-product_all.p'.format(name)), 
                     on=['order_id', 'product_id'],how='left')
    df = pd.merge(df, pd.read_pickle('../feature/{}/f303_order-product.p'.format(name)), 
                     on=['order_id', 'product_id'],how='left')
    
    df = pd.merge(df, pd.read_pickle('../feature/{}/f304-1_order-product.p'.format(name)), 
                     on=['order_id', 'product_id'],how='left')
    df = pd.merge(df, pd.read_pickle('../feature/{}/f304-2_order-product.p'.format(name)), 
                     on=['order_id', 'product_id'],how='left')
    df = pd.merge(df, pd.read_pickle('../feature/{}/f304-3_order-product.p'.format(name)), 
                     on=['order_id', 'product_id'],how='left')
    
    df = pd.merge(df, pd.read_pickle('../feature/{}/f305_order-product.p'.format(name)), 
                     on=['order_id', 'product_id'], how='left')
    
    df = pd.merge(df, pd.read_pickle('../feature/{}/f306_user-product.p'.format(name)), 
                     on=['user_id', 'product_id'], how='left')
    df = pd.merge(df, pd.read_pickle('../feature/{}/f306_user-product_n5.p'.format(name)), 
                     on=['user_id', 'product_id'], how='left')
    gc.collect()
    
    df = pd.merge(df, pd.read_pickle('../feature/{}/f307_user-product-timezone.p'.format(name)), 
                     on=['user_id', 'product_id','timezone'], how='left')
    df = pd.merge(df, pd.read_pickle('../feature/{}/f307_user-product-dow.p'.format(name)), 
                     on=['user_id', 'product_id','order_dow'], how='left')
    
    df = pd.merge(df, pd.read_pickle('../feature/{}/f308_user-product-timezone.p'.format(name)), 
                     on=['user_id', 'product_id','timezone'], how='left')
    df = pd.merge(df, pd.read_pickle('../feature/{}/f308_user-product-dow.p'.format(name)), 
                     on=['user_id', 'product_id','order_dow'], how='left')
    
    df = pd.merge(df, pd.read_pickle('../feature/{}/f309_user-product.p'.format(name)), 
                     on=['user_id', 'product_id'], how='left')
    df = pd.merge(df, pd.read_pickle('../feature/{}/f309_user-product_n5.p'.format(name)), 
                     on=['user_id', 'product_id'], how='left')
    
    df = pd.merge(df, pd.read_pickle('../feature/{}/f310_user-product.p'.format(name)), 
                     on=['user_id', 'product_id'], how='left')
    
    df = pd.merge(df, pd.read_pickle('../feature/{}/f312_user_product.p'.format(name)), 
                     on=['user_id', 'product_id'], how='left')
    df = pd.merge(df, pd.read_pickle('../feature/{}/f312_user_product_n5.p'.format(name)), 
                     on=['user_id', 'product_id'], how='left')
    
    df = pd.merge(df, pd.read_pickle('../feature/{}/f313_user_aisle.p'.format(name)), 
                     on=['user_id', 'aisle_id'], how='left')
    df = pd.merge(df, pd.read_pickle('../feature/{}/f313_user_dep.p'.format(name)), 
                     on=['user_id', 'department_id'], how='left')
    
    df = pd.merge(df, pd.read_pickle('../feature/{}/f314_user-product.p'.format(name)), 
                     on=['user_id', 'product_id'], how='left')
    
    df = pd.merge(df, pd.read_pickle('../feature/{}/f315-1_order-product.p'.format(name)), 
                     on=['order_id', 'product_id'],how='left')
    df = pd.merge(df, pd.read_pickle('../feature/{}/f315-2_order-product.p'.format(name)), 
                     on=['order_id', 'product_id'],how='left')
    df = pd.merge(df, pd.read_pickle('../feature/{}/f315-3_order-product.p'.format(name)), 
                     on=['order_id', 'product_id'],how='left')
    
    df = pd.merge(df, pd.read_pickle('../feature/{}/f316_order_product.p'.format(name)), 
                     on=['order_id', 'product_id'],how='left')
    gc.collect()
    return df

def daytime_feature(df, name):
    
    df = pd.merge(df, pd.read_pickle('../feature/{}/f401_dow.p'.format(name)), 
                  on=['order_dow'], how='left')
    df = pd.merge(df, pd.read_pickle('../feature/{}/f401_hour.p'.format(name)), 
                  on=['order_hour_of_day'], how='left')
    
    return df

def concat_pred_item(T, dryrun=False):
    if T==-1:
        name = 'test'
    else:
        name = 'trainT-'+str(T)
    #==============================================================================
    print('load label')
    #==============================================================================
    # NOTE: order_id is label
    print('load t3')
    X_base = pd.read_pickle('../feature/X_base_t3.p')
    
    label = pd.read_pickle('../feature/{}/label_reordered.p'.format(name))
    
    # 'inner' for removing t-n_order_id == NaN
    if 'train' in name:
        df = pd.merge(X_base[X_base.is_train==1], label, on='order_id', how='inner')
    elif name == 'test':
        df = pd.merge(X_base[X_base.is_train==0], label, on='order_id', how='inner')
    
    if dryrun:
        print('dryrun')
        df = df.sample(9999)
    
    df = pd.merge(df, pd.read_pickle('../input/mk/goods.p')[['product_id', 'aisle_id', 'department_id']], 
                  on='product_id', how='left')
    
    print('{}.shape:{}\n'.format(name, df.shape))
        
    #==============================================================================
    print('user feature')
    #==============================================================================
    
    df = user_feature(df, name)
    
    print('{}.shape:{}\n'.format(name, df.shape))
    
    #==============================================================================
    print('item feature')
    #==============================================================================
    
    df = item_feature(df, name)
    
    print('{}.shape:{}\n'.format(name, df.shape))
    
    #==============================================================================
    print('reduce memory')
    #==============================================================================
    utils.reduce_memory(df)
    ix_end = df.shape[1]
    
    #==============================================================================
    print('user x item')
    #==============================================================================
    
    df = user_item_feature(df, name)
    
    print('{}.shape:{}\n'.format(name, df.shape))
    
    #==============================================================================
    print('user x item')
    #==============================================================================
    def compress(df, key):
        """
        key: str
        """
        df_ = df.drop_duplicates(key)[[key]].set_index(key)
        dtypes = df.dtypes
        col = dtypes[dtypes!='O'].index
        col = [c for c in col if '_id' not in c]
        gr = df.groupby(key)
        for c in col:
            df_[c+'-min'] = gr[c].min()
            df_[c+'-mean'] = gr[c].mean()
            df_[c+'-median'] = gr[c].median()
            df_[c+'-max'] = gr[c].max()
            df_[c+'-std'] = gr[c].std()
            
        var = df_.var()
        col = var[var==0].index
        df_.drop(col, axis=1, inplace=True)
        gc.collect()
        
        return df_.reset_index()
    
    key = 'order_id'
    feature = compress(pd.read_pickle('../feature/{}/f301_order-product.p'.format(name)), key)
    df = pd.merge(df, feature, on=key, how='left')
    feature = compress(pd.read_pickle('../feature/{}/f301_order-product_n5.p'.format(name)), key)
    df = pd.merge(df, feature, on=key, how='left')
    
    key = 'order_id'
    feature = compress(pd.read_pickle('../feature/{}/f302_order-product_all.p'.format(name)), key)
    df = pd.merge(df, feature, on=key, how='left')
    
    key = 'order_id'
    feature = compress(pd.read_pickle('../feature/{}/f303_order-product.p'.format(name)), key)
    df = pd.merge(df, feature, on=key, how='left')
    
    key = 'order_id'
    feature = compress(pd.read_pickle('../feature/{}/f304-1_order-product.p'.format(name)), key)
    df = pd.merge(df, feature, on=key, how='left')
    
    key = 'order_id'
    feature = compress(pd.read_pickle('../feature/{}/f304-2_order-product.p'.format(name)), key)
    df = pd.merge(df, feature, on=key, how='left')
    
    key = 'order_id'
    feature = compress(pd.read_pickle('../feature/{}/f304-3_order-product.p'.format(name)), key)
    df = pd.merge(df, feature, on=key, how='left')
    
    key = 'order_id'
    feature = compress(pd.read_pickle('../feature/{}/f305_order-product.p'.format(name)), key)
    df = pd.merge(df, feature, on=key, how='left')
    gc.collect()
    
    key = 'user_id'
    feature = compress(pd.read_pickle('../feature/{}/f306_user-product.p'.format(name)), key)
    df = pd.merge(df, feature, on=key, how='left')
    feature = compress(pd.read_pickle('../feature/{}/f306_user-product_n5.p'.format(name)), key)
    df = pd.merge(df, feature, on=key, how='left')
    
    key = 'user_id'
    feature = compress(pd.read_pickle('../feature/{}/f307_user-product-timezone.p'.format(name)), key)
    df = pd.merge(df, feature, on=key, how='left')
    
    key = 'user_id'
    feature = compress(pd.read_pickle('../feature/{}/f308_user-product-timezone.p'.format(name)), key)
    df = pd.merge(df, feature, on=key, how='left')
    
    key = 'user_id'
    feature = compress(pd.read_pickle('../feature/{}/f308_user-product-dow.p'.format(name)), key)
    df = pd.merge(df, feature, on=key, how='left')
    
    key = 'user_id'
    feature = compress(pd.read_pickle('../feature/{}/f309_user-product.p'.format(name)), key)
    df = pd.merge(df, feature, on=key, how='left')
    feature = compress(pd.read_pickle('../feature/{}/f309_user-product_n5.p'.format(name)), key)
    df = pd.merge(df, feature, on=key, how='left')
    
    key = 'user_id'
    feature = compress(pd.read_pickle('../feature/{}/f310_user-product.p'.format(name)), key)
    df = pd.merge(df, feature, on=key, how='left')
    
    key = 'user_id'
    feature = compress(pd.read_pickle('../feature/{}/f312_user_product.p'.format(name)), key)
    df = pd.merge(df, feature, on=key, how='left')
    feature = compress(pd.read_pickle('../feature/{}/f312_user_product_n5.p'.format(name)), key)
    df = pd.merge(df, feature, on=key, how='left')
    gc.collect()
    
    key = 'user_id'
    feature = compress(pd.read_pickle('../feature/{}/f313_user_aisle.p'.format(name)), key)
    df = pd.merge(df, feature, on=key, how='left')
    
    key = 'user_id'
    feature = compress(pd.read_pickle('../feature/{}/f313_user_dep.p'.format(name)), key)
    df = pd.merge(df, feature, on=key, how='left')
    
    key = 'user_id'
    feature = compress(pd.read_pickle('../feature/{}/f314_user-product.p'.format(name)), key)
    df = pd.merge(df, feature, on=key, how='left')
    
    key = 'order_id'
    feature = compress(pd.read_pickle('../feature/{}/f315-1_order-product.p'.format(name)), key)
    df = pd.merge(df, feature, on=key, how='left')
    
    key = 'order_id'
    feature = compress(pd.read_pickle('../feature/{}/f315-2_order-product.p'.format(name)), key)
    df = pd.merge(df, feature, on=key, how='left')
    
    key = 'order_id'
    feature = compress(pd.read_pickle('../feature/{}/f315-3_order-product.p'.format(name)), key)
    df = pd.merge(df, feature, on=key, how='left')
    
    key = 'order_id'
    feature = compress(pd.read_pickle('../feature/{}/f316_order_product.p'.format(name)), key)
    df = pd.merge(df, feature, on=key, how='left')
    
    gc.collect()
    
    #==============================================================================
    print('reduce memory')
    #==============================================================================
    utils.reduce_memory(df, ix_end)
    ix_end = df.shape[1]
    
    #==============================================================================
    print('daytime')
    #==============================================================================
    
    df = daytime_feature(df, name)
    
    print('{}.shape:{}\n'.format(name, df.shape))
    
#    #==============================================================================
#    print('aisle')
#    #==============================================================================
#    order_aisdep = pd.read_pickle('../input/mk/order_aisle-department.p')
#    col = [c for c in order_aisdep.columns if 'department_' in c]
#    order_aisdep.drop(col, axis=1, inplace=1)
#    
#    df = pd.merge(df, order_aisdep.add_prefix('t-1_'), on='t-1_order_id', how='left')
#    df = pd.merge(df, order_aisdep.add_prefix('t-2_'), on='t-2_order_id', how='left')
#    
#    print('{}.shape:{}\n'.format(name, df.shape))

    #==============================================================================
    print('feature engineering')
    #==============================================================================
    df = pd.get_dummies(df, columns=['timezone'])
    df = pd.get_dummies(df, columns=['order_dow'])
    df = pd.get_dummies(df, columns=['order_hour_of_day'])
    
    df['days_near_order_cycle'] = (df.days_since_last_order_this_item - df.item_order_days_mean).abs()
    df['days_last_order-min'] = df.days_since_last_order_this_item - df.useritem_order_days_min
    df['days_last_order-max'] = df.days_since_last_order_this_item - df.useritem_order_days_max
    
    df['pos_cart_diff'] = (df.item_mean_pos_cart - df.useritem_mean_pos_cart)
    
    df['t-1_product_unq_len_diffByT-2'] = df['t-1_product_unq_len'] - df['t-2_product_unq_len']
    df['t-1_product_unq_len_diffByT-3'] = df['t-1_product_unq_len'] - df['t-3_product_unq_len']
    df['t-2_product_unq_len_diffByT-3'] = df['t-2_product_unq_len'] - df['t-3_product_unq_len']
    
    df['t-1_product_unq_len_ratioByT-2'] = df['t-1_product_unq_len'] / df['t-2_product_unq_len']
    df['t-1_product_unq_len_ratioByT-3'] = df['t-1_product_unq_len'] / df['t-3_product_unq_len']
    df['t-2_product_unq_len_ratioByT-3'] = df['t-2_product_unq_len'] / df['t-3_product_unq_len']
    
    df['T'] = T
    
    #==============================================================================
    print('reduce memory')
    #==============================================================================
    utils.reduce_memory(df, ix_end)
    
    #==============================================================================
    print('output')
    #==============================================================================
    if dryrun == True:
        return df
    else:
        utils.to_pickles(df, '../feature/{}/all'.format(name), 20, inplace=True)

def multi(name):
    concat_pred_item(name)

#==============================================================================

# multi
mp_pool = mp.Pool(2)
mp_pool.map(multi, [0,1,2,-1])



utils.end(__file__)

