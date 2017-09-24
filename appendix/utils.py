# -*- coding: utf-8 -*-
"""
Created on Wed May 17 01:21:53 2017

@author: konodera
"""

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from glob import glob
import os
from tqdm import tqdm
import pickle
import time
import gc
from itertools import chain



def start(fname):
    global st_time
    st_time = time.time()
    print("""
#==============================================================================
# START !!! {} PID: {}
#==============================================================================
""".format(fname, os.getpid()))
    
    return

def end(fname):
    
    print("""
#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(fname))
    print('time: {:.2f}min'.format( (time.time() - st_time)/60 ))
    return



def mkdir_p(path):
    try:
        os.stat(path)
    except:
        os.mkdir(path)
    
def to_pickles(df, path, split_size=3, inplace=False):
    """
    path = '../output/mydf'
    
    wirte '../output/mydf/0.p'
          '../output/mydf/1.p'
          '../output/mydf/2.p'
    
    """
    if inplace==True:
        df.reset_index(drop=True, inplace=True)
    else:
        df = df.reset_index(drop=True)
    mkdir_p(path)
    
    for i in tqdm(range(split_size)):
        df.ix[df.index%split_size==i].to_pickle(path+'/{}.p'.format(i))
    
    return

def read_pickles(path, col=None):
    if col is None:
        df = pd.concat([pd.read_pickle(f) for f in tqdm(sorted(glob(path+'/*')))])
    else:
        df = pd.concat([pd.read_pickle(f)[col] for f in tqdm(sorted(glob(path+'/*')))])
    return df


def keep_top_item():
    
    col = ['order_id', 't-1_order_id', 't-2_order_id', 't-3_order_id', 
           'user_id', 'product_id', 'aisle_id', 'department_id', 'is_train', 'y']
    # top 200
    col += [
# BEST 812_1
'total_buy_n5', 'total_buy_ratio_n5', 'days_since_prior_order', 'order_ratio_bychance', 'before_to_after_ratio',
'buy_within_sameday', 'days_last_order-max', 't-13_is_None', 'days_near_order_cycle', 'total_buy_ratio',
'days_since_last_order_this_item', 'order_ratio_bychance_n5', 'useritem_order_days_max_n5', 'item_only_one_user_cnt_ratio', 'item_100to1_ratio',
'order_number_diff', 'days_last_order-min', 'useritem_order_days_median_n5', 'total_buy_ratio-std', 't-17_reordered-median',
'delta_hour_t-1', 'item_10to1_ratio', 'item_first_ratio', 'user_dep_ratio', 'organic_ratio',
'user_aisle_ratio', 'chance-min', 'useritem_order_days_mean_n5', 'pos_cart_diff', 'item_order_days_mean',
'item_order_days_median', 't-1_buy_item_inarow_ratio', 'useritem_order_days_median', 'item_1to1_ratio', 'total_buy',
'item_dow_ratio_diff', 'item_110to1_ratio', 'item_together_mean', 'order_ratio_bychance_n5-std', 'days_order_mean',
'total_buy_ratio-mean', 'item_dow_ratio', 'item_order_days_max', 'item_flat_dow-tz_6_morning', 'item_max_pos_cart',
't-1_reordered-std', 'item_dow_ratio_unq', 'item_flat_dow-tz_2_noon', 'useritem_order_days_max', 'item_flat_dow-tz_1_noon',
'timezone_morning', 'user_dep_ratio-median', 'item_flat_dow-tz_1_night', 'user_aisle_ratio-max', 'item_reorder_ratio',
'item_N2_ratio', 'order_ratio_bychance_n5-mean', 'item_mean_pos_cart', 'delta_hour_t-3', 'useritem_max_pos_cart_n5-max',
'item_flat_dow-tz_5_night', 'item_flat_dow-tz_0_night', 't-1_days_since_prior_order', 'useritem_buy_timezone_ratio2', 'useritem_order_days_mean',
'useritem_mean_pos_cart', 'item_flat_dow-tz_5_noon', 'item_flat_dow-tz_3_midnight', 't-1_buy_item_inarow', 'useritem_buy_timezone_ratio2-std',
'item_together_std', 'useritem_buy_dow_ratio2', 'item_flat_dow-tz_3_noon', 'item_flat_dow-tz_6_noon', 'user_timezone_norm_noon',
'item_flat_dow-tz_4_noon', 'delta_hour_t-2', 'item_flat_dow-tz_6_midnight', 'item_flat_dow-tz_5_midnight', 'order_dow-std',
'item_111to1_ratio', 'order_dow-mean', 't-2_reordered-std', 'item_flat_dow-tz_4_midnight', 'user_dep_ratio-max',
'useritem_min_pos_cart', 'item_flat_dow-tz_0_midnight', 'item_flat_dow-tz_1_midnight', 't-8_reordered', 'user_dep_ratio-std',
'chance_n5-std', 'total_buy_ratio_n5-mean', 'item_flat_dow-tz_1_morning', 'item_flat_dow-tz_0_morning', 'item_flat_dow-tz_3_night',
'item_std_pos_cart', 'item_flat_dow-tz_2_night', 't-6_reordered', 'useritem_order_days_median-median', 'item_timezone_ratio',
'item_flat_dow-tz_6_night', 'item_onb_diff_skew', 'useritem_buy_dow_ratio2-std', 'item_hour_ratio', 'item_flat_dow-tz_4_night',
'item_flat_dow-tz_4_morning', 'user_aisle_cnt', 'item_flat_dow-tz_5_morning', 'item_101to1_ratio', 'days_since_last_order_this_item-median',
'total_buy_n5-mean', 'total_buy-mean', 'useritem_order_days_min-median', 'comeback_ratio_mean-max', 'useritem_order_days_min',
'item_onb_diff_std', 'user_dow_norm_4', 'item_dow-tz_ratio', 'item_flat_dow-tz_0_noon', 'item_onb_diff_mean',
'item_flat_dow-tz_3_morning', 'useritem_buy_dow_ratio', 'useritem_buy_timezone_ratio-mean', 'useritem_sum_pos_cart-median', 't-2_reordered-mean',
'item_flat_dow-tz_2_morning', 'item_flat_dow-tz_2_midnight', 'order_ratio_bychance-std', 'item_only_one_user_cnt', 'user_dow_norm_1',
'useritem_sum_pos_cart', 'user_dow-tz_norm_2_noon', 't-3_reordered-mean', 'item_onb_diff_max', 'item_11to1_ratio',
'item_timezone_ratio_uniq', 'item_hour_ratio_unq', 'user_timezone_norm_morning', 't-7_reordered', 't-1_reordered-mean',
'days_since_last_order_this_item-std', 'useritem_buy_timezone_ratio-std', 'user_dep_cnt', 'user_dow_norm_2', 'user_dow-tz_norm_1_noon',
't-4_reordered-mean', 't-3_reordered-std', 'comeback_ratio_sum-max', 't-1_buy_item_inarow-std', 'comeback_ratio_len-std',
'chance_n5-mean', 'user_aisle_ratio-std', 't-1_repeat_previous_ratio-w1', 'useritem_order_days_mean-median', 'item_dow-tz_ratio_uniq',
't-4_reordered-std', 'chance_n5-min', 'user_timezone_norm_night', 'item_is_Organic', 'comeback_ratio_max-max',
'item_N3_ratio', 'user_dow-tz_norm_0_noon', 'useritem_cooccur-min-min_x', 'useritem_order_days_min-mean', 'chance',
't-6_reordered-mean', 'comeback_ratio_sum-std', 'days_since_last_order_this_item-mean', 'useritem_order_days_max-median', 't-7_reordered-mean',
'glutenfree_ratio', 'user_dow_norm_0', 'item_N5_ratio', 'organic_cnt', 'useritem_buy_timezone_cnt-mean',
't-2_repeat_previous_ratio-w1', 't-2_days_since_prior_order', 'user_dow_norm_3', 'useritem_sum_pos_cart_n5', 'user_dow_norm_5',
'user_dow-tz_norm_4_noon', 'user_dow-tz_norm_4_morning', 'useritem_cooccur-max-min-mean', 'useritem_order_days_max_n5-median', 'user_aisle_ratio-median',
'useritem_buy_dow_ratio2-mean', 'item_together_max', 't-2_streak-mean', 'hour_order_cnt', 't-1_order_hour_of_day',
'user_dow-tz_norm_3_noon', 'item_dow_cnt', 'dow_rank_diff', 'timezone_noon', 'useritem_order_days_min_n5',
'user_dow-tz_norm_3_night', 'user_dow-tz_norm_0_morning', 'comeback_ratio_len-mean', 't-2_streak-std', 'last_order_number-min',

# BEST 812_1(201~250)
#'user_dow-tz_norm_5_noon', 'useritem_mean_pos_cart_n5', 't-3_buy_item_inarow-mean', 't-12_reordered', 'useritem_std_pos_cart_n5-median',
#'item_streak_std', 't-7_is_None', 'total_buy_ratio-max', 'useritem_buy_timezone_ratio2-mean', 't-5_reordered',
#'t-3_buy_item_inarow-std', 't-1_repeat_previous_ratio-w2', 't-3_streak-mean', 'useritem_order_days_min_n5-median', 't-10_reordered',
#'t-7_reordered-std', 'item_N4_cnt', 't-1_buy_item_inarow-mean', 'user_dow-tz_norm_5_morning', 'useritem_cooccur-max-max_x',
#'t-6_reordered-std', 't-5_reordered-std', 'useritem_median_pos_cart', 'useritem_buy_dow_cnt', 'useritem_std_pos_cart_n5-mean',
#'user_dow-tz_norm_1_morning', 't-2_buy_item_inarow-std', 'item_streak_max', 'order_number_diff-mean', 't-3_repeat_previous_ratio-w1',
#'item_streak_mean', 'useritem_cooccur-median', 'chance_n5', 't-2_buy_item_inarow-mean', 'total_buy_ratio-median',
#'t-2_order_hour_of_day', 'user_dow-tz_norm_6_noon', 't-5_reordered-mean', 't-3_days_since_prior_order', 't-3_streak-std',
#'t-8_reordered-mean', 'user_dow-tz_freq_5_noon', 'user_dow-tz_norm_1_night', 'useritem_cooccur-max-max-mean', 'useritem_std_pos_cart_n5-min',
#'t-2_product_unq_len_ratioByT-3', 't-2_ordered_item', 'useritem_std_pos_cart', 'user_dow-tz_freq_2_night', 't-1_streak-mean'


    ]    
    return col
    
def load_pred_item(name, keep_all=False):
    
    if keep_all == False:
        #==============================================================================
        print('keep top imp')
        #==============================================================================
        col = keep_top_item()
        if name=='test':
            col.remove('y')
        df = read_pickles('../feature/{}/all'.format(name), col)
        
    else:
        df = read_pickles('../feature/{}/all'.format(name))
    
    print('{}.shape:{}\n'.format(name, df.shape))
    
    return df

def load_pred_item_lowimp(name):
    
    #==============================================================================
    print('with low imp')
    #==============================================================================
    col = ['order_id', 't-1_order_id', 't-2_order_id', 't-3_order_id', 
           'user_id', 'product_id', 'aisle_id', 'department_id', 'is_train', 'y']
    col += [
'useritem_std_pos_cart-median', 'user_dep_cnt-mean', 't-3_buy_item_inarow', 't-2_days_since_prior_order', 'useritem_order_days_mean_n5-median',
'useritem_cooccur-max-max-mean', 'user_aisle_cnt', 'useritem_cooccur-mean', 't-1_reordered-mean', 'useritem_std_pos_cart_n5-mean',
'useritem_cooccur-std-median', 'item_mean_pos_cart', 'useritem_cooccur-mean-std', 'useritem_cooccur-std-std', 'useritem_cooccur-median-mean',
't-10_reordered-std', 'useritem_cooccur-median-std', 't-4_reordered-mean', 'useritem_sum_pos_cart_n5', 'useritem_sum_pos_cart-max',
'item_11to1_chance', 't-2_buy_item_inarow_ratio-mean', 'order_ratio_bychance_n5-mean', 'useritem_order_days_min_n5-min', 'total_buy_ratio-median',
'item_only_one_user_cnt', 'useritem_cooccur-mean-median', 'useritem_sum_pos_cart_n5-std', 't-5_product_unq_len', 'item_101to1_chance',
'item_first_cnt', 'useritem_std_pos_cart-std', 't-2_reordered-mean', 'item_dow_cnt', 't-7_product_unq_len',
'useritem_order_days_min-max', 't-18_is_None', 'useritem_order_days_mean_n5-min', 'order_number_diff-median', 't-8_reordered-std',
'useritem_cooccur-max-max-max', 'days_since_last_order_this_item-mean', 'user_dep_cnt-std', 'useritem_order_days_max-mean', 'useritem_order_days_median-min',
'useritem_min_pos_cart-std', 't-10_reordered', 'useritem_mean_pos_cart-median', 't-3_total_unique_item_ratio', 'useritem_order_days_max_n5-median',
'useritem_order_days_median_n5-mean', 't-8_product_unq_len', 'useritem_cooccur-max-max-median', 'useritem_cooccur-max-min-std', 't-1_reordered_ratio',
't-3_reordered_ratio', 'item_dow_ratio', 'useritem_order_days_min_n5-mean', 'order_ratio_bychance-mean', 't-2_buy_item_inarow_ratio-std',
't-9_reordered-std', 'useritem_min_pos_cart_n5-std', 't-3_repeat_previous_ratio-w1', 't-3_repeat_previous_ratio-w4', 't-1_total_ordered_item_ratio',
'useritem_mean_pos_cart-std', 'useritem_std_pos_cart-mean', 'useritem_cooccur-max-median', 't-3_buy_item_inarow_ratio-max', 'item_order_days_max',
't-2_reordered_ratio', 't-2_total_unique_item', 't-11_reordered-std', 'item_dow-tz_cnt', 'user_aisle_cnt-max',
'useritem_cooccur-max-std', 'order_hour_of_day_18', 't-1_buy_item_inarow_ratio-mean', 'user_dow_freq_4', 'item_dow-tz_cnt_uniq',
't-14_reordered-mean', 'user_timezone_norm_night', 'organic_ratio', 't-15_reordered-std', 'user_dow_freq_6',
'useritem_order_days_min_n5-max', 'useritem_cooccur-min-min-min', 't-5_reordered-std', 't-2_buy_item_inarow-max', 'user_dow_norm_5',
'days_since_last_order_this_item-median', 't-13_reordered-mean', 'useritem_mean_pos_cart_n5-median', 't-2_buy_item_inarow_ratio-max', 'useritem_cooccur-std-max',
't-3_repeat_previous_ratio-w2', 't-2_days_since_first_order', 'useritem_cooccur-std-mean', 'useritem_std_pos_cart-max', 'useritem_order_days_mean-mean',            
            ]
    if name=='test':
        col.remove('y')
    df = read_pickles('../feature/{}/all'.format(name), col)
        
    
    print('{}.shape:{}\n'.format(name, df.shape))
    
    return df

def keep_top_None(W):
    
    col = ['order_id', 't-1_order_id', 't-2_order_id', 't-3_order_id', 
           'user_id', 'is_train', 'y']
    
    if W==3:
        # top 300
        col += [
'useritem_sum_pos_cart-mean', 'days_since_prior_order', 'total_buy_n5-max', 't-1_is_None', 't-1_department_5',
'total_buy-mean', 'total_buy-max', 'order_ratio_bychance-max', 't-1_reordered_ratio', 'before_to_after_ratio-mean',
'total_buy_ratio_n5-max', 't-3_department_5', 'total_buy_ratio_n5-std', 't-2_buy_item_inarow-max', 't-1_days_since_prior_order',
'useritem_sum_pos_cart_n5-mean', 'useritem_cooccur-mean-min', 'total_buy_ratio-max', 'days_since_last_order_this_item-median', 'user_dow-tz_norm_6_midnight',
't-2_department_12', 'useritem_std_pos_cart_n5-std', 'useritem_cooccur-max-std', 't-1_department_17', 't-1_buy_item_inarow-max',
'order_dow-std', 't-2_total_ordered_item', 'before_to_after_ratio-median', 't-1_reordered-std', 'total_buy-std',
'item_order_days_mean-min', 'item_onb_diff_skew-std', 'useritem_buy_timezone_cnt-mean', 'item_flat_dow-tz_0_midnight-mean', 'days_order_mean',
'useritem_order_days_median_n5-min', 'item_onb_diff_max-std', 'user_order_size-std', 'useritem_cooccur-max-min', 'item_median_pos_cart-std',
'before_to_after_ratio-std', 'useritem_mean_pos_cart-min', 'useritem_cooccur-mean-std', 'useritem_order_days_mean_n5-min', 'item_flat_dow-tz_1_midnight-mean',
'item_onb_diff_skew-median', 'item_is_Asian-mean', 'useritem_sum_pos_cart_n5-max', 'useritem_cooccur-max-min-mean', 'item_max_pos_cart-std',
't-1_buy_item_inarow-std', 'item_flat_dow-tz_2_midnight-mean', 'item_flat_dow-tz_5_midnight-mean', 'user_dow-tz_norm_3_morning', 'item_flat_dow-tz_6_morning-median',
'order_ratio_bychance_n5-std', 'item_flat_dow-tz_4_midnight-mean', 'item_onb_diff_mean-median', 'item_dow_ratio_unq-mean', 't-2_repeat_previous_ratio-w2',
'useritem_sum_pos_cart-max', 't-2_streak-std', 'useritem_buy_timezone_cnt-std', 'useritem_buy_timezone_ratio2-std', 'user_dow-tz_freq_0_noon',
'item_dow_ratio-median', 't-2_days_since_prior_order', 'order_ratio_bychance_n5-mean', 'useritem_sum_pos_cart-min', 'useritem_order_days_max-max',
'order_ratio_bychance-std', 'item_onb_diff_skew-max', 'before_to_after_ratio-max', 't-1_total_ordered_item', 'useritem_order_days_min-mean',
'order_dow-mean', 'item_onb_diff_skew-mean', 'user_dep_cnt-mean', 'item_flat_dow-tz_1_noon-mean', 'item_max_pos_cart-min',
'useritem_buy_dow_ratio2-std', 'item_flat_dow-tz_3_midnight-mean', 't-3_is_None', 'item_flat_dow-tz_6_midnight-median', 'item_flat_dow-tz_3_night-mean',
'item_flat_dow-tz_1_noon-median', 'useritem_order_days_median_n5-max', 'useritem_order_days_max_n5-min', 'item_std_pos_cart-min', 'item_onb_diff_std-std',
'item_streak_std-max', 'item_flat_dow-tz_4_midnight-median', 'item_flat_dow-tz_0_midnight-median', 't-3_reordered-std', 'item_flat_dow-tz_3_morning-median',
'useritem_order_days_min-median', 'item_onb_diff_mean-mean', 'item_flat_dow-tz_1_midnight-median', 'item_flat_dow-tz_1_night-mean', 'item_100to1_ratio-std',
'useritem_cooccur-min-min-mean', 'item_onb_diff_std-median', 'item_mean_pos_cart-std', 't-3_department_16', 'useritem_cooccur-std-std',
'days_since_last_order_this_item-mean', 't-2_reordered_ratio', 't-1_repeat_previous_ratio-w1', 'item_std_pos_cart-std', 'user_aisle_cnt-mean',
'useritem_sum_pos_cart-median', 'item_flat_dow-tz_3_midnight-median', 'hyb_BoO-Bananas', 'user_dow_norm_1', 'item_timezone_ratio_uniq-median',
'item_together_max-std', 'item_flat_dow-tz_2_noon-mean', 'item_only_one_user_cnt_ratio-std', 'item_first_ratio-max', 'total_buy_ratio-std',
'user_aisle_ratio-std', 'user_dep_ratio-std', 'item_dow_ratio_unq-median', 'user_dow_norm_3', 'user_timezone_freq_noon',
'useritem_cooccur-max-min-std', 'item_onb_diff_mean-std', 'useritem_sum_pos_cart-std', 't-10_reordered-median', 'item_flat_dow-tz_3_morning-mean',
'item_flat_dow-tz_4_midnight-max', 'item_flat_dow-tz_2_noon-median', 'item_streak_max-max', 'item_flat_dow-tz_6_midnight-mean', 'item_onb_diff_std-max',
't-1_repeat_previous_ratio-w2', 'item_onb_diff_std-mean', 'item_flat_dow-tz_1_night-median', 'item_flat_dow-tz_5_noon-median', 'before_to_after_ratio-min',
'item_dow_ratio-max', 't-3_streak-std', 'item_onb_diff_min-mean', 'user_dep_ratio-max', 'item_flat_dow-tz_2_night-mean',
'item_flat_dow-tz_4_night-median', 'item_101to1_ratio-max', 'item_100to1_ratio-median', 'item_flat_dow-tz_2_midnight-max', 't-3_department_19',
'item_dow_ratio_diff-median', 'user_dep_ratio-median', 'item_flat_dow-tz_0_midnight-max', 'item_onb_diff_median-mean', 'item_dow_ratio_unq-max',
'item_together_std-max', 'useritem_std_pos_cart-std', 'total_buy_n5-mean', 'user_aisle_ratio-max', 'delta_hour_t-1',
'item_1to1_cnt-max', 'item_flat_dow-tz_2_midnight-median', 'item_flat_dow-tz_1_morning-max', 'item_flat_dow-tz_6_morning-mean', 'item_flat_dow-tz_1_noon-max',
'item_110to1_ratio-max', 'item_flat_dow-tz_3_noon-mean', 'useritem_order_days_min-max', 'item_together_mean-std', 'delta_hour_t-2',
'item_111to1_ratio-max', 'useritem_buy_timezone_ratio-std', 'item_10to1_ratio-median', 'item_flat_dow-tz_1_midnight-max', 'item_dow_ratio_diff-min',
't-7_is_None', 'chance_n5-std', 'item_streak_max-std', 'item_flat_dow-tz_3_midnight-max', 'user_dow-tz_norm_3_midnight',
't-2_reordered-std', 'days_since_first_order', 'useritem_order_days_median-max', 'item_dow_ratio_unq-min', 'item_flat_dow-tz_1_midnight-std',
'useritem_order_days_mean-max', 'item_order_days_max-max', 'item_flat_dow-tz_2_night-median', 'item_dow_ratio-min', 'item_flat_dow-tz_3_morning-max',
'item_reorder_ratio-max', 'item_101to1_ratio-std', 'useritem_cooccur-min-std', 'item_flat_dow-tz_5_midnight-median', 'item_together_mean-min',
'useritem_order_days_min_n5-min', 'item_onb_diff_median-std', 't-2_reordered-mean', 'useritem_sum_pos_cart_n5-min', 't-4_reordered-std',
'item_dow_ratio-mean', 'item_100to1_ratio-max', 't-1_repeat_previous_ratio-w3', 'item_flat_dow-tz_4_noon-mean', 'useritem_cooccur-median-min',
'item_onb_diff_max-mean', 'item_together_std-min', 'item_streak_mean-max', 'item_together_mean-max', 't-1_department_4_cumsum',
'useritem_cooccur-std-mean', 'item_together_std-median', 'item_flat_dow-tz_0_midnight-std', 't-2_repeat_previous_ratio-w1', 'item_hour_ratio-min',
't-3_reordered-mean', 'delta_hour_t-3', 'item_flat_dow-tz_3_noon-median', 'item_10to1_ratio-max', 'user_timezone_norm_noon',
'useritem_sum_pos_cart_n5-median', 'item_110to1_ratio-std', 'comeback_ratio_max-max', 'item_flat_dow-tz_4_night-mean', 'item_flat_dow-tz_0_night-max',
'item_flat_dow-tz_6_midnight-max', 'item_mean_pos_cart-min', 'item_flat_dow-tz_5_midnight-max', 'item_onb_diff_max-median', 'item_flat_dow-tz_3_midnight-std',
'item_flat_dow-tz_6_night-median', 'item_flat_dow-tz_3_night-median', 't-2_buy_item_inarow-std', 'item_std_pos_cart-max', 'useritem_order_days_max-median',
'item_flat_dow-tz_3_night-max', 'item_flat_dow-tz_2_morning-mean', 'useritem_mean_pos_cart-std', 'item_flat_dow-tz_6_midnight-std', 'chance_n5-mean',
'item_110to1_ratio-median', 'user_dow-tz_norm_1_morning', 'user_dow_norm_5', 'item_flat_dow-tz_2_midnight-min', 'item_only_one_user_cnt_ratio-mean',
'item_timezone_ratio-median', 't-1_department_16', 'useritem_cooccur-median-std', 'item_only_one_user_cnt_ratio-median', 'useritem_order_days_mean-min',
'item_together_std-mean', 'item_flat_dow-tz_5_night-median', 'hyb_Organic-Baby-Spinach', 'item_flat_dow-tz_4_night-max', 'item_flat_dow-tz_1_morning-mean',
'item_max_pos_cart-median', 'item_flat_dow-tz_3_noon-max', 't-1_reordered-mean', 't-2_repeat_previous_ratio-w4', 'item_streak_std-std',
'item_dow_ratio-std', 'item_max_pos_cart-max', 'item_reorder_ratio-std', 'item_flat_dow-tz_5_night-mean', 'item_dow-tz_ratio-min',
'user_timezone_norm_morning', 'item_together_max-min', 'item_flat_dow-tz_4_morning-mean', 'item_flat_dow-tz_2_midnight-std', 'user_dow_freq_0',
'item_11to1_ratio-max', 'item_dow-tz_ratio-std', 'item_flat_dow-tz_0_morning-mean', 'dow_item_cnt', 't-4_reordered-mean',
'item_dow_ratio_diff-mean', 'item_flat_dow-tz_1_morning-std', 'useritem_std_pos_cart-median', 'item_flat_dow-tz_5_noon-mean', 'item_flat_dow-tz_3_morning-std',
't-1_order_hour_of_day', 'item_flat_dow-tz_4_morning-max', 'user_dow-tz_freq_6_midnight', 'item_onb_diff_max-max', 'item_flat_dow-tz_6_night-mean',
'item_flat_dow-tz_6_night-max', 'item_flat_dow-tz_5_morning-median', 'item_101to1_ratio-median', 'item_onb_diff_mean-max', 'item_flat_dow-tz_1_night-max',
'item_flat_dow-tz_0_morning-median', 'item_flat_dow-tz_4_midnight-std', 'item_timezone_ratio_uniq-std', 'item_onb_diff_skew-min', 'item_flat_dow-tz_5_morning-mean',
'item_together_std-std', 'useritem_cooccur-std-min', 'item_dow_ratio_diff-max', 't-5_reordered-std', 'useritem_std_pos_cart_n5-mean'

# BEST (808_1)
#'useritem_sum_pos_cart-mean', 'useritem_sum_pos_cart-median', 'useritem_cooccur-max-max-max', 'useritem_cooccur-mean-min', 'item_11to1_cnt-mean',
#'useritem_cooccur-median-min', 'item_N3_chance-median', 'item_first_chance-median', 'useritem_cooccur-max-min', 'useritem_sum_pos_cart_n5-min',
#'t-2_product_unq_len', 'order_number', 'useritem_sum_pos_cart-min', 't-1_is_None', 'useritem_mean_pos_cart-min',
#'useritem_order_days_min-min', 't-1_product_unq_len', 'item_first_cnt-median', 'useritem_cooccur-max-std', 'chance-max',
#'days_since_prior_order', 'useritem_std_pos_cart_n5-min', 'useritem_buy_timezone_cnt-mean', 't-3_product_unq_len', 'item_order_per-user-median',
#'item_N5_chance-median', 'useritem_cooccur-max-min-mean', 'useritem_buy_timezone_cnt-std', 't-2_is_None', 'before_to_after_ratio-mean',
#'item_dow_cnt-mean', 't-13_product_unq_len', 'item_sum_pos_cart-mean', 't-7_product_unq_len', 'useritem_buy_timezone_ratio-std',
#'useritem_cooccur-mean-std', 't-11_product_unq_len', 't-2_total_ordered_item', 'item_reorder_ratio-max', 'user_aisle_ratio-min',
#'t-3_days_since_first_order', 't-1_total_ordered_item', 'useritem_buy_timezone_cnt-max', 'chance-std', 'useritem_cooccur-median-max',
#'useritem_order_days_median-min', 'useritem_cooccur-mean-max', 'useritem_order_days_min_n5-min', 't-3_reordered_ratio', 't-19_reordered-std',
#'useritem_min_pos_cart-max', 'useritem_order_days_median-median', 'useritem_cooccur-max-max-mean', 't-3_is_None', 'user_aisle_ratio-mean',
#'user_order_size-min', 'item_only_one_user_cnt-std', 't-6_product_unq_len', 't-2_total_unique_item', 'days_since_last_order_this_item-min',
#'t-4_product_unq_len', 'chance_n5-min', 't-5_product_unq_len', 'item_1to1_ratio-max', 't-2_reordered_ratio',
#'item_10to1_ratio-max', 'useritem_std_pos_cart_n5-std', 'before_to_after_ratio-median', 'total_buy_n5-median', 't-1_reordered_ratio',
#'t-3_total_unique_item', 'item_10to1_ratio-mean', 'useritem_buy_timezone_ratio-min', 'order_ratio_bychance-mean', 't-9_product_unq_len',
#'useritem_buy_timezone_ratio-mean', 'useritem_max_pos_cart-std', 't-1_total_unique_item', 'item_order_per-user-mean', 't-17_product_unq_len',
#'item_streak_max-min', 'item_hour_cnt_unq-mean', 'useritem_buy_timezone_ratio2-max', 'item_first_chance-max', 't-15_product_unq_len',
#'t-4_is_None', 'total_buy_n5-max', 'item_hour_cnt_unq-std', 'useritem_median_pos_cart_n5-max', 'useritem_buy_timezone_ratio2-min',
#'order_ratio_bychance-min', 't-3_department_21', 'useritem_cooccur-max-mean', 'useritem_order_days_max-median', 'useritem_cooccur-max-min-max',
#'useritem_buy_timezone_ratio2-median', 'useritem_cooccur-mean-mean', 't-8_product_unq_len', 'useritem_buy_timezone_ratio2-mean', 'item_max_pos_cart-median',
#'t-1_department_2', 'useritem_buy_dow_ratio2-max', 'useritem_cooccur-max-min-std', 'item_flat_dow-tz_5_midnight-std', 'useritem_order_days_max_n5-max',
#'t-1_repeat_previous_ratio-w3', 'order_ratio_bychance-median', 't-2_repeat_previous_ratio-w4', 't-3_repeat_previous_ratio-w5', 't-3_total_ordered_item',
#'useritem_cooccur-median-std', 't-19_product_unq_len', 'chance-median', 'item_order_days_median-mean', 'user_dow-tz_norm_5_midnight',
#'item_1to1_chance-median', 'dow_item_cnt', 'item_N5_cnt-max', 'item_unique_user-mean', 't-1_product_unq_len_ratioByT-3',
#'t-14_product_unq_len', 'item_hour_cnt_unq-min', 'item_max_pos_cart-max', 'item_std_pos_cart-min', 'useritem_order_days_max-max',
#'item_order_freq-min', 'item_first_ratio-max', 'days_order_mean', 'useritem_order_days_median_n5-max', 'useritem_max_pos_cart_n5-median',
#'item_first_ratio-mean', 't-10_product_unq_len', 'user_order_size-std', 't-5_is_None', 'useritem_order_days_median_n5-std',
#'useritem_order_days_mean-median', 't-1_repeat_previous_ratio-w5', 'item_10to1_ratio-median', 'useritem_order_days_max-min', 'useritem_order_days_min-median',
#'t-2_product_unq_len_diffByT-3', 'T', 'useritem_buy_timezone_ratio-median', 'useritem_order_days_median_n5-min', 't-2_department_2',
#'item_reorder_ratio-min', 'useritem_sum_pos_cart_n5-mean', 'useritem_buy_dow_ratio2-mean', 't-1_repeat_previous_ratio-w1', 'useritem_max_pos_cart_n5-max',
#'useritem_order_days_min_n5-std', 't-1_days_since_prior_order', 'chance-mean', 'useritem_order_days_median-mean', 't-18_product_unq_len',
#'useritem_order_days_max-mean', 'item_unique_user-min', 'useritem_order_days_mean-min', 'order_dow_5', 't-12_product_unq_len',
#'useritem_sum_pos_cart-max', 'useritem_sum_pos_cart-std', 'useritem_cooccur-min-mean', 'item_only_one_user_cnt_ratio-min', 't-1_repeat_previous_ratio-w2',
#'item_only_one_user_cnt_ratio-median', 'item_order_days_median-median', 'order_hour_of_day_4', 'useritem_order_days_max_n5-median', 't-2_department_21',
#'item_N2_cnt-mean', 't-16_product_unq_len', 'item_10to1_cnt-max', 'useritem_order_days_median_n5-median', 'useritem_cooccur-std-mean',
#'t-1_ordered_item', 'user_dow_norm_0', 'item_only_one_user_cnt-mean', 'item_N2_chance-min', 'timezone_night',
#'t-1_product_unq_len_ratioByT-2', 'item_1to1_chance-min', 'useritem_cooccur-std-max', 't-2_total_ordered_item_ratio', 'item_streak_max-mean',
#'useritem_max_pos_cart_n5-std', 'user_dow-tz_freq_6_noon', 'useritem_mean_pos_cart-median', 'useritem_cooccur-std-std', 'useritem_order_days_mean-mean',
#'useritem_median_pos_cart-std', 'item_flat_dow-tz_2_midnight-min', 'item_flat_dow-tz_6_midnight-mean', 'useritem_cooccur-median-mean', 'chance-min',
#'item_only_one_user_cnt-max', 't-2_days_since_first_order', 'useritem_order_days_min-std', 't-3_department_8', 'days_since_first_order',
#'item_sum_pos_cart-max', 'item_dow_cnt-max', 'total_buy_ratio_n5-max', 't-1_total_ordered_item_ratio', 't-3_repeat_previous_ratio-w1',
#'item_streak_mean-max', 'before_to_after_ratio-std', 'useritem_buy_timezone_ratio2-std', 'useritem_order_days_min-max', 't-9_is_None',
#'t-11_is_None', 'user_aisle_cnt-max', 't-2_repeat_previous_ratio-w3', 't-2_repeat_previous_ratio-w2', 'user_dow_freq_1',
#'useritem_order_days_max_n5-std', 'useritem_cooccur-min-std', 'timezone_noon', 'chance_n5-std', 'user_dow-tz_freq_4_midnight',
#'user_dep_ratio-min', 'item_10to1_ratio-min', 't-1_repeat_previous_ratio-w4', 'user_aisle_cnt-mean', 'user_timezone_freq_morning',
#'user_order_size-max', 'item_11to1_ratio-median', 'useritem_order_days_min_n5-max', 't-1_total_unique_item_ratio', 'useritem_mean_pos_cart-max',
#'organic_cnt', 'item_N3_ratio-median', 't-1_department_8', 't-3_department_10', 'order_hour_of_day_22',
#'item_flat_dow-tz_6_midnight-min', 'item_11to1_chance-median', 't-2_repeat_previous_ratio-w5', 't-6_is_None', 'item_sum_pos_cart-min',
#'item_order_days_mean-median', 'user_dow_norm_6', 'user_dow-tz_freq_5_morning', 'useritem_min_pos_cart-median', 't-3_repeat_previous_ratio-w3',
#'t-3_repeat_previous_ratio-w2', 'item_together_min-max', 'user_dow-tz_freq_1_midnight', 't-2_days_since_prior_order', 't-12_reordered-std',
#'user_dow-tz_norm_6_midnight', 'user_dow_norm_1', 'user_dow-tz_freq_1_morning', 't-1_days_since_first_order', 'user_dep_cnt-max',
#'t-2_department_5', 'order_hour_of_day_18', 'useritem_order_days_mean_n5-mean', 'user_dep_cnt-mean', 'useritem_order_days_median-std',
#'user_dow_freq_3', 'useritem_buy_timezone_cnt-median', 'item_reorderd_freq-std', 't-3_total_ordered_item_ratio', 't-3_repeat_previous_ratio-w4',
#'useritem_order_days_min-mean', 'user_dow_freq_0', 't-2_repeat_previous_ratio-w1', 'order_ratio_bychance_n5-min', 'item_flat_dow-tz_0_noon-mean',
#'t-2_department_4', 'user_dow-tz_freq_3_morning', 'item_N4_ratio-min', 'order_hour_of_day_21', 'item_order_freq-median',
#'useritem_median_pos_cart_n5-mean', 'hyb_Organic-Hass-Avocado', 'useritem_cooccur-std-median', 'useritem_median_pos_cart_n5-std', 't-1_department_16',
#'useritem_sum_pos_cart_n5-median', 'useritem_median_pos_cart-max', 'item_order_per-user-std', 't-17_reordered-mean', 't-3_department_18',
#'item_flat_dow-tz_5_night-min', 'item_N3_cnt-mean', 'user_order_size-mean', 't-2_total_unique_item_ratio', 'user_dow-tz_freq_5_noon',
#'t-2_department_11', 'user_dow-tz_freq_6_night', 'total_buy_ratio-min', 'item_flat_dow-tz_5_morning-min', 'useritem_std_pos_cart-median',
#'user_dow-tz_freq_0_noon', 'useritem_order_days_max-std', 'item_hour_ratio_unq-min', 'useritem_buy_dow_ratio2-min', 'user_dow-tz_norm_1_morning'                
                ]
    elif W==5:
        raise
        col += [
                ]
    return col

def load_pred_None(name, W, keep_all=False):
    print('window size:{}'.format(W))
    
    if keep_all == False:
        #==============================================================================
        print('keep top imp')
        #==============================================================================
        col = keep_top_None(W)
        if name=='test':
            col.remove('y')
        df = read_pickles('../feature/{}/all_None_w{}'.format(name, W), col)
    else:
        df = read_pickles('../feature/{}/all_None_w{}'.format(name, W))
    
    print('{}.shape:{}\n'.format(name, df.shape))
    
    return df

def load_pred_None_lowimp(name):
    
    #==============================================================================
    print('with low imp')
    #==============================================================================
    col = ['order_id', 't-1_order_id', 't-2_order_id', 't-3_order_id', 
           'user_id', 'is_train', 'y']
    col += [
'user_dow_norm_1', 'user_dow-tz_norm_6_noon', 'user_timezone_freq_midnight', 'user_dow-tz_freq_4_morning', 'user_aisle_ratio-std',
'user_dow-tz_norm_2_morning', 'item_flat_dow-tz_4_midnight-max', 'useritem_cooccur-std-mean', 'order_hour_of_day_5', 'user_dow-tz_norm_4_night',
'order_hour_of_day_6', 't-6_reordered-mean', 'user_dow_freq_2', 'user_dow-tz_norm_1_noon', 'user_dow-tz_freq_2_morning',
't-3_buy_item_inarow_ratio-max', 'item_first_chance-min', 't-2_department_14', 'order_dow_5', 'item_hour_ratio_unq-max',
'chance_n5-median', 'item_together_std-std', 'item_first_chance-mean', 'user_dow-tz_norm_2_night', 'useritem_min_pos_cart_n5-mean',
'user_dow_norm_5', 'user_dow-tz_norm_0_night', 'order_ratio_bychance-std', 'useritem_std_pos_cart_n5-median', 'user_dow-tz_freq_0_midnight',
'user_dep_cnt-max', 'item_flat_dow-tz_0_morning-mean', 'order_hour_of_day_23', 't-1_department_14', 'user_aisle_cnt-std',
'useritem_std_pos_cart_n5-mean', 'item_flat_dow-tz_5_morning-mean', 'useritem_order_days_mean_n5-min', 'user_dow-tz_freq_4_night', 'item_together_min-max',
'user_dow-tz_norm_6_midnight', 'user_timezone_freq_night', 't-2_department_20', 'user_dow-tz_norm_3_night', 'item_dow_ratio_diff-max',
't-13_reordered-std', 'item_timezone_ratio_uniq-max', 'user_dow-tz_norm_3_morning', 't-12_is_None', 'glutenfree_ratio',
't-9_is_None', 't-1_department_8', 'useritem_mean_pos_cart-mean', 'user_dow-tz_norm_5_noon', 't-2_total_unique_item_ratio',
'item_flat_dow-tz_4_morning-median', 'item_order_days_min-mean', 'item_flat_dow-tz_2_morning-std', 't-5_reordered-std', 'item_order_days_max-std',
'user_aisle_cnt-max', 'useritem_mean_pos_cart-std', 'delta_hour_t-1', 'order_hour_of_day_12', 'item_flat_dow-tz_3_noon-mean',
'user_timezone_norm_night', 'hour_order_cnt', 't-3_streak-mean', 't-3_department_12', 'order_hour_of_day_14',
'useritem_order_days_mean_n5-max', 'useritem_order_days_max_n5-min', 'item_flat_dow-tz_2_morning-median', 'user_dow-tz_norm_5_morning', 't-15_reordered-mean',
'item_dow-tz_ratio-median', 'item_order_per-user-median', 'user_dow-tz_norm_6_night', 'item_N3_ratio-mean', 'glutenfree_cnt',
'item_flat_dow-tz_6_morning-median', 'item_timezone_ratio_uniq-mean', 'user_dow-tz_freq_1_night', 't-7_is_None', 'user_dow-tz_freq_5_night',
'useritem_order_days_min_n5-mean', 't-11_reordered-mean', 'before_to_after_ratio-min', 'useritem_std_pos_cart-mean', 'user_dep_cnt-median',
't-1_department_6', 'user_timezone_freq_noon', 'user_dow-tz_freq_6_night', 't-12_reordered-std', 'user_timezone_norm_morning',
'useritem_cooccur-mean-median', 'useritem_median_pos_cart_n5-median', 'user_dow_norm_4', 'item_streak_max-max', 'useritem_std_pos_cart_n5-max'
            ]
    if name=='test':
        col.remove('y')
    df = read_pickles('../feature/{}/all_None_w{}'.format(name, 3), col)
    
    print('{}.shape:{}\n'.format(name, df.shape))
    
    return df

#==============================================================================
# main
#==============================================================================
if __name__ == "__main__":
    pass

