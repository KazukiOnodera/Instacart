#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 22:36:10 2017

@author: konodera


"""

import pandas as pd
import gc
import numpy as np
from collections import defaultdict
from scipy.stats import skew
import utils
utils.start(__file__)

#==============================================================================
# load
#==============================================================================

col = ['product_id', 'user_id', 'order_number', 'order_number_rev']
log = utils.read_pickles('../input/mk/log', col).sort_values(col[:3])

"""
1 1 1
1 1 2
1 1 4
1 2 3
1 2 4
2 2 5
"""
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
    
    log_ = log[log.order_number_rev>T]
    log_['user_max_onb'] = log_.groupby('user_id').order_number.transform(np.max)
    
    item_min = defaultdict(int)
    item_mean = defaultdict(int)
    item_median = defaultdict(int)
    item_max = defaultdict(int)
    item_std = defaultdict(int)
    item_skew = defaultdict(int)
    
    pid_bk = uid_bk = onb_bk = None
    diff = []
    
    for pid, uid, onb, max_onb in log_[['product_id', 'user_id', 'order_number', 'user_max_onb']].values:
        
        if pid==pid_bk and uid==uid_bk:
            diff.append(onb-onb_bk)
            """
            pattern would be like:
                     onb     ->  diff 
            1111    1,2,3,4  ->  [1,1,1]
            11101   1,2,3,5  ->  [1,1,2]
            111     1,2,3    ->  [1,1]
            1101    1,2,4    ->  [1,2]
            1011    1,3,4    ->  [2,1]
            """
        
        elif pid==pid_bk and uid!=uid_bk:
            pass
        elif pid!=pid_bk:
            if len(diff)>0:
                item_min[pid]    = np.min(diff)
                item_mean[pid]   = np.mean(diff)
                item_median[pid] = np.median(diff)
                item_max[pid]    = np.max(diff)
                item_std[pid]    = np.std(diff)
                item_skew[pid]   = skew(diff)
            diff = []
        
        pid_bk = pid
        uid_bk = uid
        onb_bk = onb
    
    item_min = pd.DataFrame.from_dict(item_min, orient='index').reset_index()
    item_min.columns = ['product_id', 'item_onb_diff_min']
    item_mean = pd.DataFrame.from_dict(item_mean, orient='index').reset_index()
    item_mean.columns = ['product_id', 'item_onb_diff_mean']
    item_median = pd.DataFrame.from_dict(item_median, orient='index').reset_index()
    item_median.columns = ['product_id', 'item_onb_diff_median']
    item_max = pd.DataFrame.from_dict(item_max, orient='index').reset_index()
    item_max.columns = ['product_id', 'item_onb_diff_max']
    item_std = pd.DataFrame.from_dict(item_std, orient='index').reset_index()
    item_std.columns = ['product_id', 'item_onb_diff_std']
    item_skew = pd.DataFrame.from_dict(item_skew, orient='index').reset_index()
    item_skew.columns = ['product_id', 'item_onb_diff_skew']
    
    df1 = pd.merge(item_min,   item_mean, on='product_id', how='outer')
    df2 = pd.merge(item_median, item_max, on='product_id', how='outer')
    df3 = pd.merge(item_std, item_skew, on='product_id', how='outer')
    
    df = pd.merge(pd.merge(df1, df2, on='product_id', how='outer'), 
                  df3, on='product_id', how='outer')
    
    df.fillna(-99, inplace=True)
    df.to_pickle('../feature/{}/f215_product.p'.format(folder))
    

#==============================================================================
# main
#==============================================================================
make(0)
make(1)
make(2)

make(-1)


#==============================================================================
utils.end(__file__)

