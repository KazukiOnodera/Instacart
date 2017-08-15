#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 07:41:26 2017

@author: konodera

そのユーザがそのアイテム注文したのは何日前か？
*リークじゃない

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
import multiprocessing as mp
import utils
utils.start(__file__)


kfold = 10

X_base = pd.read_pickle('../feature/X_base_t3.p')

label_train = pd.read_pickle('../feature/trainT-0/label_reordered.p')
label_test  = pd.read_pickle('../feature/test/label_reordered.p')

train = pd.merge(X_base[X_base.is_train==1], label_train, on='order_id', how='inner')
test = pd.merge(X_base[X_base.is_train==0], label_test, on='order_id', how='inner')

#==============================================================================
# mk train * test log
#==============================================================================
col = ['order_id', 'user_id', 'product_id']
train_log = utils.read_pickles('../input/mk/log', col)

order_tbl = pd.read_pickle('../input/mk/order_tbl.p')\
            [['order_id', 'user_id', 'order_number', 'days_since_first_order']]

# merge user_id -> ['order_id', 'user_id', 'product_id']
train_log = pd.merge(train_log[['order_id', 'product_id']], 
                     order_tbl[['order_id','user_id']], 
                     on='order_id', how='left')[['order_id', 'user_id', 'product_id']]
test_log  = pd.merge(test[['order_id', 'product_id']], 
                     order_tbl[['order_id','user_id']], 
                     on='order_id', how='left')[['order_id', 'user_id', 'product_id']]

log = pd.concat([train_log, test_log])
del X_base, train_log, test_log; gc.collect()
log.sort_values(['user_id', 'product_id'], inplace=True)

user_item = log.drop_duplicates(['user_id', 'product_id'])[['user_id', 'product_id']]
order_user = order_tbl[['order_id', 'user_id',]]

log = pd.merge(order_user, user_item, on='user_id', how='left')
del order_user, user_item; gc.collect()

users = log[['user_id']].drop_duplicates().reset_index(drop=1)
users['kfold'] = users.index%kfold


usecols = [ 'order_id', 'product_id']
buy_tbl = utils.read_pickles('../input/mk/log', usecols)
buy_tbl['key'] = buy_tbl.order_id.map(str) + ' ' + buy_tbl.product_id.map(str)

utils.mkdir_p('../input/mk/days_since_last_order')


#==============================================================================
# days_since_last_order_this_item
#==============================================================================
def multi(i):
    target_users = users[users.kfold==i].user_id
    
    tbl = pd.merge(log[log.user_id.isin(target_users)], 
                    order_tbl[['order_id','order_number', 'days_since_first_order']], 
                    on='order_id', how='left')
    
    tbl.sort_values(['user_id', 'product_id', 'order_number'], inplace=True)
    
    
    tbl['key'] = tbl.order_id.map(str) + ' ' + tbl.product_id.map(str)
    tbl['buy'] = tbl.key.isin(buy_tbl.key)*1
    
    tbl.days_since_first_order = tbl.days_since_first_order.fillna(0)
    
    tbl.sort_values(['user_id', 'product_id', 'order_number'], inplace=True)
    
    tbl.reset_index(drop=1, inplace=True)
    
    uid_bk = pid_bk = day_bk = last_date = None
    first_buy = False
    ret = []
    miniters = int(tbl.shape[0]/50)
    for uid,pid,day,buy in tqdm(tbl[['user_id', 'product_id','days_since_first_order','buy']].values, 
                                miniters=miniters):
        if uid_bk is None:
            if buy==1 and first_buy is False:
                ret.append(None)
                last_date = day
                first_buy = True
            elif buy==1:
                ret.append(day-last_date)
                last_date = day
            elif buy==0 and first_buy is True:
                ret.append(day-last_date)
            else:
                ret.append(None)
                
        elif uid == uid_bk and pid == pid_bk:
            if buy==1 and first_buy is False:
                ret.append(None)
                last_date = day
                first_buy = True
            elif buy==1:
                ret.append(day-last_date)
                last_date = day
            elif buy==0 and first_buy is True:
                ret.append(day-last_date)
            else:
                ret.append(None)
                
        elif uid == uid_bk and pid != pid_bk: # item change
            last_date = None
            first_buy = False
            if buy==1 and first_buy is False:
                ret.append(None)
                last_date = day
                first_buy = True
            elif buy==1:
                ret.append(day-last_date)
                last_date = day
            elif buy==0 and first_buy is True:
                ret.append(day-last_date)
            else:
                ret.append(None)
                
        elif uid != uid_bk: # user change
            last_date = None
            first_buy = False
            if buy==1 and first_buy is False:
                ret.append(None)
                last_date = day
                first_buy = True
            elif buy==1:
                ret.append(day-last_date)
                last_date = day
            elif buy==0 and first_buy is True:
                ret.append(day-last_date)
            else:
                ret.append(None)
        uid_bk = uid
        pid_bk = pid
        day_bk = day
    tbl['days_since_last_order_this_item'] = ret
    
    col = ['order_id', 'product_id','days_since_last_order_this_item']
    tbl[col].to_pickle('../input/mk/days_since_last_order/{}.p'.format(i))
    
#==============================================================================



mp_pool = mp.Pool(kfold)
mp_pool.map(multi, range(kfold))











utils.end(__file__)

