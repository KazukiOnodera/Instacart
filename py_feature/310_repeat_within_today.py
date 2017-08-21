#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 15:58:38 2017

@author: konodera

同じアイテムを同日に買ったことがあるか


"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import utils
utils.start(__file__)

#==============================================================================
# load
#==============================================================================
col = ['order_id', 'user_id', 'product_id', 'order_number','days_since_prior_order', 'order_number_rev']
log = utils.read_pickles('../input/mk/log', col).sort_values(['user_id', 'product_id', 'order_number'])
log.user_id    = log.user_id.map(str)
log.product_id = log.product_id.map(str)


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
    
    uid_pid = {}
    uid_bk = pid_bk = onb_bk = None
    col = ['user_id', 'product_id', 'order_number', 'days_since_prior_order']
    
    for uid,pid,onb,days in log_[col].values:
    #    uid = str(uid)
    #    pid = str(pid)
        if uid_bk is None:
            pass
        elif uid+'@'+pid in uid_pid:
            continue
        elif days == 0 and uid == uid_bk and pid == pid_bk and onb-onb_bk==1:
            uid_pid[uid+'@'+pid] = 1
            
        uid_bk = uid
        pid_bk = pid
        onb_bk = onb
    
    df = pd.DataFrame().from_dict(uid_pid, orient='index').reset_index()
    df.columns = ['uidpid', 'buy_within_sameday']
    df['user_id'] = df.uidpid.map(lambda x:x.split('@')[0])
    df['product_id'] = df.uidpid.map(lambda x:x.split('@')[1])
    
    df = df[['user_id', 'product_id', 'buy_within_sameday']]
    for c in df.columns:
        df[c] = df[c].map(int)
    df.sort_values(df.columns.tolist(), inplace=True)
    df.reset_index(drop=1, inplace=1)
    
    df.to_pickle('../feature/{}/f310_user-product.p'.format(folder))

#==============================================================================
# main
#==============================================================================
make(0)
make(1)
make(2)

make(-1)

#==============================================================================
utils.end(__file__)

