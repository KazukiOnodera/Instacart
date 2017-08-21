#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 07:55:48 2017

@author: konodera

None

Leak!

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import utils
utils.start(__file__)

#==============================================================================
# load
#==============================================================================
col = ['order_id', 'user_id','order_number','product_name', 'eval_set']
order_tbl = pd.read_pickle('../input/mk/order_tbl.p')[col]
order_tbl.sort_values(['user_id', 'order_number'], inplace=True)
order_tbl = order_tbl[order_tbl.eval_set!='test']

#==============================================================================
# main
#==============================================================================

uid_bk = None
product_name_all = [] # 2d list
pname_unq = []        # 1d list
pname_unq_len = []     # 1d list
for uid,pnames in tqdm(order_tbl[['user_id', 'product_name']].values):
    if uid_bk is None:
        pname_unq += pnames
    elif uid == uid_bk:
        pname_unq += pnames
    elif uid != uid_bk:
        pname_unq = pnames[:]
        
    uid_bk = uid
    pname_unq = list(set(pname_unq))
    pname_unq_len.append(len(pname_unq))
    product_name_all.append(pname_unq)

order_tbl['product_name_all'] = product_name_all
order_tbl['product_unq_len'] = pname_unq_len
order_tbl['new_item_cnt'] = order_tbl.groupby('user_id').product_unq_len.diff()
order_tbl['product_len'] = order_tbl['product_name'].map(len)
order_tbl['is_None'] = (order_tbl.new_item_cnt == order_tbl.product_len)*1

col = ['order_id', 'product_unq_len', 'is_None']
order_tbl[col].to_pickle('../input/mk/order_None.p')



utils.end(__file__)

