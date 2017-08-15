#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 00:29:00 2017

@author: konodera


"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import utils
utils.start(__file__)


col = ['order_id', 'user_id', 'product_id', 'order_number', 'reordered']
log = utils.read_pickles('../input/mk/log', col)
log.sort_values(['user_id', 'product_id', 'order_number'], inplace=True)



uid_bk = pid_bk = onum_bk = None
ret = []
miniters = int(log.shape[0]/50)
col = ['user_id', 'product_id', 'order_number']
for uid,pid,onum in tqdm(log[col].values,miniters=miniters):
    if uid_bk is None:
        cnt = 1
        ret.append(cnt)
    elif uid == uid_bk and pid == pid_bk:
        if onum - onum_bk == 1:
            cnt+=1
            ret.append(cnt)
        else:
            cnt = 1
            ret.append(cnt)
        pass
    elif uid == uid_bk and pid != pid_bk: # item change
        cnt = 1
        ret.append(cnt)
    elif uid != uid_bk: # user change
        cnt = 1
        ret.append(cnt)
    else:
        raise Exception('?')

    uid_bk = uid
    pid_bk = pid
    onum_bk = onum
log['buy_item_inarow'] = ret

log.reset_index(drop=1, inplace=1)

log.to_pickle('../input/mk/log_inarow.p')


utils.end(__file__)

