#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 23:31:30 2017

@author: konodera

LEAK

"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
total_proc = 60
import utils
utils.start(__file__)

usecols = [ 'order_id', 'user_id', 'product_id', 'order_number']
log = utils.read_pickles('../input/mk/log', usecols)


def multi(uid):
    tmp = log[log.user_id==uid]
    ct = pd.crosstab(tmp.order_number, tmp.product_id).reset_index().set_index('order_number')
    li = []
    for pid in ct.columns:
        streak = 0
        sw_odr = False
        for onb,odr in enumerate(ct[pid].values):
            onb+=1
            if sw_odr == False and odr == 1:
                sw_odr = True
                streak = 1
                li.append([uid, pid, onb, streak])
                continue
            if sw_odr == True:
                if odr == 1 and streak>0:
                    streak += 1
                    li.append([uid, pid, onb, streak])
                elif odr == 1 and streak<=0:
                    streak = 1
                    li.append([uid, pid, onb, streak])
                elif odr == 0 and streak>0:
                    streak = 0
                    li.append([uid, pid, onb, streak])
                elif odr == 0 and streak<=0:
                    streak -= 1
                    li.append([uid, pid, onb, streak])
    return pd.DataFrame(li, columns=['user_id', 'product_id', 'order_number', 'streak'])


user_id = log.user_id.unique()
mp_pool = mp.Pool(total_proc)
callback = mp_pool.map(multi, user_id)

df = pd.concat(callback, ignore_index=True)

order = log[['order_id', 'user_id', 'order_number']].drop_duplicates().reset_index(drop=True)
df = pd.merge(df, order, on=['user_id', 'order_number'], how='left')

df[['order_id', 'product_id', 'streak']].to_pickle('../input/mk/streak_order-product.p')


utils.end(__file__)

