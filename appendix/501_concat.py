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
#utils.start(__file__)

#==============================================================================
# def
#==============================================================================
def concat_pred_item(T, dryrun=False):
    if T==-1:
        name = 'test'
    else:
        name = 'trainT-'+str(T)
    
    df = utils.load_pred_item(name)
    
    df = pd.merge(df, pd.read_pickle('../feature/{}/f317_user-product.p'.format(name)), 
                  on=['user_id', 'product_id'],how='left')
    
    gc.collect()
    
    #==============================================================================
    print('output')
    #==============================================================================
    if dryrun == True:
        return df
    else:
        utils.to_pickles(df, '../feature/{}/all_apdx'.format(name), 20, inplace=True)

def multi(name):
    concat_pred_item(name)

#==============================================================================

# multi
mp_pool = mp.Pool(2)
mp_pool.map(multi, [0, -1])



utils.end(__file__)

