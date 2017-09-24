#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 12:55:38 2017

@author: konodera

https://twitter.com/jeremystan/status/911357665481080832

6/ most novel feature: 
    binary user by product purchase sequence -> 
    decimal -> XGBoost learns non-trivial sequence patterns

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from decimal import Decimal
import utils
#utils.start(__file__)


#==============================================================================
# load
#==============================================================================
col = ['order_id', 'user_id', 'product_id', 'order_number', 'order_number_rev']
log = utils.read_pickles('../input/mk/log', col).sort_values(['user_id', 'product_id', 'order_number'])


#==============================================================================
# def
#==============================================================================
def conv_bi2dec(seq, onb_max, reverse=True, deci=10):
    """
    ex.
    seq     = [1,3,4]
    onb_max = 6
            101100 -> 44
            001101 -> 13
    """
    
    bi = [0]*onb_max
    for i in seq:
        bi[i-1] = 1
    
    if reverse:
        bi = ''.join(map(str, bi))[::-1]
    else:
        bi = ''.join(map(str, bi))
    
    if deci==10:
        return int(bi, 2)
    elif deci==2:
        return int(bi)
    elif deci==.2:
        return float(bi[0] + '.' + bi[1:])
    else:
        raise

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
    log_['onb_max'] = log_.groupby('user_id').order_number.transform(np.max)
    
    r1_d10 = []
    r1_d2 = []
    r1_df2 = []
    r0_d10 = []
    r0_d2 = []
    r0_df2 = []
    
    seq = []
    uid_bk = pid_bk = onb_max_bk = None
    for uid,pid,onb,onb_max in tqdm(log_[['user_id', 'product_id', 'order_number', 'onb_max']].values):
        
        if uid_bk is None:
            pass
        
        elif uid==uid_bk and pid==pid_bk:
            pass
        
        elif uid!=uid_bk or pid!=pid_bk:
            r1_d10.append(conv_bi2dec(seq, onb_max_bk, True,  10))
            r1_d2.append(conv_bi2dec(seq, onb_max_bk,  True,  2))
            r1_df2.append(conv_bi2dec(seq, onb_max_bk, False, .2))
            r0_d10.append(conv_bi2dec(seq, onb_max_bk, True,  10))
            r0_d2.append(conv_bi2dec(seq, onb_max_bk,  True,  2))
            r0_df2.append(conv_bi2dec(seq, onb_max_bk, False, .2))
            seq = []
            
        seq.append(onb)
        uid_bk = uid
        pid_bk = pid
        onb_max_bk = onb_max
    
    r1_d10.append(conv_bi2dec(seq, onb_max_bk, True,  10))
    r1_d2.append(conv_bi2dec(seq, onb_max_bk,  True,  2))
    r1_df2.append(conv_bi2dec(seq, onb_max_bk, False, .2))
    r0_d10.append(conv_bi2dec(seq, onb_max_bk, True,  10))
    r0_d2.append(conv_bi2dec(seq, onb_max_bk,  True,  2))
    r0_df2.append(conv_bi2dec(seq, onb_max_bk, False, .2))
    
    df = log_[['user_id', 'product_id']].drop_duplicates(keep='first').reset_index(drop=True)
    df['seq2dec_r1_d10'] = r1_d10
    df['seq2dec_r1_d2']  = r1_d2
    df['seq2dec_r1_df2'] = r1_df2
    df['seq2dec_r0_d10'] = r0_d10
    df['seq2dec_r0_d2']  = r0_d2
    df['seq2dec_r0_df2'] = r0_df2
    
    df.to_pickle('../feature/{}/f317_user-product.p'.format(folder))
    
    
#==============================================================================
# main
#==============================================================================
make(0)
#make(1)
#make(2)

make(-1)



#==============================================================================
utils.end(__file__)

