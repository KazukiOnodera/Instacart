#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 00:54:02 2017

@author: konodera

pid      freq
-------------
24852    57186
13176    47063
21137    39871
21903    38095
47209    30047
47626    28741
47766    28478
26209    26199
16797    25621
24964    21090
22935    20824
27966    20193
39275    20134
45007    19652
49683    17508
4605     16176
27845    16134
40706    16054
5876     15765
4920     15150
28204    14802
42265    14766
30391    14089
31717    13949
8277     13900
8518     13770
27104    13719
17794    13642
46979    13491
45066    13289

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import utils
utils.start(__file__)

#==============================================================================
# load
#==============================================================================

col = [ 'order_id', 'user_id', 'product_id', 'order_number', 'reordered', 'order_number_rev']
log = utils.read_pickles('../input/mk/log', col)


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
    
    user = log_.drop_duplicates('user_id')[['user_id']].reset_index(drop=True)
    
    # have you bought -> hyb
    tag_user = log_[log_.product_id==24852].user_id
    user['hyb_Banana'] = 0
    user.loc[user.user_id.isin(tag_user), 'hyb_Banana'] = 1
    
    tag_user = log_[log_.product_id==13176].user_id
    user['hyb_BoO-Bananas'] = 0
    user.loc[user.user_id.isin(tag_user), 'hyb_BoO-Bananas'] = 1
    
    tag_user = log_[log_.product_id==21137].user_id
    user['hyb_Organic-Strawberries'] = 0
    user.loc[user.user_id.isin(tag_user), 'hyb_Organic-Strawberries'] = 1
    
    tag_user = log_[log_.product_id==21903].user_id
    user['hyb_Organic-Baby-Spinach'] = 0
    user.loc[user.user_id.isin(tag_user), 'hyb_Organic-Baby-Spinach'] = 1
    
    tag_user = log_[log_.product_id==47209].user_id
    user['hyb_Organic-Hass-Avocado'] = 0
    user.loc[user.user_id.isin(tag_user), 'hyb_Organic-Hass-Avocado'] = 1
    
    user.to_pickle('../feature/{}/f109_user.p'.format(folder))

#==============================================================================
# main
#==============================================================================
make(0)
make(1)
make(2)

make(-1)














utils.end(__file__)

