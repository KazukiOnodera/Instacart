#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 18:06:05 2017

@author: konodera

そのユーザがそのアイテム注文したのは何日前か？
*リークじゃない

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
import utils
utils.start(__file__)


#==============================================================================
# mk train * test log
#==============================================================================
tbl = utils.read_pickles('../input/mk/days_since_last_order')

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
        
    label = pd.read_pickle('../feature/{}/label_reordered.p'.format(folder))
    
    df = pd.merge(label[['order_id', 'product_id']], 
                 tbl[['order_id', 'product_id','days_since_last_order_this_item']], 
                 on=['order_id', 'product_id'], how='left')
    
    df.to_pickle('../feature/{}/f303_order-product.p'.format(folder))
#==============================================================================
# main
#==============================================================================
make(0)
make(1)
make(2)

make(-1)



#==============================================================================
utils.end(__file__)

