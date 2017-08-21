#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 00:48:08 2017

@author: konodera

aisle & department

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import utils
utils.start(__file__)


#==============================================================================
# load
#==============================================================================
usecols = [ 'order_id', 'user_id', 'product_id', 'order_number', 'order_number_rev']
log = utils.read_pickles('../input/mk/log', usecols)

goods = pd.read_pickle('../input/mk/goods.p')[['product_id', 'aisle_id', 'department_id']]

log = pd.merge(log, goods, on='product_id', how='left')

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
    
    user = log_.groupby(['user_id']).size().to_frame()
    user.columns = ['total']
    user.reset_index(inplace=True)
    
    user_aisle = log_.groupby(['user_id', 'aisle_id']).size().to_frame()
    user_aisle.columns = ['user_aisle_cnt']
    user_aisle.reset_index(inplace=True)
    user_aisle = pd.merge(user_aisle, user, on='user_id', how='left')
    user_aisle['user_aisle_ratio'] = user_aisle.user_aisle_cnt / user_aisle.total
    user_aisle.drop('total', axis=1, inplace=True)
    user_aisle.to_pickle('../feature/{}/f313_user_aisle.p'.format(folder))
    
    user_dep = log_.groupby(['user_id', 'department_id']).size().to_frame()
    user_dep.columns = ['user_dep_cnt']
    user_dep.reset_index(inplace=True)
    user_dep = pd.merge(user, user_dep, on='user_id', how='left')
    user_dep['user_dep_ratio'] = user_dep.user_dep_cnt / user_dep.total
    user_dep.drop('total', axis=1, inplace=True)
    user_dep.to_pickle('../feature/{}/f313_user_dep.p'.format(folder))

#==============================================================================
# main
#==============================================================================
make(0)
make(1)
make(2)

make(-1)



#==============================================================================
utils.end(__file__)

