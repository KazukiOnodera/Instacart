#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 05:30:59 2017

@author: konodera
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
col = ['user_id', 'order_number', 'order_id']
log = utils.read_pickles('../input/mk/log', col).drop_duplicates().sort_values(col)

ai_dep = pd.read_pickle('../input/mk/order_aisle-department.p')

log = pd.merge(log, ai_dep, on='order_id', how='left')

#==============================================================================
# calc
#==============================================================================
col = [c for c in log.columns if 'aisle_' in c or 'dep' in c]
di = defaultdict(int)
uid_bk = None

li1 = []
for args in tqdm(log[['user_id']+col].values):
    uid = args[0]
    
    if uid_bk is None:
        pass
    elif uid == uid_bk:
        pass
    elif uid != uid_bk:
        di = defaultdict(int)
    li2 = []
    for i,c in enumerate(col):
        di[c] += args[i+1]
        li2.append(di[c])
    li1.append(li2)
    
    uid_bk = uid
#==============================================================================
df = pd.DataFrame(li1, columns=col).add_suffix('_cumsum')
df['order_id'] = log['order_id']

df.to_pickle('../input/mk/order_aisle-department_cumsum.p')


#==============================================================================
utils.end(__file__)

