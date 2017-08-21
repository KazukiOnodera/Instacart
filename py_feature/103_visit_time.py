#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 08:45:57 2017

@author: konodera

visit time ratio

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import utils
utils.start(__file__)

#==============================================================================
# load
#==============================================================================
X_base = pd.read_pickle('../feature/X_base_t3.p')

col = ['order_id', 'user_id', 'product_id', 'order_dow', 'order_hour_of_day', 'order_number_rev']
log = utils.read_pickles('../input/mk/log', col)
log = pd.merge(log, pd.read_pickle('../input/mk/timezone.p'), 
               on='order_hour_of_day', how='left')
log['dow_tz'] = log.order_dow.map(str) + '_' + log.timezone

#==============================================================================
# train
#==============================================================================
def make(T):
    log_tr = log[log.order_number_rev>T]
    
    # dow
    dow  = pd.crosstab(log_tr.user_id, log_tr.order_dow).add_prefix('user_dow_freq_')
    dow_ = pd.crosstab(log_tr.user_id, log_tr.order_dow, normalize='index').add_prefix('user_dow_norm_')
    
    # timezone
    timezone  = pd.crosstab(log_tr.user_id, log_tr.timezone).add_prefix('user_timezone_freq_')
    timezone_ = pd.crosstab(log_tr.user_id, log_tr.timezone, normalize='index').add_prefix('user_timezone_norm_')
    
    # dow * timezone
    dow_tz  = pd.crosstab(log_tr.user_id, log_tr.dow_tz).add_prefix('user_dow-tz_freq_')
    dow_tz_ = pd.crosstab(log_tr.user_id, log_tr.dow_tz, normalize='index').add_prefix('user_dow-tz_norm_')
    
    tab = pd.concat([dow, dow_, timezone, timezone_, dow_tz, dow_tz_], axis=1)
    
    tab.reset_index().to_pickle('../feature/trainT-{}/f103_user.p'.format(T))

make(0)
make(1)
make(2)

#==============================================================================
# test
#==============================================================================

# dow
dow  = pd.crosstab(log.user_id, log.order_dow).add_prefix('user_dow_freq_')
dow_ = pd.crosstab(log.user_id, log.order_dow, normalize='index').add_prefix('user_dow_norm_')

# timezone
timezone  = pd.crosstab(log.user_id, log.timezone).add_prefix('user_timezone_freq_')
timezone_ = pd.crosstab(log.user_id, log.timezone, normalize='index').add_prefix('user_timezone_norm_')

# dow * timezone
dow_tz  = pd.crosstab(log.user_id, log.dow_tz).add_prefix('user_dow-tz_freq_')
dow_tz_ = pd.crosstab(log.user_id, log.dow_tz, normalize='index').add_prefix('user_dow-tz_norm_')

tab = pd.concat([dow, dow_, timezone, timezone_, dow_tz, dow_tz_], axis=1)

tab.reset_index().to_pickle('../feature/test/f103_user.p')








utils.end(__file__)

