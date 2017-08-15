#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 23:28:19 2017

@author: konodera

nohup python -u 101_check-imp.py > LOG/_ch-imp_None.txt &


"""

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
import sys
sys.path.append('/home/konodera/Python')
import xgbextension as ex
import xgboost as xgb
import ini
#import time
#time.sleep(3600*2)
import utils
utils.start(__file__)



# setting
LOOP = 5
ESR = 20
W = 3

seed = np.random.randint(99999)
#seed = 71

np.random.seed(seed)

valid_size = 0.2


# XGB param
nround = 10000
#nround = 10

param = {'max_depth':10, 
         'eta':0.1,
         'colsample_bytree':1,
         'subsample':1,
         'silent':1, 
         'nthread':27,
#         'scale_pos_weight':y_build.mean(),
#         'eval_metric':'auc',
         'eval_metric':'logloss',
         'objective':'binary:logistic',
         'tree_method':'hist'}

print("""#==== print param ======""")
print('seed:', seed)

#==============================================================================
# prepare
#==============================================================================
train = pd.concat([utils.load_pred_None('trainT-0', W, True).sample(frac=.2),
                   utils.load_pred_None('trainT-1', W, True).sample(frac=.2),
                   utils.load_pred_None('trainT-2', W, True).sample(frac=.2)
                   ], ignore_index=True, join='inner')

sub_train = train[['order_id', 'y']]
y_train = train['y']
X_train = train.drop('y', axis=1)
del train; gc.collect()

# drop id
col = [c for c in X_train.columns if '_id' in c] + ['is_train']
col.remove('user_id')
print('drop1',col)
X_train.drop(col, axis=1, inplace=True) # keep user_id

# drop obj
col = X_train.dtypes[X_train.dtypes=='object'].index.tolist()
print('drop2',col)
X_train.drop(col, axis=1, inplace=True)

X_train.fillna(-1, inplace=1)

#==============================================================================
# check corr
#==============================================================================
from itertools import combinations as comb
tmp = X_train.sample(2999)

li = []
col = comb(tmp.columns,2)
for c1,c2 in col:
    if abs(tmp[c1].corr(tmp[c2])) == 1:
        print(c1,c2)
        li.append(c2)
del tmp
gc.collect()

print('dup', li)
X_train.drop(li, axis=1, inplace=True)

#==============================================================================
# SPLIT!
print('split by user')
#==============================================================================
train_user = X_train[['user_id']].drop_duplicates()

def split_build_valid():
    
    train_user['is_valid'] = np.random.choice([0,1], size=len(train_user), 
                                              p=[1-valid_size, valid_size])
    valid_n = train_user['is_valid'].sum()
    build_n = (train_user.shape[0] - valid_n)
    
    print('build user:{}, valid user:{}'.format(build_n, valid_n))
    valid_user = train_user[train_user['is_valid']==1].user_id
    is_valid = X_train.user_id.isin(valid_user)
    
    sub_build = sub_train[~is_valid]
    sub_valid = sub_train[is_valid]
    
    dbuild = xgb.DMatrix(X_train[~is_valid].drop('user_id', axis=1), y_train[~is_valid])
    dvalid = xgb.DMatrix(X_train[is_valid].drop('user_id', axis=1), label=y_train[is_valid])
    watchlist = [(dbuild, 'build'),(dvalid, 'valid')]
    
    print('FINAL SHAPE')
    print('dbuild.shape:{}  dvalid.shape:{}\n'.format((dbuild.num_row(), dbuild.num_col()),
                                                      (dvalid.num_row(), dvalid.num_col())))

    return dbuild, dvalid, watchlist, sub_build, sub_valid

#==============================================================================
# cv
#==============================================================================
models = []
for i in range(LOOP):
    dbuild, dvalid, watchlist, sub_build, sub_valid = split_build_valid()
    model = xgb.train(param, dbuild, nround, watchlist,
                      early_stopping_rounds=ESR, verbose_eval=5)
    models.append(model)
    del dbuild, dvalid, watchlist, sub_build, sub_valid
    gc.collect()

imp = ex.getImp(models)
imp.to_csv('imp_None.csv', index=0)




#==============================================================================
utils.end(__file__)



