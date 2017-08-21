#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 23:28:19 2017

@author: konodera

nohup python -u 002_xgb_holdout_item_813_1.py > LOG/_xgb_item.txt &


"""

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import gc
import xgboost as xgb
import utils

utils.start(__file__)



# setting
DATE = '813_1'
LOOP = 2
ESR = 60

#seed = np.random.randint(99999)
seed = 72

np.random.seed(seed)

valid_size = 0.05


# XGB param
nround = 10000
#nround = 10

param = {'max_depth':10, 
         'eta':0.02,
         'colsample_bytree':0.4,
         'subsample':0.75,
         'silent':1,
         'nthread':27,
         'eval_metric':'logloss',
         'objective':'binary:logistic',
         'tree_method':'hist'
         }

print("""#==== print param ======""")
print('DATE:', DATE)
print('seed:', seed)

#==============================================================================
# prepare
#==============================================================================
train = pd.concat([utils.load_pred_item('trainT-0'),
                   utils.load_pred_item('trainT-1'),
                   utils.load_pred_item('trainT-2')
                   ], ignore_index=True)

y_train = train['y']
X_train = train.drop('y', axis=1)
del train
gc.collect()

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
    
    dbuild = xgb.DMatrix(X_train[~is_valid].drop('user_id', axis=1), y_train[~is_valid])
    dvalid = xgb.DMatrix(X_train[is_valid].drop('user_id', axis=1), label=y_train[is_valid])
    watchlist = [(dbuild, 'build'),(dvalid, 'valid')]
    
    print('FINAL SHAPE')
    print('dbuild.shape:{}  dvalid.shape:{}\n'.format((dbuild.num_row(), dbuild.num_col()),
                                                      (dvalid.num_row(), dvalid.num_col())))

    return dbuild, dvalid, watchlist

#==============================================================================
print('hold out')
#==============================================================================
utils.mkdir_p('../output/model/{}/'.format(DATE))
utils.mkdir_p('../output/imp/{}/'.format(DATE))
utils.mkdir_p('../output/sub/{}/'.format(DATE))

# hold out
models = []
for i in range(LOOP):
    print('LOOP',i)
    dbuild, dvalid, watchlist = split_build_valid()
    
    if i==0:
        col_train = dbuild.feature_names
        
    model = xgb.train(param, dbuild, nround, watchlist,
                      early_stopping_rounds=ESR, verbose_eval=5)
    models.append(model)
    model.save_model('../output/model/{}/xgb_item_{}.model'.format(DATE, i))
    # VALID
    valid_yhat = model.predict(dvalid)
    print('Valid Mean:', np.mean(valid_yhat))
    del dbuild, dvalid, watchlist
    gc.collect()

del train_user, X_train, y_train
gc.collect()

#==============================================================================
print('test')
#==============================================================================
test = utils.load_pred_item('test').fillna(-1)

sub_test = test[['order_id', 'product_id']]

dtest  = xgb.DMatrix(test[col_train])
sub_test['yhat'] = 0
for model in models:
    sub_test['yhat'] += model.predict(dtest)
sub_test['yhat'] /= LOOP
print('Test Mean:', sub_test['yhat'].mean())

sub_test.to_pickle('../output/sub/{}/sub_test.p'.format(DATE))


#==============================================================================
utils.end(__file__)



