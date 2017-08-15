# -*- coding: utf-8 -*-
"""
Created on Wed May 17 08:55:13 2017

@author: konodera
"""

import pandas as pd
import numpy as np
import gc
import utils
utils.start(__file__)

#==============================================================================
# test user
#==============================================================================
orders = pd.read_csv('../input/orders.csv.gz')

test_user = orders.loc[orders.eval_set=='test'].reset_index(drop=1)
test_user[['order_id', 'user_id']].to_pickle('../input/mk/test_user.p')


#==============================================================================
# goods
#==============================================================================
products = pd.read_csv('../input/products.csv')
products.product_name = products.product_name.str.replace(' ', '-')

aisles = pd.read_csv('../input/aisles.csv', engine='c')
departments = pd.read_csv('../input/departments.csv', engine='c')

goods = pd.merge(left=pd.merge(left=products, right=departments, how='left'), right=aisles, how='left')


goods.to_pickle('../input/mk/goods.p')
gc.collect()
#==============================================================================
# log
#==============================================================================
log = pd.concat([pd.read_csv('../input/order_products__prior.csv.gz'), 
                 pd.read_csv('../input/order_products__train.csv.gz')], 
                ignore_index=1)

log.sort_values(['order_id', 'add_to_cart_order'], inplace=True)
log.reset_index(drop=1, inplace=True)
log = pd.merge(log, goods, on='product_id', how='left')
log = pd.merge(log, orders, on='order_id', how='left')
log['order_number_rev'] = log.groupby('user_id').order_number.transform(np.max) - log.order_number

utils.to_pickles(log, '../input/mk/log', 20)

gc.collect()
#==============================================================================
# order_tbl
#==============================================================================
order_product = log.groupby('order_id').product_name.apply(list).reset_index()
order_tbl = pd.merge(orders, order_product, on='order_id', how='left')

order_tbl.sort_values(['user_id', 'order_number'],inplace=True)
order_tbl.reset_index(drop=1, inplace=True)
order_tbl = pd.merge(order_tbl, log[['order_id','order_number_rev']].drop_duplicates(), on='order_id', how='left')
order_tbl.order_number_rev = order_tbl.order_number_rev.fillna(-1).astype(int)
#order_tbl['order_number_rev'] = order_tbl.groupby('user_id').order_number.transform(np.max) - order_tbl.order_number
order_tbl['days_since_first_order'] = order_tbl.groupby('user_id').days_since_prior_order.cumsum()

def set_diff(items1, items2):
    if  isinstance(items1, float) or isinstance(items2, float):
        return items1
    return [i1 for i1 in items1 if i1 not in items2]

def same_products(items1, items2):
    if  isinstance(items1, float) or isinstance(items2, float):
        return []
    return [i1 for i1 in items1 if i1 in items2]

order_tbl['t-1_product_name'] = order_tbl.groupby('user_id')['product_name'].shift(1)
order_tbl['set_diff_products'] = order_tbl.apply(lambda x: set_diff(x['product_name'], x['t-1_product_name']), axis=1)
order_tbl['same_products'] = order_tbl.apply(lambda x: same_products(x['product_name'], x['t-1_product_name']), axis=1)

order_tbl.to_pickle('../input/mk/order_tbl.p')
gc.collect()
#==============================================================================
# order_aisle-department
#==============================================================================
order_aisle      = pd.crosstab(log['order_id'], 
                               log['aisle_id']).add_prefix('aisle_').reset_index()

order_department = pd.crosstab(log['order_id'], 
                               log['department_id']).add_prefix('department_').reset_index()

order_aisle = pd.merge(order_aisle, order_department, on='order_id', how='left')

order_aisle.to_pickle('../input/mk/order_aisle-department.p')

del order_aisle, order_department
gc.collect()

#==============================================================================
# order_reorderd
#==============================================================================
log_ = log.loc[log.reordered==1]
order_reorderd = log_.groupby('order_id').product_id.apply(list).reset_index()

order_reorderd.to_pickle('../input/mk/order_reorderd.p')
gc.collect()

#==============================================================================
# user_order
#==============================================================================
from itertools import chain

order_tbl = pd.read_pickle('../input/mk/order_tbl.p')
order_tbl = order_tbl.loc[order_tbl.eval_set!='test']

goods = pd.read_pickle('../input/mk/goods.p')

goods_di = {}
for k,v in zip(goods.product_name, goods.product_id):
    goods_di[k] = v


def sum_list(x):
    return list(chain.from_iterable(x))

def to_unique(lists):
    li = sum_list(lists)
    return list(set(li))

def to_ids(names):
    ids = [goods_di[n] for n in names]
    return ids

user_hist = order_tbl.groupby('user_id').product_name.apply(to_unique).reset_index()
user_hist['product_id'] = user_hist.product_name.map(to_ids)

user_hist.to_pickle('../input/mk/user_order.p')











utils.end(__file__)


