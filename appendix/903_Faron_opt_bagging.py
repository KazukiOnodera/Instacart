#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 18:59:46 2017

@author: konodera

nohup python -u 201_Faron_opt_bagging_815_3.py > LOG/_Faron-opt.txt &

"""

import pandas as pd
import sys
sys.path.append('../py_model')
from opt_fscore import get_best_prediction
import multiprocessing as mp
import time
import utils
utils.start(__file__)


# setting
DATE_None = ['813_3', '814_1', '814_2', '814_3']

total_proc = 60


utils.mkdir_p('../output/sub/apdx')
#==============================================================================
# def
#==============================================================================
def multi(i):
    if i%1000==0:
        print('{:.3f} min'.format((time.time()-st_time)/60))
    items = sub.loc[i,'product_id']
    preds = sub.loc[i,'yhat']
    pNone = sub.loc[i,'yhat_None']
    ret = get_best_prediction(items, preds, pNone)
    return ret

def mk_sub(DATE_item):
    print("""#==== print param ======""")
    print('OUTF:', OUTF)
    print('DATE_item:', DATE_item)
    print('DATE_None:', DATE_None)
    print('total_proc:', total_proc)
    
    global sub, st_time
    
    sub_item = pd.read_pickle('../output/sub/{}/sub_test.p'.format(DATE_item))
    sub = sub_item.groupby('order_id').product_id.apply(list).to_frame()
    sub['yhat'] = sub_item.groupby('order_id').yhat.apply(list)
    
    # weighted
    for i,(w,d) in enumerate(zip([0.1, 0.1, 0.4, 0.4], DATE_None)):
        tmp = pd.read_pickle('../output/sub/{}/sub_test_None.p'.format(d)).rename(columns={'yhat':'yhat_None'})
        tmp.yhat_None *= w
        if i==0:
            sub_None = tmp
        else:
            sub_None = pd.concat([sub_None, tmp])
    
    sub_None = sub_None.groupby('order_id').yhat_None.sum().reset_index()
    
    sub = pd.merge(sub.reset_index(), sub_None, on='order_id', how='left')
    
    # optimize start!!!
    st_time = time.time()
    pool = mp.Pool(total_proc)
    callback = pool.map(multi, range(sub.shape[0]))
    
    sub['products'] = callback
    
    print('writing...')
    sub[['order_id', 'products']].to_csv(OUTF, index=0, compression='gzip')
#==============================================================================
OUTF = "../output/sub/apdx/bench.csv.gz"
mk_sub('apdx_base')

OUTF = "../output/sub/apdx/seq2dec.csv.gz"
mk_sub('apdx')





#==============================================================================
utils.end(__file__)

