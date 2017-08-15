# -*- coding: utf-8 -*-
"""
Created on Wed May 17 01:21:53 2017

@author: konodera
"""

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from glob import glob
import os
from tqdm import tqdm
#import pickle
import time
import gc
#from itertools import chain



def start(fname):
    global st_time
    st_time = time.time()
    print("""
#==============================================================================
# START !!! {} PID: {}
#==============================================================================
""".format(fname, os.getpid()))
    
    return

def end(fname):
    
    print("""
#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(fname))
    print('time: {:.2f}min'.format( (time.time() - st_time)/60 ))
    return

def mkdir_p(path):
    try:
        os.stat(path)
    except:
        os.mkdir(path)
    
def to_pickles(df, path, split_size=3, inplace=False):
    """
    path = '../output/mydf'
    
    wirte '../output/mydf/0.p'
          '../output/mydf/1.p'
          '../output/mydf/2.p'
    
    """
    if inplace==True:
        df.reset_index(drop=True, inplace=True)
    else:
        df = df.reset_index(drop=True)
    mkdir_p(path)
    
    for i in tqdm(range(split_size)):
        df.ix[df.index%split_size==i].to_pickle(path+'/{}.p'.format(i))
    
    return

def read_pickles(path, col=None):
    if col is None:
        df = pd.concat([pd.read_pickle(f) for f in tqdm(sorted(glob(path+'/*')))])
    else:
        df = pd.concat([pd.read_pickle(f)[col] for f in tqdm(sorted(glob(path+'/*')))])
    return df

def reduce_memory(df, ix_start=0):
    df.fillna(-1, inplace=True)
    df_ = df.sample(9999, random_state=71)
    ## int
    col_int8 = []
    col_int16 = []
    col_int32 = []
    for c in tqdm(df.columns[ix_start:], miniters=20):
        if df[c].dtype=='O':
            continue
        if (df_[c] == df_[c].astype(np.int8)).all():
            col_int8.append(c)
        elif (df_[c] == df_[c].astype(np.int16)).all():
            col_int16.append(c)
        elif (df_[c] == df_[c].astype(np.int32)).all():
            col_int32.append(c)
    
    df[col_int8]  = df[col_int8].astype(np.int8)
    df[col_int16] = df[col_int16].astype(np.int16)
    df[col_int32] = df[col_int32].astype(np.int32)
    
    ## float
    col = [c for c in df.dtypes[df.dtypes==np.float64].index if '_id' not in c]
    df[col] = df[col].astype(np.float32)

    gc.collect()

#==============================================================================
# main
#==============================================================================
if __name__ == "__main__":
    
    files = sorted(glob('../input/*'))
    data = {}
    for f in files:
        if os.path.isfile(f):
            data[f.split('/')[-1]] = pd.read_csv(f)
    
    print("""
    #==============================================================================
    # SUCCESS !!! {}
    #==============================================================================
    """.format(__file__))

