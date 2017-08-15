"""
Created on Fri Jun 30 15:09:33 2017

@author: konodera
"""
from operator import itemgetter
import numpy as np

LOOP = 9999
np.random.seed(71)

cdef int __tp__(y_true, y_pred):
    return len(y_true & y_pred)

cdef int __tpfp__(y_pred):
    return len(y_pred)

cdef int __tpfn__(y_true):
    return len(y_true)

cdef double multilabel_fscore(y_true, y_pred):
    cdef double precision, recall
    cdef double tp, tpfp, tpfn
    
    tp = __tp__(y_true, y_pred)
    tpfp = __tpfp__(y_pred)
    tpfn = __tpfn__(y_true)
    
    precision = tp/tpfp
    recall = tp/tpfn
    
    if precision + recall == 0:
        return 0
    return (2 * precision * recall) / (precision + recall)

cdef get_y_true(items):
    """
    items: dict
    {A:0.9, B:0.3}
    """
    cdef list y_true = []
    for k in items.keys():
        if items[k]>np.random.uniform():
            y_true.append(k)
    if len(y_true)==0 or 'None' in y_true:
        y_true = ['None']
    return y_true

def get_best_items(items, preds):
    """
    items: list
    [1, 2, 3...]
    
    preds: list
    [0.3, 0.9, 0.2...]
    
    items = [1, 2, 3, 4, 5, 6, 7]
    preds = [0.2, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14]
    
    """
    items_true = dict(zip(items, preds))
    cdef list items_pred = sorted(list(zip(items, preds)), key=itemgetter(1), reverse=True)
    items_pred = [k for k,v in items_pred]
    cdef list y_trues = [set(get_y_true(items_true)) for i in range(LOOP)]
    cdef list best_items
    
    cdef double best_score = 0
    for i in range(1,len(items_pred)+1):
        score = np.mean([multilabel_fscore(y_trues[j], set(items_pred[:i])) for j in range(LOOP)])
        if best_score < score:
            best_score = score
        elif best_score > score:
            best_items = items_pred[:i-1]
            break
        if i==len(items_pred):
            # last
            best_items = items_pred[:]
            break
    
    if 'None' in best_items:
        return ' '.join(map(str, best_items))
    
    # search None
    best_items = best_items[::-1] # low is head
    for i in range(len(best_items)+1):
        score = np.mean([multilabel_fscore(y_trues[j], set(best_items[i:]+['None'])) for j in range(LOOP)])
        if best_score < score:
            best_score = score
        elif best_score > score and i==0:
            break
        elif best_score > score:
            best_items = best_items[i-1:]+['None']
            break
        elif i==len(best_items):
            # last
            best_items = ['None']
            break
    
    return ' '.join(map(str, best_items))

def get_best_items2(items, preds):
    """
    items: list
    [1, 2, 3...]
    
    preds: list
    [0.3, 0.9, 0.2...]
    
    ex:
    items = [1, 2, 3, 4, 5, 6, 7]
    preds = [0.2, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14]
    
    """
    items_true = dict(zip(items, preds))
    cdef list items_pred = sorted(list(zip(items, preds)), key=itemgetter(1), reverse=True)
    items_pred = [k for k,v in items_pred]
    cdef list y_trues = [set(get_y_true(items_true)) for i in range(LOOP)]
    cdef list best_items
    
    cdef double best_score = 0
    for i in range(1,len(items_pred)+1):
        score = np.mean([multilabel_fscore(y_trues[j], set(items_pred[:i])) for j in range(LOOP)])
        if best_score < score:
            best_score = score
        elif best_score > score:
            best_items = items_pred[:i-1]
            break
        if i==len(items_pred):
            # last
            best_items = items_pred[:]
            break
    
    return ' '.join(map(str, best_items))

