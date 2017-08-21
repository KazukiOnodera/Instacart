#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 13:13:57 2017

@author: konodera
"""

import os
import utils
utils.start(__file__)


os.system('python -u 002_xgb_holdout_item_812_1.py')
os.system('python -u 002_xgb_holdout_item_813_1.py')
os.system('python -u 002_xgb_holdout_item_813_3.py')

os.system('python -u 102_xgb_holdout_None_813_3.py')
os.system('python -u 102_xgb_holdout_None_814_1.py')
os.system('python -u 102_xgb_holdout_None_814_2.py')
os.system('python -u 102_xgb_holdout_None_814_3.py')

os.system('python -u 201_Faron_opt_bagging_815_3.py')

utils.end(__file__)

