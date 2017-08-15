#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 17:41:54 2017

@author: konodera
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import utils
utils.start(__file__)


item = pd.read_csv('../input/products.csv')


item['item_is_Organic'] = item.product_name.map(lambda x: 'organic' in x.lower())*1
item['item_is_Gluten-Free'] = item.product_name.map(lambda x: 'gluten' in x.lower() and 'free' in x.lower())*1
item['item_is_Asian'] = item.product_name.map(lambda x: 'asian' in x.lower())*1


col = ['product_id', 'item_is_Organic', 'item_is_Gluten-Free', 'item_is_Asian']
item[col].to_pickle('../input/mk/products_feature.p')






utils.end(__file__)

