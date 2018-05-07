#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 19:37:49 2018

@author: Kazuki
"""

import os
from time import sleep
import sys
argv = sys.argv

file = argv[1]
if len(argv)>2:
    sec = 60 * int(argv[2])
    print(f'wait {sec} sec')
else:
    sec = 0

sleep(sec)
os.system(f'nohup python -u {file} > LOG/log_{file}.txt &')

