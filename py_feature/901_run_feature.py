#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 23:13:37 2017

@author: konodera


nohup python -u 901_run_feature.py > log_run_feature.txt &


from glob import glob
files = [f for f in sorted(glob('*.py')) if f[0].isdigit()]
for f in files:
    print("os.system('python -u {}')".format(f))


"""

import os
import utils
utils.start(__file__)

utils.mkdir_p('../input/mk')
utils.mkdir_p('../output')
utils.mkdir_p('../output/model')
utils.mkdir_p('../output/sub')
utils.mkdir_p('../output/imp')
utils.mkdir_p('../feature')
utils.mkdir_p('../feature/trainT-0')
utils.mkdir_p('../feature/trainT-1')
utils.mkdir_p('../feature/trainT-2')
utils.mkdir_p('../feature/test')


os.system('python -u 000_mk.py')
os.system('python -u 003_X_base_T.py')
os.system('python -u 004_label.py')
os.system('python -u 005_inarow.py')
os.system('python -u 006_days_since_last_order.py')
os.system('python -u 007_timezone.py')
os.system('python -u 008_product_feature.py')
os.system('python -u 009_None.py')
os.system('python -u 010_streak.py')
os.system('python -u 011_replacement.py')
os.system('python -u 012_aisle_dep_cumsum.py')

os.system('nohup python -u 101_repeat_previous_ratio_T.py &')
os.system('python -u 102_orderspan_average.py')
os.system('nohup python -u 103_visit_time.py &')
os.system('python -u 104_organic.py')
os.system('python -u 105_delta_time.py')
os.system('python -u 108_order_size.py')
os.system('python -u 109_have_you_bought.py')
os.system('python -u 110_None.py')

os.system('nohup python -u 202_buy_time.py &')
os.system('python -u 203_cycle.py')
os.system('nohup python -u 205_co-occur.py &')
os.system('python -u 207_mean_pos_cart.py')
os.system('python -u 208_one-shot.py')
os.system('python -u 209_together.py')
os.system('nohup python -u 210_streak.py &')
os.system('nohup python -u 211_1to1.py &')
os.system('nohup python -u 212_withinN.py &')
os.system('nohup python -u 213_dow_diff.py &')
os.system('nohup python -u 214_first_order.py &')
os.system('nohup python -u 215_onb_diff.py &')

os.system('python -u 301_total_buy.py')
os.system('nohup python -u 302-1_reorderd_all.py &')
os.system('nohup python -u 303_last_order_date.py &')
os.system('nohup python -u 304_buy_item_inarow.py &')
os.system('nohup python -u 305_last_order_num.py &')
os.system('nohup python -u 306_mean_pos_cart.py &')
os.system('nohup python -u 307_timezone_dow.py &')
os.system('nohup python -u 308_timezone_dow.py &')
os.system('nohup python -u 309_order_ratio_by-chance.py &')
os.system('python -u 310_repeat_within_today.py')
os.system('python -u 312_cycle.py')
os.system('python -u 313_aisle_dep.py')
os.system('python -u 314_co-occur.py')
os.system('nohup python -u 315_streak.py &')
os.system('nohup python -u 316_replacement.py &')

os.system('python -u 401_how_many_come.py')



#==============================================================================
utils.end(__file__)
