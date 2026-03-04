###############################################################################
# Copyright 2026, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
# Written by Bradley Marx 08/27/2025
#
# Functions and classes used to store data from different device channels
###############################################################################
import pandas as pd
import numpy as np
from datetime import datetime
import pickle as pk
import json
from DataStore import DataStore
# from FHMM import FHMM
# from HMM import HMM, HMM_MAD
from Preprocessing import Appliance, train_test_split, create_matrix

DStore = DataStore('house_1')
all_channels = [1, 3, 5, 6, 8, 12, 53, 38, 42, 25, 14, 10, 13, 51, 39, 36, 32, 22, 16, 11, 34, 47, 27, 7]
# all_channels = [1, 12, 13, 14]
select_channels = all_channels #[1, 3, 5, 6, 12]
# select_channels = [12, 5, 6]

DStore.create_store(all_channels)
# top_10 = DStore.select_top_k(10,'2013-08-01','2013-09-01')

combined = DStore.create_combined_df('2013-03-12 00:00:00', '2017-12-31 23:59:59', select_channels = select_channels, freq='1Min')

with open('combined.pkl', 'wb') as f:
    pk.dump(combined,f)

