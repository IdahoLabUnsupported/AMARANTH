# -*- coding: utf-8 -*-
import torch
#torch.backends.cudnn.benchmark = True

import sys
import time
sys.path.append("./Models")
import os

from Models.tsrnn import TSRNN, DPTrainableTSRNN
from Models.BigBirdSparse.SparseTransformerBB import TransformerBBSparse, TransformerBBFixed
from Models.transformer_base import Transformer_Base, DPTrainable

from Datasets.LondonSmartMeter.lsm_def import LondonSmartMeter
from Datasets.PJM_energy_datasets.aep_def import AEP #PJM AEP
from Datasets.PJM_energy_datasets.dayton_def import DAYTON
from Datasets.Spain_EW.spain_def import REE #Spain
<<<<<<< HEAD
<<<<<<< HEAD
from Datasets.CAISO.caiso_def import CAISO
from Datasets.AustinTX.austin_def import Austin

=======
>>>>>>> 626db60 (Added a little transformer for convienence)
=======
from Datasets.CAISO.caiso_def import CAISO
from Datasets.AustinTX.austin_def import Austin

>>>>>>> 6313c49 (Added Student-Teacher Model and SVD)

import copy
import random
#import matplotlib.pyplot as plt
#import numpy as np
import time
import timeit

import warnings
warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore",category=UserWarning)

attr_dset_smpl_rt = 24 #Samples per day. Spain, AEP: 24, London: 48 
param_dset_lookback_weeks = 5
param_dset_forecast = 48

param_dset_lookback = param_dset_lookback_weeks*7*attr_dset_smpl_rt - param_dset_forecast

#Transformer only parameters
param_trf_edim = 24
param_trf_heads = 4
param_trf_elyr = 4
param_trf_dlyr = 4
param_trf_ffdim = 256
param_trf_weather = False

#Bigbird only parameters
param_trf_bksz = 48
nl_bsz = param_trf_bksz
nu_bsz = param_trf_bksz

while ((param_dset_lookback%nl_bsz) != 0) and ((param_dset_lookback%nu_bsz) != 0):
    nl_bsz = nl_bsz - 1
    nu_bsz = nu_bsz + 1

if (param_dset_lookback % nl_bsz) == 0:
    param_trf_bksz = nl_bsz
else:
    param_trf_bksz = nu_bsz

assert (param_trf_bksz >= 24), "Computed block size too small"
assert (param_trf_bksz <= 64), "Computed block size too large"

print("BigBird block size autoset to: " + str(param_trf_bksz))

model = TransformerBBSparse(seq_len = param_dset_lookback,
                                out_seq_len = param_dset_forecast,
                                emb_dim = param_trf_edim,
                                n_heads = param_trf_heads,
                                n_enc_layers = param_trf_elyr,
                                n_dec_layers = param_trf_dlyr,
                                block_size=param_trf_bksz,
                                ffdim = param_trf_ffdim)

model.to('cuda:1')
#jmodel = torch.jit.script(model)

modelf = TransformerBBFixed(seq_len = param_dset_lookback,
                                out_seq_len = param_dset_forecast,
                                emb_dim = param_trf_edim,
                                n_heads = param_trf_heads,
                                n_enc_layers = param_trf_elyr,
                                n_dec_layers = param_trf_dlyr,
                                block_size=param_trf_bksz,
                                ffdim = param_trf_ffdim)

modelf.to('cuda:1')

modelt = TSRNN(smpl_rate = attr_dset_smpl_rt,
              pred_horz = param_dset_forecast,
              num_weeks = param_dset_lookback_weeks)
modelt.eval()
modelt.to('cuda:1')

inp1 = torch.rand((1,param_dset_lookback,1),device='cuda:1')
inp2 = torch.rand((1,param_dset_lookback,1),device='cuda:1')
inp3 = torch.rand((1,param_dset_lookback,1),device='cuda:1')
inp4 = torch.rand((1,param_dset_lookback,1),device='cuda:1')

# for i in range(30):
#     start = time.perf_counter()
#     modelt(inp)
#     end = time.perf_counter()
#     print("time: {0}".format(end-start))

print("trfbf: {0}".format(
min(
timeit.repeat(
"""
modelf(inp1)
modelf(inp2)
modelf(inp3)
modelf(inp4)""","""from __main__ import modelf, inp1, inp2, inp3, inp4""",repeat=10,number = 250) )/1000))
print("trfbb: {0}".format(
min(
timeit.repeat(
"""
model(inp1)
model(inp2)
model(inp3)
model(inp4)""","from __main__ import model, inp1, inp2, inp3, inp4",repeat=10,number = 250) )/1000))
print("tsrnn: {0}".format(
min(
timeit.repeat(
"""
modelt(inp1)
modelt(inp2)
modelt(inp3)
modelt(inp4)""","from __main__ import modelt, inp1, inp2, inp3, inp4",repeat=10,number = 250) )/1000))


# print("trfbf s: {0}".format(timeit.timeit(
# """
# modelf(inp1)
# modelf(inp1)
# modelf(inp1)
# modelf(inp1)""","""from __main__ import modelf, inp1, inp2, inp3, inp4""",number = 250)/1000))
# print("trfbb s: {0}".format(timeit.timeit(
# """
# model(inp1)
# model(inp1)
# model(inp1)
# model(inp1)""","from __main__ import model, inp1, inp2, inp3, inp4",number = 250)/1000))
# print("tsrnn s: {0}".format(timeit.timeit(
# """
# modelt(inp1)
# modelt(inp1)
# modelt(inp1)
# modelt(inp1)""","from __main__ import modelt, inp1, inp2, inp3, inp4",number = 250)/1000))