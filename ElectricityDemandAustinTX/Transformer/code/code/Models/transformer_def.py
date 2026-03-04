# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
from itf_trainable import Itrainable
from embedding import *
import time
from sparse_attn import BlockSparseMultiheadAttention
from fixed_attn import BlockSparseMheadAttnFixed
from nlattention import NonlinearAttention
from conv_attention import ConvAttnEncoderLayer, ConvAttnDecoderLayer
from cascade_pe import TransformerEncoder_CascadePE, TransformerDecoder_CascadePE

from torch import Tensor
from typing import Optional


class TBatchNorm(nn.Module):
    def __init__(self,num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
        super(TBatchNorm,self).__init__()
        self.bn = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats, device, dtype)
    
    def forward(self,x):
        x = x.permute(0,2,1)
        x = self.bn(x)
        x = x.permute(0,2,1)
        return x


class TransformerEncoderLayer_vis(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        super(TransformerEncoderLayer_vis,self).__init__(d_model, nhead, dim_feedforward,
                                                         dropout, activation,
                                                         layer_norm_eps, batch_first, norm_first,
                                                         device, dtype)
        self.attn_ = {}
        
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x, attn_map = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True)
        self.attn_['sa'] = attn_map.detach()
        return self.dropout1(x)


class TransformerDecoderLayer_vis(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        super(TransformerDecoderLayer_vis,self).__init__(d_model, nhead, dim_feedforward,
                                                         dropout, activation,
                                                         layer_norm_eps, batch_first, norm_first,
                                                         device, dtype)
        self.attn_ = {}
        
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x, attn_map = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True)
        self.attn_['sa'] = attn_map.detach()
        return self.dropout1(x)
    
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x, attn_map = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True)
        self.attn_['mha'] = attn_map.detach()
        return self.dropout2(x)


class Transformer_C(nn.Module):
    def __init__(self, seq_len=2560, out_seq_len=24, interval=1, inp_dim=1, emb_dim=64,\
                 n_heads=4, n_enc_layers=2, n_dec_layers=2, ffdim=128,
                 pe_drop = 0.1,
                 ffdrop = 0.5,
                 attn_drop = 0.2,  
                 pos_enc = CosineEmbedding,
                 enc_layer = ConvAttnEncoderLayer,
                 dec_layer = ConvAttnDecoderLayer):
        super(Transformer_C,self).__init__()
        self.seq_len = seq_len
        self.out_seq_len = out_seq_len
        self.emb_dim = emb_dim

        self.input_linear = nn.Linear(inp_dim,emb_dim)

        #self.pe = LearnablePositionalEncoding(emb_dim,dropout=0.1,max_len=seq_len)
        #self.pe = CosineEmbedding(emb_dim,max_len=seq_len,scale_factor=1)
        self.pe_scale = nn.Parameter(torch.tensor(1.0))
        
        if pos_enc.__name__ == 'TimestampCosineEmbedding' or 'TimestampLearnableEmbedding':
            self.tpe = pos_enc(emb_dim,interval,max_len=seq_len,scale_factor=self.pe_scale,dropout=pe_drop)
        else:
            self.pe = pos_enc(emb_dim,max_len=seq_len,scale_factor=self.pe_scale,dropout=pe_drop)
        
        drop_p = ffdrop
        if enc_layer.__name__ == 'ConvAttnEncoderLayer':
            self.trf_el = enc_layer(emb_dim,n_heads,12,ffdim,
                                                      activation=F.gelu,dropout=drop_p,
                                                      batch_first=True,norm_first=True)
        else:
            self.trf_el = enc_layer(emb_dim,n_heads,ffdim,
                                                      activation=F.gelu,dropout=drop_p,
                                                      batch_first=True,norm_first=True)
        
        #self.trf_el.self_attn = BlockSparseMultiheadAttention(emb_dim, n_heads, block_size,batch_first=True)
        self.trf_el.self_attn = NonlinearAttention(emb_dim, 512, n_heads,batch_first=True)
        
        if dec_layer.__name__ == 'ConvAttnDecoderLayer':
            self.trf_dl = dec_layer(emb_dim,n_heads,12,ffdim,
                                                      activation=F.gelu,dropout=drop_p,
                                                      batch_first=True,norm_first=True)
        else:    
            self.trf_dl = dec_layer(emb_dim,n_heads,ffdim,
                                                      activation=F.gelu,dropout=drop_p,
                                                      batch_first=True,norm_first=True)
        
        self.trf_dl.self_attn = NonlinearAttention(emb_dim, 512, n_heads,batch_first=True)
        self.trf_dl.multihead_attn = NonlinearAttention(emb_dim, 512, n_heads,batch_first=True)
        
        self.trf_el.norm1 = TBatchNorm(num_features=emb_dim) #nn.BatchNorm1d(num_features=seq_len)
        self.trf_el.norm2 = TBatchNorm(num_features=emb_dim) #nn.BatchNorm1d(num_features=seq_len)
        
        self.trf_dl.norm1 = TBatchNorm(num_features=emb_dim) #nn.BatchNorm1d(num_features=out_seq_len)
        self.trf_dl.norm2 = TBatchNorm(num_features=emb_dim)#nn.BatchNorm1d(num_features=out_seq_len)
        self.trf_dl.norm3 = TBatchNorm(num_features=emb_dim)#nn.BatchNorm1d(num_features=out_seq_len)
        
        self.trf_e = nn.TransformerEncoder(self.trf_el,num_layers=n_enc_layers)
        self.trf_d = nn.TransformerDecoder(self.trf_dl,num_layers=n_dec_layers)
        
        self.out = nn.Linear(emb_dim,inp_dim)
        # self.out = nn.Sequential(nn.Linear(emb_dim,ffdim),
        #                          nn.GELU(),
        #                          nn.Dropout(p = drop_p),
        #                          nn.Linear(ffdim,inp_dim))
        self.drop = nn.Dropout(p=drop_p)
        
        # self.dec_iv = nn.Parameter(torch.empty((1,out_seq_len,1)))
        # nn.init.normal_(self.dec_iv)
        
        self.dec_sa_mask = torch.zeros((2*self.out_seq_len,2*self.out_seq_len),dtype = torch.bool)
        #self.dec_sa_mask[self.out_seq_len:,:self.out_seq_len] = True
        self.dec_sa_mask[:self.out_seq_len,self.out_seq_len:] = True
        
    def forward(self,x,in_start_time,pred_start_time, pred_gt = None):
    #def forward(self,x, pred_gt = None):
        '''x: [batch dim, sequence length, variable dim]'''
        x1 = self.input_linear(x) # x1.shape: [batch, seq len, emb dim]
        #x1 =  self.pe(x1.permute(1,0,2)).permute(1,0,2)
        x1 = self.tpe(x1.permute(1,0,2),in_start_time).permute(1,0,2)
        mem = self.trf_e(x1)
        
        pred_task_iv = torch.zeros((x.shape[0],self.out_seq_len,1),device=mem.device)
        #pred_task_iv = self.dec_iv.broadcast_to((x.shape[0],self.out_seq_len,1))
        pred_task_iv = self.input_linear(pred_task_iv)
        pred_task_iv = self.tpe(pred_task_iv.permute(1,0,2),pred_start_time).permute(1,0,2)
        #pred_task_iv = self.pe(pred_task_iv.permute(1,0,2)).permute(1,0,2)
        
        #Denoising task
        dn_task_iv = torch.tensor((),device=mem.device)
        dec_samask = None
        if pred_gt is not None:
            dn_task_iv = pred_gt
            dn_task_iv = self.input_linear(pred_gt)
            #dn_task_iv = self.pe(dn_task_iv.permute(1,0,2)).permute(1,0,2)
            dn_task_iv = self.tpe(dn_task_iv.permute(1,0,2),pred_start_time).permute(1,0,2)
            dn_task_iv[:,:,-1] = dn_task_iv[:,:,-1] + 1.0
            
            dec_samask = self.dec_sa_mask.to(mem.device)
            
        dec_in = torch.cat((pred_task_iv,dn_task_iv),dim=1)

        x2 = self.trf_d(dec_in,mem,tgt_mask=dec_samask)
        o1 = self.out(x2)

        return o1


class Transformer_CCas(Transformer_C):
    def __init__(self, seq_len=2560, out_seq_len=24, inp_dim=1, emb_dim=64,\
                 n_heads=4, n_enc_layers=2, n_dec_layers=2, ffdim=128, drop_p = 0.1,
                pos_enc = CosineEmbedding,
                enc_layer = ConvAttnEncoderLayer,
                dec_layer = ConvAttnDecoderLayer):
        super(Transformer_CCas,self).__init__(seq_len, out_seq_len, inp_dim, emb_dim,
                                              n_heads, n_enc_layers, n_dec_layers, ffdim,
                                              drop_p, pos_enc, enc_layer, dec_layer)
        self.trf_e = TransformerEncoder_CascadePE(self.trf_el, n_enc_layers)
        self.trf_d = TransformerDecoder_CascadePE(self.trf_dl, n_dec_layers)

        
    def forward(self,x,in_start_time,pred_start_time, pred_gt = None):
    #def forward(self,x, pred_gt = None):
        '''x: [batch dim, sequence length, variable dim]'''
        x1 = self.input_linear(x) # x1.shape: [batch, seq len, emb dim]
        #x1 =  self.pe(x1.permute(1,0,2)).permute(1,0,2)
        pe = self.tpe.getPE(x1.permute(1,0,2),in_start_time).permute(1,0,2)
        #x1 = self.tpe(x1.permute(1,0,2),in_start_time).permute(1,0,2)
        mem = self.trf_e(x1,pe)
        
        pred_task_iv = torch.zeros((x.shape[0],self.out_seq_len,1),device=mem.device)
        #pred_task_iv = self.dec_iv.broadcast_to((x.shape[0],self.out_seq_len,1))
        pred_task_iv = self.input_linear(pred_task_iv)
        dpe = self.tpe.getPE(pred_task_iv.permute(1,0,2),pred_start_time).permute(1,0,2)
        #pred_task_iv = self.tpe(pred_task_iv.permute(1,0,2),pred_start_time).permute(1,0,2)
        #pred_task_iv = self.pe(pred_task_iv.permute(1,0,2)).permute(1,0,2)
        
        #Denoising task
        dn_task_iv = torch.tensor((),device=mem.device)
        dec_samask = None
        if pred_gt is not None:
            dn_task_iv = pred_gt
            dn_task_iv = self.input_linear(pred_gt)
            #dn_task_iv = self.pe(dn_task_iv.permute(1,0,2)).permute(1,0,2)
            #dn_task_iv = self.tpe(dn_task_iv.permute(1,0,2),pred_start_time).permute(1,0,2)
            dpe = torch.cat((dpe,dpe),dim=1)
            dn_task_iv[:,:,-1] = dn_task_iv[:,:,-1] + 1.0
            
            dec_samask = self.dec_sa_mask.to(mem.device)
            
        dec_in = torch.cat((pred_task_iv,dn_task_iv),dim=1)

        x2 = self.trf_d(dec_in,dpe,mem,tgt_mask=dec_samask)
        o1 = self.out(x2)

        return o1

class Transformer_EncoderOnly(nn.Module):
    def __init__(self, seq_len=2560, out_seq_len=24, inp_dim=1, emb_dim=64,\
                 n_heads=4, n_enc_layers=6, block_size=40,\
                ffdim=128, drop_p = 0.1):
        super(Transformer_EncoderOnly,self).__init__()
        self.seq_len = seq_len
        self.out_seq_len = out_seq_len
        self.emb_dim = emb_dim
        #self.in_norm = nn.BatchNorm1d(seq_len,affine=False)
        self.input_linear = nn.Linear(inp_dim,emb_dim)
        #self.input_pool = nn.AvgPool1d(kernel_size=5,stride=5)
        #self.pe = LearnablePositionalEncoding(emb_dim,dropout=0.1,max_len=seq_len)
        #self.pe = CosineEmbedding(emb_dim,max_len=seq_len,scale_factor=0.1)
        self.pe_scale = nn.Parameter(torch.tensor(1.))
        #self.tpe = TimestampCosineEmbedding(emb_dim,1,max_len=seq_len,scale_factor=self.pe_scale)
        self.tpe = TimestampLearnableEmbedding(emb_dim,1,max_len=seq_len,scale_factor=self.pe_scale)
        
        # self.trf_el = nn.TransformerEncoderLayer(emb_dim,n_heads,ffdim,
        #                                           activation=F.gelu,dropout=drop_p,
        #                                           batch_first=True,norm_first=True)
        self.trf_el = ConvAttnEncoderLayer(emb_dim,n_heads,12,ffdim,
                                                 activation=F.gelu,dropout=drop_p,
                                                 batch_first=True,norm_first=True)
        
        #self.trf_el.self_attn = BlockSparseMultiheadAttention(emb_dim, n_heads, block_size,batch_first=True)
        
        self.trf_el.norm1 = TBatchNorm(num_features=emb_dim) #nn.BatchNorm1d(num_features=seq_len)
        self.trf_el.norm2 = TBatchNorm(num_features=emb_dim) #nn.BatchNorm1d(num_features=seq_len)
        
        
        self.trf_e = nn.TransformerEncoder(self.trf_el,num_layers=n_enc_layers)
        
        self.out = nn.Linear(seq_len*emb_dim,out_seq_len)
        self.drop = nn.Dropout(p=drop_p)
          
        self.dec_sa_mask = torch.zeros((2*self.out_seq_len,2*self.out_seq_len),dtype = torch.bool)
        self.dec_sa_mask[self.out_seq_len:,:self.out_seq_len] = True
        self.dec_sa_mask[:self.out_seq_len,self.out_seq_len:] = True
        
    def forward(self,x,in_start_time,pred_start_time):
        '''x: [batch dim, sequence length, variable dim]'''
        x1 = self.input_linear(x) # x1.shape: [batch, seq len, emb dim]
        x1 = self.tpe(x1.permute(1,0,2),in_start_time).permute(1,0,2)
        mem = self.trf_e(x1)

        o1 = self.out(mem.view((x.shape[0],self.seq_len*self.emb_dim,-1)).permute(0,2,1)).permute(0,2,1)

        return o1

class Transformer_(nn.Module):
    def __init__(self, seq_len=2560, out_seq_len=24, inp_dim=1, emb_dim=64,\
                 n_heads=4, n_enc_layers=2, n_dec_layers=2, block_size=40,\
                ffdim=128, drop_p = 0.1):
        super(Transformer_,self).__init__()
        self.seq_len = seq_len
        self.out_seq_len = out_seq_len
        self.emb_dim = emb_dim
        #self.in_norm = nn.BatchNorm1d(seq_len,affine=False)
        self.input_linear = nn.Linear(inp_dim,emb_dim)
        #self.input_pool = nn.AvgPool1d(kernel_size=5,stride=5)
        #self.pe = LearnablePositionalEncoding(emb_dim,dropout=0.1,max_len=seq_len)
        self.pe = CosineEmbedding(emb_dim,max_len=seq_len,scale_factor=0.1)
        #self.tpe = TimestampCosineEmbedding(emb_dim,1,max_len=seq_len,scale_factor=self.pe_scale)
        
        self.trf_el = nn.TransformerEncoderLayer(emb_dim,n_heads,ffdim,
                                                 activation=F.gelu,dropout=drop_p,
                                                 batch_first=True,norm_first=False)
        
        self.trf_dl = nn.TransformerDecoderLayer(emb_dim,n_heads,ffdim,
                                                  activation=F.gelu,dropout=drop_p,
                                                  batch_first=True,norm_first=False)

        
        self.trf_el.norm1 = TBatchNorm(num_features=emb_dim) #nn.BatchNorm1d(num_features=seq_len)
        self.trf_el.norm2 = TBatchNorm(num_features=emb_dim) #nn.BatchNorm1d(num_features=seq_len)
        
        self.trf_dl.norm1 = TBatchNorm(num_features=emb_dim) #nn.BatchNorm1d(num_features=out_seq_len)
        self.trf_dl.norm2 = TBatchNorm(num_features=emb_dim)#nn.BatchNorm1d(num_features=out_seq_len)
        self.trf_dl.norm3 = TBatchNorm(num_features=emb_dim)#nn.BatchNorm1d(num_features=out_seq_len)
        
        self.trf_e = nn.TransformerEncoder(self.trf_el,num_layers=n_enc_layers)
        self.trf_d = nn.TransformerDecoder(self.trf_dl,num_layers=n_dec_layers)
        
        self.out = nn.Linear(emb_dim,inp_dim)
        
    def forward(self,x):
        '''x: [batch dim, sequence length, variable dim]'''
        x1 = self.input_linear(x) # x1.shape: [batch, seq len, emb dim]
        #x1 = self.tpe(x1.permute(1,0,2),in_start_time).permute(1,0,2)
        x1 = self.pe(x1.permute(1,0,2)).permute(1,0,2)
        mem = self.trf_e(x1)
        
        #Denoising task
        dn_task_iv = torch.tensor((),device=mem.device)
        dec_samask = None
        
        pred_task_iv = torch.zeros((x.shape[0],self.out_seq_len,1),device=mem.device)
        pred_task_iv = self.input_linear(pred_task_iv)
        pred_task_iv = self.pe(pred_task_iv.permute(1,0,2)).permute(1,0,2)
        
        dec_in = torch.cat((pred_task_iv,dn_task_iv),dim=1)
        
         # x2 = self.trf_d(self.dec_tar.broadcast_to(x.shape[0],self.out_seq_len,self.emb_dim)\
         #                 ,mem)
        x2 = self.trf_d(dec_in,mem,tgt_mask=dec_samask)
        o1 = self.out(x2)
        return o1

class Transformer_SparseDec_E(nn.Module):
    def __init__(self, seq_len=2560, out_seq_len=24, interval=1, inp_dim=1, emb_dim=64,\
                 n_heads=4, n_enc_layers=2, n_dec_layers=2, block_size=40,\
                ffdim=128, drop_p = 0.1):
        super(Transformer_SparseDec_E,self).__init__()
        self.seq_len = seq_len
        self.out_seq_len = out_seq_len
        self.emb_dim = emb_dim
        #self.in_norm = nn.BatchNorm1d(seq_len,affine=False)
        self.input_linear = nn.Linear(inp_dim,emb_dim)
        #self.input_pool = nn.AvgPool1d(kernel_size=5,stride=5)
        self.pe = LearnablePositionalEncoding(emb_dim,dropout=0.1,max_len=seq_len)
        #self.pe = CosineEmbedding(emb_dim,max_len=seq_len,scale_factor=0.1)
        #self.tpe = TimestampCosineEmbedding(emb_dim, interval, max_len = seq_len)
        
        self.trf_el = nn.TransformerEncoderLayer(emb_dim,n_heads,ffdim,
                                                 activation=F.gelu,dropout=drop_p,
                                                 batch_first=True,norm_first=True)
        self.trf_el.self_attn = BlockSparseMultiheadAttention(emb_dim, n_heads, block_size,batch_first=True)
        #self.trf_el.self_attn = BlockSparseMheadAttnFixed(emb_dim, n_heads, block_size,batch_first=True)
        
        self.trf_dl = nn.TransformerDecoderLayer(emb_dim,n_heads,ffdim,
                                                 activation=F.gelu,dropout=drop_p,
                                                 batch_first=True,norm_first=True)
        
        self.trf_el.norm1 = TBatchNorm(num_features=emb_dim)
        self.trf_el.norm1 = TBatchNorm(num_features=emb_dim)
        self.trf_el.norm2 = TBatchNorm(num_features=emb_dim)
        
        self.trf_dl.norm1 = TBatchNorm(num_features=emb_dim)
        self.trf_dl.norm2 = TBatchNorm(num_features=emb_dim)
        self.trf_dl.norm3 = TBatchNorm(num_features=emb_dim)
        
        self.trf_e = nn.TransformerEncoder(self.trf_el,num_layers=n_enc_layers)
        self.trf_d = nn.TransformerDecoder(self.trf_dl,num_layers=n_dec_layers)
        
        #self.out = nn.Linear(emb_dim,inp_dim)
        self.out = nn.Linear(emb_dim,1)
        self.drop = nn.Dropout(p=drop_p)
        
        # self.dec_tar = torch.nn.Parameter(torch.zeros((out_seq_len,emb_dim)))
        # nn.init.normal_(self.dec_tar)
        self.aux_out = nn.Linear(emb_dim,1)#inp_dim)
        #self.reduce = nn.Linear(seq_len,out_seq_len)
        self.aux_in = nn.Linear(1,emb_dim)#inp_dim,emb_dim)
        
        self.expand = nn.Linear(seq_len,5120)
        self.compress = nn.Linear(5120,out_seq_len)
        
    def forward(self,x,in_start_time,pred_start_time):
        '''x: [batch dim, sequence length, variable dim]'''
        x1 = self.input_linear(x) # x1.shape: [batch, seq len, emb dim]
        x1 = self.pe(x1.permute(1,0,2)).permute(1,0,2)
        #x1 = self.tpe(x1.permute(1,0,2),in_start_time).permute(1,0,2)
        mem = self.trf_e(x1)
        
        dec_in = self.aux_out(mem)
        dec_in = F.gelu(self.expand(dec_in.view((-1,self.seq_len))))
        dec_in = self.drop(dec_in)
        dec_in = self.compress(dec_in)
        #dec_in = F.gelu(self.reduce(dec_in.view((-1,self.seq_len))))
        
        dec_in = self.aux_in(dec_in.unsqueeze(-1))

        # x2 = self.trf_d(self.dec_tar.broadcast_to(x.shape[0],self.out_seq_len,self.emb_dim)\
        #                 ,mem)
        x2 = self.trf_d(self.pe(dec_in.permute(1,0,2))\
                        .permute(1,0,2),mem)
        # x2 = self.trf_d(self.tpe(dec_in.permute(1,0,2),pred_start_time)\
        #                 .permute(1,0,2),mem)

        o1 = self.out(x2)

        return o1

class Transformer_SparseDec_EF(nn.Module):
    def __init__(self, seq_len=2560, out_seq_len=24, interval=1, inp_dim=1, emb_dim=64,\
                 n_heads=4, n_enc_layers=2, n_dec_layers=2, block_size=40,\
                ffdim=128, drop_p = 0.1):
        super(Transformer_SparseDec_EF,self).__init__()
        self.seq_len = seq_len
        self.out_seq_len = out_seq_len
        self.emb_dim = emb_dim
        #self.in_norm = nn.BatchNorm1d(seq_len,affine=False)
        self.input_linear = nn.Linear(inp_dim,emb_dim)
        #self.input_pool = nn.AvgPool1d(kernel_size=5,stride=5)
        self.pe = LearnablePositionalEncoding(emb_dim,dropout=0.1,max_len=seq_len)
        #self.pe = CosineEmbedding(emb_dim,max_len=seq_len,scale_factor=0.1)
        #self.tpe = TimestampCosineEmbedding(emb_dim, interval, max_len = seq_len)
        
        self.trf_el = nn.TransformerEncoderLayer(emb_dim,n_heads,ffdim,
                                                 activation=F.gelu,dropout=drop_p,
                                                 batch_first=True,norm_first=True)
        #self.trf_el.self_attn = BlockSparseMultiheadAttention(emb_dim, n_heads, block_size,batch_first=True)
        self.trf_el.self_attn = BlockSparseMheadAttnFixed(emb_dim, n_heads, block_size,batch_first=True)
        
        self.trf_dl = nn.TransformerDecoderLayer(emb_dim,n_heads,ffdim,
                                                 activation=F.gelu,dropout=drop_p,
                                                 batch_first=True,norm_first=True)
        
        self.trf_el.norm1 = TBatchNorm(num_features=emb_dim)
        self.trf_el.norm1 = TBatchNorm(num_features=emb_dim)
        self.trf_el.norm2 = TBatchNorm(num_features=emb_dim)
        
        self.trf_dl.norm1 = TBatchNorm(num_features=emb_dim)
        self.trf_dl.norm2 = TBatchNorm(num_features=emb_dim)
        self.trf_dl.norm3 = TBatchNorm(num_features=emb_dim)
        
        self.trf_e = nn.TransformerEncoder(self.trf_el,num_layers=n_enc_layers)
        self.trf_d = nn.TransformerDecoder(self.trf_dl,num_layers=n_dec_layers)
        
        self.out = nn.Linear(emb_dim,inp_dim)
        self.drop = nn.Dropout(p=drop_p)
        
        # self.dec_tar = torch.nn.Parameter(torch.zeros((out_seq_len,emb_dim)))
        # nn.init.normal_(self.dec_tar)
        self.aux_out = nn.Linear(emb_dim,inp_dim)
        #self.reduce = nn.Linear(seq_len,out_seq_len)
        self.aux_in = nn.Linear(inp_dim,emb_dim)
        
        self.expand = nn.Linear(seq_len,2560)
        self.compress = nn.Linear(2560,out_seq_len)
        
    def forward(self,x,in_start_time,pred_start_time):
        '''x: [batch dim, sequence length, variable dim]'''
        x1 = self.input_linear(x) # x1.shape: [batch, seq len, emb dim]
        x1 = self.pe(x1.permute(1,0,2)).permute(1,0,2)
        #x1 = self.tpe(x1.permute(1,0,2),in_start_time).permute(1,0,2)
        mem = self.trf_e(x1)
        
        dec_in = self.aux_out(mem)
        dec_in = F.gelu(self.expand(dec_in.view((-1,self.seq_len))))
        dec_in = self.drop(dec_in)
        dec_in = self.compress(dec_in)
        #dec_in = F.gelu(self.reduce(dec_in.view((-1,self.seq_len))))
        
        dec_in = self.aux_in(dec_in.unsqueeze(-1))

        # x2 = self.trf_d(self.dec_tar.broadcast_to(x.shape[0],self.out_seq_len,self.emb_dim)\
        #                 ,mem)
        x2 = self.trf_d(self.pe(dec_in.permute(1,0,2))\
                        .permute(1,0,2),mem)
        # x2 = self.trf_d(self.tpe(dec_in.permute(1,0,2),pred_start_time)\
        #                 .permute(1,0,2),mem)

        o1 = self.out(x2)

        return o1
    
    def train_epoch(self,
                    loader: torch.utils.data.DataLoader,
                    scaler: torch.cuda.amp.GradScaler,
                    device: torch.device,
                    epoch: int,
                    optimizer):
        self.train()
        losses = np.zeros(len(loader))
        ep_st = time.time()
        optimizer.zero_grad(set_to_none = False) ###
        for batch_idx, (src_, tar_, in_tstp, pr_tstp) in enumerate(loader):
        
            tar = tar_
            src = src_.nan_to_num()
            
            #noisy_tar = tar + 0.2 * torch.randn(tar.shape)
            src = src.to(device)
            tar = tar.to(device)
            #noisy_tar = noisy_tar.to(device)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                out = self(src, in_tstp, pr_tstp)
                
                loss_fn = torch.nn.MSELoss(reduction='mean')(out[~tar.isnan()],tar[~tar.isnan()])
                loss = loss_fn(out,tar)
                    
            optimizer.zero_grad()
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()


            losses[batch_idx] = loss.detach().item()

        ep_et = time.time()
        print("Epoch:", epoch,"\t Loss (\u03BC \u00B1 s): ",np.mean(losses),
              "\u00B1",np.std(losses,ddof=1), '\t Epoch time:',ep_et-ep_st)

        return np.mean(losses), np.std(losses,ddof=1)