import torch
import torch.nn as nn

from utils import *

class Encoder(nn.Module):
    def __init__(self, in_feat_dim, in_dynamic_rg_dim, in_static_rg_dim, time_steps=91, feature_dim=256, 
    head_num=4, max_dynamic_rg=16, max_static_rg=1400, k=4): # what is k?
        super().__init__()

        self.time_steps = time_steps                # T
        self.feature_dim = feature_dim              # D
        self.head_num = head_num                    # H
        self.max_dynamic_rg = max_dynamic_rg        # GD
        self.max_static_rg = max_static_rg          # GS
        assert feature_dim % head_num == 0      
        self.head_dim = int(feature_dim/head_num)   # d

        # layer A : input -> [B,A,T,in_feat_dim] / output -> [B,A,T,D]
        self.layer_A = nn.Sequential(nn.Linear(in_feat_dim,feature_dim), Permute4Batchnorm(0,3,1,2), 
                                        nn.BatchNorm2d(self.feature_dim), Permute4Batchnorm(0,2,3,1))
        # layer B : input -> [B,GD,T,in_dynamic_rg_dim] / output
