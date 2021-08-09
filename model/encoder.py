import torch
import torch.nn as nn

from model.utils import *

from collections import OrderedDict

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
        self.k = k                                  # k

        # layer A : input -> [A,T,in_feat_dim] / output -> [A,T,D]
        self.layer_A = nn.Sequential(nn.Linear(in_feat_dim,feature_dim), nn.ReLU(), Permute4Batchnorm((0,2,1)),
                            nn.BatchNorm1d(feature_dim), Permute4Batchnorm((0,2,1)))
        # layer B : input -> [GD,T,in_dynamic_rg_dim] / output -> [GD,T,D]
        self.layer_B = nn.Sequential(nn.Linear(in_dynamic_rg_dim,feature_dim), nn.ReLU(), Permute4Batchnorm((0,2,1)),
                            nn.BatchNorm1d(feature_dim), Permute4Batchnorm((0,2,1)))
        # layer C : input -> [GD,T,in_dynamic_rg_dim] / output -> [GD,T,D]
        self.layer_C = nn.Sequential(nn.Linear(in_static_rg_dim,feature_dim), nn.ReLU(), Permute4Batchnorm((0,2,1)),
                            nn.BatchNorm1d(feature_dim), Permute4Batchnorm((0,2,1)))
        # layer D,E,F,G,H,I : input -> [A,T,D] / outpu -> [A,T,D]
        self.layer_D = SelfAttLayer(self.time_steps,self.feature_dim,self.head_num,self.k,across_time=True)
        self.layer_E = SelfAttLayer(self.time_steps,self.feature_dim,self.head_num,self.k,across_time=False)
        self.layer_F = SelfAttLayer(self.time_steps,self.feature_dim,self.head_num,self.k,across_time=True)
        self.layer_G = SelfAttLayer(self.time_steps,self.feature_dim,self.head_num,self.k,across_time=False)
        self.layer_H = SelfAttLayer(self.time_steps,self.feature_dim,self.head_num,self.k,across_time=True)
        self.layer_I = SelfAttLayer(self.time_steps,self.feature_dim,self.head_num,self.k,across_time=False)

        self.layer_J = CrossAttLayer(self.time_steps,self.feature_dim,self.head_num,self.k)
        self.layer_K = CrossAttLayer(self.time_steps,self.feature_dim,self.head_num,self.k)

        self.layer_L = SelfAttLayer(self.time_steps,self.feature_dim,self.head_num,self.k,across_time=True)
        self.layer_M = SelfAttLayer(self.time_steps,self.feature_dim,self.head_num,self.k,across_time=False)

        self.layer_N = CrossAttLayer(self.time_steps,self.feature_dim,self.head_num,self.k)
        self.layer_O = CrossAttLayer(self.time_steps,self.feature_dim,self.head_num,self.k)

        self.layer_P = SelfAttLayer(self.time_steps,self.feature_dim,self.head_num,self.k,across_time=True)
        self.layer_Q = SelfAttLayer(self.time_steps,self.feature_dim,self.head_num,self.k,across_time=False)
        

    def forward(self, state_feat, agent_batch_mask, road_feat, traffic_light_feat):
        A_ = self.layer_A(state_feat)
        B_ = self.layer_B(traffic_light_feat)
        C_ = self.layer_C(road_feat)
        
        D_,_,_,_ = self.layer_D(A_,agent_batch_mask)
        E_,_,_,_ = self.layer_E(D_,agent_batch_mask)
        F_,_,_,_ = self.layer_F(E_,agent_batch_mask)
        G_,_,_,_ = self.layer_G(F_,agent_batch_mask)
        H_,_,_,_ = self.layer_H(G_,agent_batch_mask)
        I_,_,_,_ = self.layer_I(H_,agent_batch_mask)

        J_,_,_,_ = self.layer_J(I_,C_)
        K_,_,_,_ = self.layer_K(J_,B_)

        L_,_,_,_ = self.layer_L(K_,agent_batch_mask)
        M_,_,_,_ = self.layer_M(L_,agent_batch_mask)

        N_,_,_,_ = self.layer_N(M_,C_)
        O_,_,_,_ = self.layer_O(N_,B_)

        P_,_,_,_ = self.layer_P(O_,agent_batch_mask)
        Q_,_,_,_ = self.layer_Q(P_,agent_batch_mask)

        return Q_

class SelfAttLayer(nn.Module):
    def __init__(self, time_steps=91, feature_dim=256, head_num=4, k=4, across_time=True):
        super().__init__()

        self.viewmodule_ = View((-1,time_steps,head_num, int(feature_dim/head_num)))
        self.layer_K_ = nn.Sequential(nn.Linear(feature_dim,feature_dim), nn.ReLU(), self.viewmodule_)
        self.layer_V_ = nn.Sequential(nn.Linear(feature_dim,feature_dim), nn.ReLU(), self.viewmodule_)
        self.layer_Q0_ = nn.Sequential(nn.Linear(feature_dim,feature_dim), nn.ReLU(), self.viewmodule_)
        self.layer_Q_ = ScaleLayer(int(feature_dim/head_num))

        self.scale = torch.sqrt(torch.FloatTensor([head_num]))

        self.layer_Y2_ = nn.Sequential(View((-1,time_steps,feature_dim)), nn.Linear(feature_dim,feature_dim), nn.ReLU())
        self.layer_F1_ = nn.Sequential(nn.Linear(feature_dim,k*feature_dim), nn.ReLU())
        self.layer_F2_ = nn.Sequential(nn.Linear(k*feature_dim,feature_dim), nn.ReLU())
        self.layer_Z_ = nn.LayerNorm(feature_dim)

        self.across_time = across_time

    def forward(self, x, batch_mask, padding_mask=None, hidden_mask=None):
        K = self.layer_K_(x)
        V = self.layer_V_(x)
        Q0 = self.layer_Q0_(x)
        Q = self.layer_Q_(Q0)    # Q,K,V -> [A,T,H,d]

        if self.across_time:
            Q, K, V = Q.permute(0,2,1,3), K.permute(0,2,1,3), V.permute(0,2,1,3)    # Q,K,V -> [A,H,T,d]
            energy = torch.matmul(Q,K.permute(0,1,3,2)) / self.scale                               # [A,H,T,T]
            attention = torch.softmax(energy, dim=-1)                               # [A,H,T,T]
            Y1_ = torch.matmul(attention, V)                                        # [A,H,T,d]
            Y1_ = Y1_.permute(0,2,1,3).contiguous()                                 # [A,T,H,d]

        else:
            Q, K, V = Q.permute(1,2,0,3), K.permute(1,2,0,3), V.permute(1,2,0,3)    # Q,K,V -> [T,H,A,d]
            energy = torch.matmul(Q,K.permute(0,1,3,2)) / self.scale                               # [T,H,A,A]
            attention = torch.softmax(energy, dim=-1)                               # [T,H,A,A]

            if batch_mask is not None:                                              # batch_mask -> [A,A]
                energy = energy.masked_fill(batch_mask==0, -1e10)   # 0 for ignoring attention

            Y1_ = torch.matmul(attention, V)                                        # [T,H,A,d]
            Y1_ = Y1_.permute(2,0,1,3).contiguous()                                 # [A,T,H,d]
        

        Y2_ = self.layer_Y2_(Y1_)
        S_ = Y2_ + x
        F1_ = self.layer_F1_(S_)
        F2_ = self.layer_F2_(F1_)
        Z_ = self.layer_Z_(F2_)

        return Z_, Q, K, V # -> [A,T,D], [A,T,H,d]*3

class CrossAttLayer(nn.Module):
    def __init__(self, time_steps=91, feature_dim=256, head_num=4, k=4):
        super().__init__()

        self.viewmodule_ = View((-1,time_steps,head_num, int(feature_dim/head_num)))
        self.layer_K_ = nn.Sequential(nn.Linear(feature_dim,feature_dim), nn.ReLU(), self.viewmodule_)
        self.layer_V_ = nn.Sequential(nn.Linear(feature_dim,feature_dim), nn.ReLU(), self.viewmodule_)
        self.layer_Q0_ = nn.Sequential(nn.Linear(feature_dim,feature_dim), nn.ReLU(), self.viewmodule_)
        self.layer_Q_ = ScaleLayer(int(feature_dim/head_num))

        self.scale = torch.sqrt(torch.FloatTensor([head_num]))

        self.layer_Y2_ = nn.Sequential(View((-1,time_steps,feature_dim)), nn.Linear(feature_dim,feature_dim), nn.ReLU())
        self.layer_F1_ = nn.Sequential(nn.Linear(feature_dim,k*feature_dim), nn.ReLU())
        self.layer_F2_ = nn.Sequential(nn.Linear(k*feature_dim,feature_dim), nn.ReLU())
        self.layer_Z_ = nn.LayerNorm(feature_dim)

    def forward(self, agent, rg): # agent -> [A,T,D] / rg -> [G,T,D]
        K = self.layer_K_(rg)                                                       # [G,T,H,d]
        V = self.layer_V_(rg)                                                       # [G,T,H,d]
        Q0 = self.layer_Q0_(agent)
        Q = self.layer_Q_(Q0)                                                       # [A,T,H,d]

        Q, K, V = Q.permute(1,2,0,3), K.permute(1,2,0,3), V.permute(1,2,0,3)    # Q -> [T,H,A,d] / K,V -> [T,H,G,d]
        energy = torch.matmul(Q,K.permute(0,1,3,2)) / self.scale                               # [T,H,A,G]
        attention = torch.softmax(energy, dim=-1)                               # [T,H,A,G]

        Y1_ = torch.matmul(attention, V)                                        # [T,H,A,d]
        Y1_ = Y1_.permute(2,0,1,3).contiguous()                                 # [A,T,H,d]

        Y2_ = self.layer_Y2_(Y1_)
        S_ = Y2_ + agent
        F1_ = self.layer_F1_(S_)
        F2_ = self.layer_F2_(F1_)
        Z_ = self.layer_Z_(F2_)

        return Z_, Q, K, V # -> [A,T,D], [G,T,H,d], [A,T,H,d]*2

class ScaleLayer(nn.Module):

   def __init__(self, shape, init_value=1e-3):
       super().__init__()
       self.scale = nn.Parameter(torch.FloatTensor(shape).fill_(init_value))

   def forward(self, input):
       return input * self.scale