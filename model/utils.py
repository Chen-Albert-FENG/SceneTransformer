import torch
import torch.nn as nn

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class Permute4Batchnorm(nn.Module):
    def __init__(self,order):
        super(Permute4Batchnorm, self).__init__()
        self.order = order
    
    def forward(self, x):
        assert len(self.order) == len(x.shape)
        return x.permute(self.order)

class ScaleLayer(nn.Module):

   def __init__(self, shape, init_value=1e-3):
       super().__init__()
       self.scale = nn.Parameter(torch.FloatTensor(shape).fill_(init_value))

   def forward(self, input):
       return input * self.scale

def init_xavier_glorot(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

class SelfAttLayer_Enc(nn.Module):
    def __init__(self, device, time_steps=91, feature_dim=256, head_num=4, k=4, across_time=True):
        super().__init__()
        self.device = device
        self.across_time = across_time
        self.time_steps = time_steps
        self.feature_dim = feature_dim
        self.head_num = head_num
        self.k = k

        self.layer_X_ = nn.LayerNorm(feature_dim)
        self.layer_att_ = nn.MultiheadAttention(embed_dim=feature_dim,num_heads=head_num,add_zero_attn=True)
        self.layer_F1_ = nn.Sequential(nn.Linear(feature_dim,feature_dim), nn.ReLU())
        self.layer_F1_.apply(init_xavier_glorot)
        #self.layer_F2_ = nn.Sequential(nn.Linear(k*feature_dim,feature_dim), nn.ReLU())
        self.layer_Z_ = nn.LayerNorm(feature_dim)

    def forward(self, x, batch_mask, padding_mask=None, hidden_mask=None):
        A,T,D = x.shape
        assert (T==self.time_steps and D==self.feature_dim)
        A_,A__ = batch_mask.shape
        assert (A==A_ and A==A__)
        A___,T_ = padding_mask.shape
        assert (A==A___ and T==T_)

        x_ = self.layer_X_(x) # [A,T,D]

        if self.across_time:
            q_ = x_.permute(1,0,2)                       # [L,N,E] : [A,T,D]->[T,A,D]
            k,v = x_.permute(1,0,2), x_.permute(1,0,2)                   # [S,N,E] : [T,A,D]

            key_padding_mask = padding_mask             # [N,S] : [A,T]
            attn_mask = None  
            # att_output : [L,N,E] : [T,A,D]
            att_output, _ = self.layer_att_(q_,k,v,key_padding_mask=key_padding_mask,attn_mask=attn_mask)
            # att_output : [A,T,D]
            att_output = att_output.permute(1,0,2)
        else:
            q_ = x_                                      # [L,N,E] = [A,T,D]
            k, v = x_, x_                 # [S,N,E] = [A,T,D]

            key_padding_mask = padding_mask.permute(1,0)# [N,S] = [T,A]
            attn_mask = batch_mask                      # [L,S] = [A,A]
            # att_output : [L,N,E] : [A,T,D]
            att_output, _ = self.layer_att_(q_,k,v,key_padding_mask=key_padding_mask,attn_mask=attn_mask)
        S_ = att_output + x
        F1_ = self.layer_F1_(S_)
        #F2_ = self.layer_F2_(F1_)
        Z_ = self.layer_Z_(F1_)

        return Z_

class SelfAttLayer_Dec(nn.Module):
    def __init__(self, device, time_steps=91, feature_dim=256, head_num=4, k=4, across_time=True):
        super().__init__()
        self.device = device
        self.across_time = across_time
        self.time_steps = time_steps
        self.feature_dim = feature_dim
        self.head_num = head_num
        self.k = k

        self.layer_X_ = nn.LayerNorm(feature_dim)
        self.layer_att_ = nn.MultiheadAttention(embed_dim=feature_dim,num_heads=head_num,add_zero_attn=True)
        self.layer_F1_ = nn.Sequential(nn.Linear(feature_dim,feature_dim), nn.ReLU())
        self.layer_F1_.apply(init_xavier_glorot)
        #self.layer_F2_ = nn.Sequential(nn.Linear(k*feature_dim,feature_dim), nn.ReLU())
        self.layer_Z_ = nn.LayerNorm(feature_dim)

    def forward(self, x, batch_mask, padding_mask=None, hidden_mask=None):
        F,A,T,D = x.shape
        assert (T==self.time_steps and D==self.feature_dim)
        A_,A__ = batch_mask.shape
        assert (A==A_ and A==A__)
        A___,T_ = padding_mask.shape
        assert (A==A___ and T==T_)

        x_ = self.layer_X_(x)                                           # [F,A,T,D]

        if self.across_time:
            q = x_.reshape((-1,T,D)).permute(1,0,2)                     # [L,N,E] : [F,A,T,D]->[F*A,T,D]->[T,F*A,D]
            k,v = q, q                                   # [S,N,E] : [T,F*A,D]

            key_padding_mask = padding_mask.repeat(F,1)                 # [N,S] : [A*F,T]
            attn_mask = None  
            # att_output : [L,N,E] : [T,F*A,D]
            att_output, _ = self.layer_att_(q,k,v,key_padding_mask=key_padding_mask,attn_mask=attn_mask)
            # att_output : [F,A,T,D]
            att_output = att_output.reshape((T,F,A,D)).permute(1,2,0,3)
        else:
            q = x_.permute(0,2,1,3).reshape((-1,A,D)).permute(1,0,2)    # [L,N,E] : [F,A,T,D]->[F,T,A,D]->[F*T,A,D]->[A,T*F,D]
            k, v = q, q                                 # [S,N,E] : [A,T*F,D]

            key_padding_mask = padding_mask.permute(1,0).repeat(F,1)    # [N,S] = [T*F,A]
            attn_mask = batch_mask                                      # [L,S] = [A,A]
            # att_output : [L,N,E] : [A,T*F,D]
            att_output, _ = self.layer_att_(q,k,v,key_padding_mask=key_padding_mask,attn_mask=attn_mask)
            # att_output : [F,A,T,D]
            att_output = att_output.reshape((A,F,T,D)).permute(1,0,2,3)

        S_ = att_output + x
        F1_ = self.layer_F1_(S_)
        #F2_ = self.layer_F2_(F1_)
        Z_ = self.layer_Z_(F1_)

        return Z_


class CrossAttLayer_Enc(nn.Module):
    def __init__(self, device, time_steps=91, feature_dim=256, head_num=4, k=4):
        super().__init__()
        self.device = device
        self.time_steps = time_steps
        self.feature_dim = feature_dim
        self.head_num = head_num
        self.k = k

        self.layer_X_ = nn.LayerNorm(feature_dim)
        self.layer_att_ = nn.MultiheadAttention(embed_dim=feature_dim,num_heads=head_num, add_zero_attn=True)
        self.layer_F1_ = nn.Sequential(nn.Linear(feature_dim,feature_dim), nn.ReLU())
        self.layer_F1_.apply(init_xavier_glorot)
        #self.layer_F2_ = nn.Sequential(nn.Linear(k*feature_dim,feature_dim), nn.ReLU())
        self.layer_Z_ = nn.LayerNorm(feature_dim)

    def forward(self, q_, kv, batch_mask, padding_mask=None, hidden_mask=None):
        A,T,D = q_.shape                                     # [L,N,E] : [A,T,D]
        assert (T==self.time_steps and D==self.feature_dim)
        G,T,D = kv.shape                                    # [S,N,E] : [G,T,D]
        assert (T==self.time_steps and D==self.feature_dim)
        A_,G_ = batch_mask.shape                            # [L,S] : [A,G]
        assert (A==A_ and G==G_)
        G__,T__ = padding_mask.shape                        
        padding_mask = padding_mask.permute(1,0)            # [N,S] : T,G
        assert (G==G__ and T==T__)

        k, v = kv, kv
        # att_output : [L,N,E] : [A,T,D]
        att_output,_ = self.layer_att_(q_,k,v,key_padding_mask=padding_mask,attn_mask=batch_mask)
        if torch.isnan(att_output.sum()):
            import ipdb
            ipdb.set_trace()
        S_ = att_output + q_
        F1_ = self.layer_F1_(S_)
        #F2_ = self.layer_F2_(F1_)
        Z_ = self.layer_Z_(F1_)

        return Z_
