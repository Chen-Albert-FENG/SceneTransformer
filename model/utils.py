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

class SelfAttLayer(nn.Module):
    def __init__(self, device, time_steps=91, feature_dim=256, head_num=4, k=4, across_time=True):
        super().__init__()

        self.viewmodule_ = View((-1,time_steps,head_num, int(feature_dim/head_num)))
        self.layer_X_ = nn.LayerNorm(feature_dim)
        self.layer_K_ = nn.Sequential(nn.Linear(feature_dim,feature_dim), nn.ReLU(), self.viewmodule_)
        self.layer_V_ = nn.Sequential(nn.Linear(feature_dim,feature_dim), nn.ReLU(), self.viewmodule_)
        self.layer_Q0_ = nn.Sequential(nn.Linear(feature_dim,feature_dim), nn.ReLU(), self.viewmodule_)
        self.layer_Q_ = ScaleLayer(int(feature_dim/head_num))

        self.scale = torch.sqrt(torch.FloatTensor([head_num])).to(device)

        self.layer_Y2_ = nn.Sequential(View((-1,time_steps,feature_dim)), nn.Linear(feature_dim,feature_dim), nn.ReLU())
        self.layer_F1_ = nn.Sequential(nn.Linear(feature_dim,k*feature_dim), nn.ReLU())
        self.layer_F2_ = nn.Sequential(nn.Linear(k*feature_dim,feature_dim), nn.ReLU())
        self.layer_Z_ = nn.LayerNorm(feature_dim)

        self.across_time = across_time

    def forward(self, x, batch_mask, padding_mask=None, hidden_mask=None):
        x = self.layer_X_(x)
        K = self.layer_K_(x)
        V = self.layer_V_(x)
        Q0 = self.layer_Q0_(x)
        Q = self.layer_Q_(Q0)    # Q,K,V -> [A,T,H,d]
        
        self.scale = self.scale.to(K.device)

        if self.across_time:
            Q, K, V = Q.permute(0,2,1,3), K.permute(0,2,1,3), V.permute(0,2,1,3)    # Q,K,V -> [A,H,T,d]
            energy = torch.matmul(Q,K.permute(0,1,3,2)) / self.scale                               # [A,H,T,T]
            energy.permute(0,3,1,2)[padding_mask==False] = -1e10
            energy.permute(0,2,1,3)[padding_mask==False] = -1e10
            attention = torch.softmax(energy, dim=-1)                               # [A,H,T,T]
            Y1_ = torch.matmul(attention, V)                                        # [A,H,T,d]
            Y1_ = Y1_.permute(0,2,1,3).contiguous()                                 # [A,T,H,d]

        else:
            Q, K, V = Q.permute(1,2,0,3), K.permute(1,2,0,3), V.permute(1,2,0,3)    # Q,K,V -> [T,H,A,d]
            energy = torch.matmul(Q,K.permute(0,1,3,2)) / self.scale                               # [T,H,A,A]

            # if batch_mask is not None:                                              # batch_mask -> [A,A]
            energy = energy.masked_fill(batch_mask==0, -1e10)   # 0 for ignoring attention
            energy.permute(2,0,1,3)[padding_mask==False] = -1e10
            energy.permute(3,0,1,2)[padding_mask==False] = -1e10

            attention = torch.softmax(energy, dim=-1)                               # [T,H,A,A]

            Y1_ = torch.matmul(attention, V)                                        # [T,H,A,d]
            Y1_ = Y1_.permute(2,0,1,3).contiguous()                                 # [A,T,H,d]
        

        Y2_ = self.layer_Y2_(Y1_)
        S_ = Y2_ + x
        F1_ = self.layer_F1_(S_)
        F2_ = self.layer_F2_(F1_)
        Z_ = self.layer_Z_(F2_)

        return Z_, Q, K, V # -> [A,T,D], [A,T,H,d]*3

class CrossAttLayer(nn.Module):
    def __init__(self, device, time_steps=91, feature_dim=256, head_num=4, k=4):
        super().__init__()

        self.viewmodule_ = View((-1,time_steps,head_num, int(feature_dim/head_num)))
        self.layer_X_ = nn.LayerNorm(256)
        
        self.layer_K_ = nn.Sequential(nn.Linear(feature_dim,feature_dim), nn.ReLU(), self.viewmodule_)
        self.layer_V_ = nn.Sequential(nn.Linear(feature_dim,feature_dim), nn.ReLU(), self.viewmodule_)
        self.layer_Q0_ = nn.Sequential(nn.Linear(feature_dim,feature_dim), nn.ReLU(), self.viewmodule_)
        self.layer_Q_ = ScaleLayer(int(feature_dim/head_num))

        self.scale = torch.sqrt(torch.FloatTensor([head_num])).to(device)

        self.layer_Y2_ = nn.Sequential(View((-1,time_steps,feature_dim)), nn.Linear(feature_dim,feature_dim), nn.ReLU())
        self.layer_F1_ = nn.Sequential(nn.Linear(feature_dim,k*feature_dim), nn.ReLU())
        self.layer_F2_ = nn.Sequential(nn.Linear(k*feature_dim,feature_dim), nn.ReLU())
        self.layer_Z_ = nn.LayerNorm(feature_dim)

    def forward(self, agent, rg, agent_rg_mask, padding_mask, rg_valid_mask): # agent -> [A,T,D] / rg -> [G,T,D]
        agent = self.layer_X_(agent)
        rg = self.layer_X_(rg)
        
        K = self.layer_K_(rg)                                                       # [G,T,H,d]
        V = self.layer_V_(rg)                                                       # [G,T,H,d]
        Q0 = self.layer_Q0_(agent)
        Q = self.layer_Q_(Q0)                                                       # [A,T,H,d]

        Q, K, V = Q.permute(1,2,0,3), K.permute(1,2,0,3), V.permute(1,2,0,3)    # Q -> [T,H,A,d] / K,V -> [T,H,G,d]
        energy = torch.matmul(Q,K.permute(0,1,3,2)) / Q.shape[1]               # [T,H,A,G]

        energy.permute(2,3,0,1)[agent_rg_mask==False] = -1e10
        energy.permute(2,0,1,3)[padding_mask==False] = -1e10
        energy.permute(3,0,1,2)[rg_valid_mask==False] = -1e10

        attention = torch.softmax(energy, dim=-1)                               # [T,H,A,G]

        Y1_ = torch.matmul(attention, V)                                        # [T,H,A,d]
        Y1_ = Y1_.permute(2,0,1,3).contiguous()                                 # [A,T,H,d]

        Y2_ = self.layer_Y2_(Y1_)
        S_ = Y2_ + agent
        F1_ = self.layer_F1_(S_)
        F2_ = self.layer_F2_(F1_)
        Z_ = self.layer_Z_(F2_)

        return Z_, Q, K, V # -> [A,T,D], [G,T,H,d], [A,T,H,d]*2

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
        self.layer_att_ = nn.MultiheadAttention(embed_dim=feature_dim,num_heads=head_num)
        self.layer_F1_ = nn.Sequential(nn.Linear(feature_dim,k*feature_dim), nn.ReLU())
        self.layer_F2_ = nn.Sequential(nn.Linear(k*feature_dim,feature_dim), nn.ReLU())
        self.layer_Z_ = nn.LayerNorm(feature_dim)

    def forward(self, x, batch_mask, padding_mask=None, hidden_mask=None):
        A,T,D = x.shape
        assert (T==self.time_steps and D==self.feature_dim)
        A,A = batch_mask.shape
        A,T = padding_mask.shape
        assert T==self.time_steps

        x_ = self.layer_X_(x) # [A,T,D]

        if self.across_time:
            q = x_.permute(1,0,2)                       # [L,N,E] : [A,T,D]->[T,A,D]
            k,v = q.clone(),q.clone()                   # [S,N,E] : [T,A,D]

            key_padding_mask = padding_mask             # [N,S] : [A,T]
            attn_mask = None  
            # att_output : [L,N,E] : [T,A,D]
            att_output, _ = self.layer_att_(q,k,v,key_padding_mask=key_padding_mask,attn_mask=attn_mask)
            # att_output : [A,T,D]
            att_output = att_output.permute(1,0,2)
        else:
            q = x_                                      # [L,N,E] = [A,T,D]
            k, v = q.clone(), q.clone()                 # [S,N,E] = [A,T,D]

            key_padding_mask = padding_mask.permute(1,0)# [N,S] = [T,A]
            attn_mask = batch_mask                      # [L,S] = [A,A]
            # att_output : [L,N,E] : [A,T,D]
            att_output, _ = self.layer_att_(q,k,v,key_padding_mask=key_padding_mask,attn_mask=attn_mask)

        S_ = att_output + x
        F1_ = self.layer_F1_(S_)
        F2_ = self.layer_F2_(F1_)
        Z_ = self.layer_Z_(F2_)

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
        self.layer_att_ = nn.MultiheadAttention(embed_dim=feature_dim,num_heads=head_num)
        self.layer_F1_ = nn.Sequential(nn.Linear(feature_dim,k*feature_dim), nn.ReLU())
        self.layer_F2_ = nn.Sequential(nn.Linear(k*feature_dim,feature_dim), nn.ReLU())
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
            k,v = q.clone(),q.clone()                                   # [S,N,E] : [T,F*A,D]

            key_padding_mask = padding_mask.repeat(F,1)                 # [N,S] : [A*F,T]
            attn_mask = None  
            # att_output : [L,N,E] : [T,F*A,D]
            att_output, _ = self.layer_att_(q,k,v,key_padding_mask=key_padding_mask,attn_mask=attn_mask)
            # att_output : [F,A,T,D]
            att_output = att_output.reshape((T,F,A,D)).permute(1,2,0,3)
        else:
            q = x_.permute(0,2,1,3).reshape((-1,A,D)).permute(1,0,2)    # [L,N,E] : [F,A,T,D]->[F,T,A,D]->[F*T,A,D]->[A,T*F,D]
            k, v = q.clone(), q.clone()                                 # [S,N,E] : [A,T*F,D]

            key_padding_mask = padding_mask.permute(1,0).repeat(F,1)    # [N,S] = [T*F,A]
            attn_mask = batch_mask                                      # [L,S] = [A,A]
            # att_output : [L,N,E] : [A,T*F,D]
            att_output, _ = self.layer_att_(q,k,v,key_padding_mask=key_padding_mask,attn_mask=attn_mask)
            # att_output : [F,A,T,D]
            att_output = att_output.reshape((A,F,T,D)).permute(1,0,2,3)

        S_ = att_output + x
        F1_ = self.layer_F1_(S_)
        F2_ = self.layer_F2_(F1_)
        Z_ = self.layer_Z_(F2_)

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
        self.layer_att_ = nn.MultiheadAttention(embed_dim=feature_dim,num_heads=head_num)
        self.layer_F1_ = nn.Sequential(nn.Linear(feature_dim,k*feature_dim), nn.ReLU())
        self.layer_F2_ = nn.Sequential(nn.Linear(k*feature_dim,feature_dim), nn.ReLU())
        self.layer_Z_ = nn.LayerNorm(feature_dim)

    def forward(self, q, kv, batch_mask, padding_mask=None, hidden_mask=None):
        A,T,D = q.shape                                     # [L,N,E] : [A,T,D]
        assert (T==self.time_steps and D==self.feature_dim)
        G,T,D = kv.shape                                    # [S,N,E] : [G,T,D]
        assert (T==self.time_steps and D==self.feature_dim)
        A_,G_ = batch_mask.shape                            # [L,S] : [A,G]
        assert (A==A_ and G==G_)
        G__,T__ = padding_mask.shape                        
        padding_mask = padding_mask.permute(1,0)            # [N,S] : T,G
        assert (G==G__ and T==T__)

        k, v = kv.clone(), kv.clone()
        # att_output : [L,N,E] : [A,T,D]
        att_output,_ = self.layer_att_(q,k,v,key_padding_mask=padding_mask,attn_mask=batch_mask)

        S_ = att_output + q
        F1_ = self.layer_F1_(S_)
        F2_ = self.layer_F2_(F1_)
        Z_ = self.layer_Z_(F2_)

        return Z_