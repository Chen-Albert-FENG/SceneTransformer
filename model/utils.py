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

class AttLayer(nn.Module):
    def __init__(self, time_steps, feature_dim, head_num, across_time = True):
        super().__init__()
        self.time_steps = time_steps
        self.feature_dim = feature_dim
        self.head_num = head_num
        assert feature_dim % head_num == 0
        self.head_dim = int(feature_dim/head_num)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        self.across_time = across_time

    def forward(self, Q, K, V, mask = None): # Q, K, V -> [B,A,T,H,d] -> [batch size, agent num, time steps, head num, head dim]
        batch_size = Q.shape[0]
        agent_num = Q.shape[1]
        
        if self.across_time:
            Q, KT, V = Q.permute(0,3,1,2,4), K.permute(0,3,1,4,2), V.permute(0,3,1,2,4) # [B,H,A,T,d] and [B,H,A,d,T] and [B,H,A,T,d]
        else:
            Q, KT, V = Q.permute(0,3,2,1,4), K.permute(0,3,2,4,1), V.permute(0,3,2,1,4) # [B,H,T,A,d] and [B,H,T,d,A] and [B,H,T,A,d]
        
        energy = torch.matmul(Q,KT) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask==0, -1e10) # 0 for ignoring attention

        attention = torch.softmax(energy, dim=-1)       # [B,H,A,T,T] or [B,H,T,A,A]
        output = torch.matmul(attention, V)             # [B,H,A,T,d] or [B,H,T,A,d]

        if self.across_time:
            output = output.permute(0,2,3,1,4).contiguous() # [B,A,T,H,d]
        else:
            output = output.permute(0,3,2,1,4).contiguous() # [B,A,T,H,d]

        output = output.view(batch_size, agent_num, self.time_steps, self.feature_dim)         # [B,A,T,D]

        return output