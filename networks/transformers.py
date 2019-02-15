import torch
import torch.nn.functional as F
from torch import nn

import math,copy,pdb

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                # pdb.set_trace()
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2. / n))
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                n = module.weight.size(1)
                module.weight.data.normal_(0, math.sqrt(2. / n))
                module.bias.data.zero_()

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
    / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Sequential(nn.Linear(d_model, d_model),nn.ReLU()), 4)
        #self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
        [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
        dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
        .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class MultiHeadedAttention2(nn.Module):
    def __init__(self, h, d_model_in, d_model_out, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention2, self).__init__()
        assert d_model_out % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model_out // h
        self.h = h
        self.linears = clones(nn.Sequential(nn.Linear(d_model_in, d_model_out),nn.ReLU()), 3)
        self.last_linear = nn.Sequential(nn.Linear(d_model_out, d_model_in),nn.ReLU())
        #self.linears = clones(nn.Linear(d_model_in, d_model_out), 3)
        #self.last_linear = nn.Linear(d_model_out, d_model_in)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model_out => h x d_k
        query, key, value = \
        [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
        dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
        .view(nbatches, -1, self.h * self.d_k)
        return self.last_linear(x)
        
class transformer(nn.Module):
    def __init__(self,h,d_model,type='channel'):
        super(transformer,self).__init__()
        self.type = type
        self.mha = MultiHeadedAttention(h,d_model)
        
    def forward(self, x, y):
        # 2,1280,30,40
        n,c,h,w = x.size()
        c2 = y.size(1)
        #pdb.set_trace()
        if self.type == 'channel':
            x = x.view(n,c,h*w)
            y = y.view(n,c2,h*w)
            x = self.mha(x,y,y)
            x = x.view(n,c,h,w)
        elif self.type == 'spatial':
            x = x.view(n,c,h*w).transpose(-2,-1)
            x = self.mha(x,x,x)
            x = x.transpose(-2,-1).view(n,c,h,w)
        return x

class self_transformer(nn.Module):
    def __init__(self,h,d_model):
        super(self_transformer,self).__init__()
        self.mha = MultiHeadedAttention(h,d_model)
        #self.mha2 = MultiHeadedAttention(h,d_model)
        
    def forward(self, x):
        # 2,256,30,40
        n,c,h,w = x.size()
        #pdb.set_trace()
        #h_dim:n,c,h,1
        x1 = x.mean(3).squeeze().transpose(1,2)
        x1 = self.mha(x1,x1,x1)
        x1 = x1.transpose(1,2).view(n,c,h,1)
        #pdb.set_trace()
        x = (x + x1)/2
        #w_dim:n,c,1,w
        x2 = x.mean(2).squeeze().transpose(1,2)
        x2 = self.mha(x2,x2,x2)
        x2 = x2.transpose(1,2).view(n,c,1,w)
        x = (x + x2)/2
        return x

class self_transformer2(nn.Module):
    def __init__(self):
        super(self_transformer2,self).__init__()
        self.attn1=None
        self.attn2=None
        
    def forward(self, x):
        # 2,256,30,40
        n,c,h,w = x.size()
        #pdb.set_trace()
        #h_dim:n,1,h,w
        x1 = x.mean(1).squeeze()
        x1, self.attn1 = attention(x1,x1,x1)
        x1 = x1.view(n,1,h,w)
        #pdb.set_trace()
        x = (x + x1)/2
        #w_dim:n,1,h,w
        x2 = x.mean(1).squeeze().transpose(1,2)
        x2, self.attn2 = attention(x2,x2,x2)
        x2 = x2.transpose(1,2).view(n,1,h,w)
        x = (x + x2)/2
        return x
        
class self_transformer3(nn.Module):
    def __init__(self):
        super(self_transformer3,self).__init__()
        self.attn1=None
        self.attn2=None
        
    def forward(self, x):
        # 2,256,30,40
        #h_dim:n,c,h,w
        x, self.attn1 = attention(x,x,x)
        #pdb.set_trace()
        #w_dim:n,c,w,h
        x = x.transpose(-2,-1)
        x, self.attn2 = attention(x,x,x)
        x = x.transpose(-2,-1)
        return x
        
class self_transformer4(nn.Module):
    def __init__(self):
        super(self_transformer4,self).__init__()
        self.attn1=None
        self.attn2=None
        
    def forward(self, x):
        # 2,256,30,40
        n,c,h,w = x.size()
        #pdb.set_trace()
        #h_dim:n,c,h,1
        x1 = x.mean(3).squeeze().transpose(1,2)
        x1,self.attn1 = attention(x1,x1,x1)
        x1 = x1.transpose(1,2).view(n,c,h,1)
        #pdb.set_trace()
        x = (x + x1)/2
        #w_dim:n,c,1,w
        x2 = x.mean(2).squeeze().transpose(1,2)
        x2,self.attn2 = attention(x2,x2,x2)
        x2 = x2.transpose(1,2).view(n,c,1,w)
        x = (x + x2)/2
        return x        

class self_transformer5(nn.Module):
    def __init__(self):
        super(self_transformer5,self).__init__()
        self.mha1 = MultiHeadedAttention2(1,40,256)
        self.mha2 = MultiHeadedAttention2(1,30,256)
        
    def forward(self, x):
        # 2,256,30,40
        n,c,h,w = x.size()
        #pdb.set_trace()
        #h_dim:n,1,h,w
        x1 = x.mean(1).squeeze()
        x1 = self.mha1(x1,x1,x1)
        x1 = x1.view(n,1,h,w)
        #pdb.set_trace()
        x = (x + x1)/2
        #w_dim:n,1,h,w
        x2 = x.mean(1).squeeze().transpose(1,2)
        x2 = self.mha2(x2,x2,x2)
        x2 = x2.transpose(1,2).view(n,1,h,w)
        x = (x + x2)/2
        return x