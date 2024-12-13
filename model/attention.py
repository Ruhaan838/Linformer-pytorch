import torch
from torch import nn
from torch import functional as F
import math

class LinearSelfAttention(nn.Module):
    def __init__(self, d_model:int, seq_len:int, k:int, d_k:int, headwise_sharing:bool=True, key_val_sharing:bool=False, dropout:float=0.2):
        """LinearSelfAttention args

        Args:
            d_model (int): eg. 512, 1024 ...
            seq_len (int): eg. 6, 1000, 20000 ...
            k (int): Reduction Term as per the paper
            d_k (int): d_model // head
            dropout (float, optional): Dropout Rate. Defaults to 0.2.
        """
        super().__init__()
        
        self.d_model = d_model
        self.seq_len = seq_len
        self.k = k
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
        self.headwise_sharing = headwise_sharing
        self.key_val_sharing = key_val_sharing

        self.E = nn.Linear(seq_len, k)
        self.F = nn.Linear(seq_len, k) if headwise_sharing else self.E if key_val_sharing else None
        
        assert(self.F is not None), "At least one sharing is require to perform the Linear Attention."
    
    def forward(self, q, k, v, mask=None):
        
        K = k.transpose(-1, -2)
        K_ = self.E(K)
        
        attention = (q @ K_) / math.sqrt(self.d_k)

        if mask:
            mask = torch.triu(torch.ones((self.seq_len, self.k))).bool().to(q.device)
            attention.masked_fill_(mask == 0, -1e9)
        
        attention = attention.softmax(dim=-1)
        
        attention = self.dropout(attention)
        
        V = v.transpose(-1, -2)
        V_ = self.F(V)
        V_ = V_.transpose(-1, -2)
        return attention @ V_  

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, seq_len:int, k:int, head:int, 
                 headwise_sharing:bool=True, key_val_sharing:bool=False ,dropout:float=0.2):
        """MultiHeadAttention

        Args:
            d_model (int)
            seq_len (int)
            k (int) 
            head (int) 
            dropout (float, optional). Defaults to 0.2.
        """
        super().__init__()
        
        assert d_model % head == 0, "D_model is must divided by the head"
        
        self.d_model = d_model
        self.seq_len = seq_len
        self.k = k
        self.head = head
        self.d_k = d_model // head
        
        self.Wk = nn.Linear(d_model, d_model)
        self.Wq = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        
        self.Wo = nn.Linear(d_model, d_model)
        
        self.self_attention = LinearSelfAttention(self.d_k, seq_len, k, self.d_k, headwise_sharing, key_val_sharing, dropout)
        
    def forward(self, q, k, v, mask=None):

        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)
        
        query = q.view(q.shape[0], q.shape[1], self.head, self.d_k).transpose(1, 2) 
        key = k.view(k.shape[0], v.shape[1], self.head, self.d_k).transpose(1, 2)
        value = v.view(v.shape[0], v.shape[1], self.head, self.d_k).transpose(1, 2)

        attention_score = self.self_attention(query, key, value, mask)
        
        x = attention_score.transpose(1, 2).contiguous().view(attention_score.shape[0], -1, self.head*self.d_k)
        
        return self.Wo(x)

if __name__ == "__main__":   
    m = MultiHeadAttention(512, 6, 256, 8, True, False)
    mask = torch.triu(torch.ones((1, 1, 256)), diagonal=1)
    mask = mask == 0
    a = torch.randn(1, 6, 512)
    print(m(a, a, a, mask).size())
