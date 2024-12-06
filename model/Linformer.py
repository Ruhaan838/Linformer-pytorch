import torch
from torch import nn
from torch.nn import functional as F

from model.attention import MultiHeadAttention
from model.embeddings import InputEmbedding, PositionalEmbedding

class FeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float=0.2):
        """FeedForward Network convert from d_model-> d_ff and again d_ff -> d_model

        Args:
            d_model (int): 
            d_ff (int): 
            dropout (float, optional): Defaults to 0.2.
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.layer1 = nn.Linear(d_model, d_ff)
        self.layer2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return self.dropout(x)
        
class EncoderBlock(nn.Module):
    def __init__(self, d_model:int, seq_len:int, k:int, 
                 d_ff:int, head:int, headwise_sharing:bool=True, 
                 key_val_sharing:bool=False, dropout:float=0.2):
        """One Single Encdoer Block

        Args:
            d_model (int): 
            seq_len (int): 
            k (int): 
            d_ff (int): 
            head (int): 
            dropout (float, optional): Defaults to 0.2.
        """
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.multihead = MultiHeadAttention(d_model, seq_len, k, head, headwise_sharing, key_val_sharing, dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
    
    def forward(self, x):
        res = x
        x = self.norm1(x)
        x = self.multihead(x, x, x)
        x += res
        
        res = x
        x = self.norm2(x)
        x = self.ff(x)
        x += res
        
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model:int, seq_len:int, 
                 k:int, d_ff:int, head:int, headwise_sharing:bool=True,
                 key_val_sharing:bool=False, dropout:float=0.2):
        """One Single Decoder Block

        Args:
            d_model (int): 
            seq_len (int): 
            k (int): 
            d_ff (int): 
            head (int): 
            dropout (float, optional): . Defaults to 0.2.
        """
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.maskmultihead = MultiHeadAttention(d_model, seq_len, k, head, headwise_sharing, key_val_sharing, dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.crossmultihead = MultiHeadAttention(d_model, seq_len, k, head, headwise_sharing, key_val_sharing, dropout)
        
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        
    def forward(self, x, encoder_output, dec_mask):
        
        res = x
        x = self.norm1(x)
        x = self.maskmultihead(x, x, x, dec_mask)
        x += res
        
        res = x
        x = self.norm2(x)
        x = self.crossmultihead(x, encoder_output, encoder_output)
        x += res
        
        x = self.norm3(x)
        x = self.ff(x)
        x += res
        
        return x        

class Projection(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        
        self.layer = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        return self.layer(x)

class LinFormer(nn.Module):
    
    def __init__(self, d_model:int, 
                       enc_seq_len:int, 
                       dec_seq_len:int,
                       enc_vocab_size:int,
                       dec_vocab_size:int, 
                       d_ff:int, 
                       k:int, 
                       head:int, 
                       N:int,
                       headwise_sharing:bool=True, 
                       key_val_sharing:bool=False,
                       dropout:float=0.2):
        
        super().__init__()
        
        self.enc_embd = InputEmbedding(d_model, enc_vocab_size)
        self.enc_pos = PositionalEmbedding(d_model, dec_seq_len)
        self.dec_embd = InputEmbedding(d_model, enc_vocab_size)
        self.dec_pos = PositionalEmbedding(d_model, dec_seq_len)
        
        self.encoders = nn.ModuleList([EncoderBlock(d_model, enc_seq_len, k, 
                                                    d_ff, head, headwise_sharing, 
                                                    key_val_sharing, dropout) for _ in range(N)])
        self.decoders = nn.ModuleList([DecoderBlock(d_model, dec_seq_len, k, 
                                                    d_ff, head, headwise_sharing,
                                                    key_val_sharing,
                                                    dropout) for _ in range(N)])
        
        self.f_layer = Projection(d_model, dec_vocab_size)
        
    def encode(self, x):
        x = self.enc_embd(x)
        x = self.enc_pos(x)
        
        for layer in self.encoders:
            x = layer(x)
        return x
    
    def decode(self, x, encoder_output, tgt_mask):
        x = self.dec_embd(x) # 1,1 -> vs, 512
        x = self.dec_pos(x) # b, seq_len, d_model
        
        for layer in self.decoders:
            x = layer(x, encoder_output, tgt_mask)
        return x
    
    def projection(self, x):
        return self.f_layer(x)

def get_init_model(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model, d_ff, N, head, K, headwise_sharing=True, key_val_sharing=False, dropout=0.2):
    
    model = LinFormer(d_model=d_model,
                      enc_seq_len=src_seq_len,
                      dec_seq_len=tgt_seq_len,
                      enc_vocab_size=src_vocab_size,
                      dec_vocab_size=tgt_vocab_size,
                      d_ff=d_ff,
                      k=K,
                      head=head,
                      N=N,
                      headwise_sharing=headwise_sharing,
                      key_val_sharing=key_val_sharing,
                      dropout=dropout)    
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return model

#test
if __name__ == "__main__":
    model = get_init_model(3, 3, 100, 100, 512, 2048, 1, 8, 256)

    en = torch.randint(0, 3, (1, 100))  
    de = torch.randint(0, 3, (1, 100))

    encode = model.encode(en)
    print("Encode output size:", encode.size())

    decode = model.decode(de, encode, True)
    print("Decode output size:", decode.size())

    proj = model.projection(decode)
    print("Final output:", proj.size())
