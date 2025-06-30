import torch
import torch.nn as nn
import math


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        print(f'''임베딩: {self.embedding(x)}''')
        return self.embedding(x)
    


class PositionalEncodingLayer(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        print(f'''위치벡터: {self.pe[:, :x.size(1)]}''')
        print(f'''포지셔널 인코딩 결과: {x}''')
        return x
    


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
    def forward(self, x):
        # key, value, query 모두 동일 입력, mask 없음(인코더 기준)
        attn_out, _ = self.mha(x, x, x)
        print(f'''어텐션 레이어: {attn_out}''')
        return attn_out


class ResidualLayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x, sublayer_out):
        print(f'''Rescon + Norm: {self.norm(x + sublayer_out)}''')
        return self.norm(x + sublayer_out)


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),                 # 논문은 ReLU, 실전은 GELU도 가능
            nn.Linear(d_ff, d_model)
        )
    def forward(self, x):
        print(f'''FFN: {x}''')
        return self.ffn(x)

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = SelfAttentionLayer(d_model, num_heads)
        self.resnorm1 = ResidualLayerNorm(d_model)
        self.ffn = FeedForwardLayer(d_model, d_ff)
        self.resnorm2 = ResidualLayerNorm(d_model)

    def forward(self, x):
        # 1. Multi-Head Self-Attention + Residual + Norm
        attn_out = self.self_attn(x)
        out1 = self.resnorm1(x, attn_out)
        # 2. FFN + Residual + Norm
        ffn_out = self.ffn(out1)
        out2 = self.resnorm2(out1, ffn_out)
        print(f'''최종 결과: {out2}''')
        return out2
