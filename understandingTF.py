from myTransformer import *

import torch
import torch.nn as nn
import math

# 하이퍼파라미터 예시
vocab = {"I":0, "love":1, "you":2}
sentence = ["I", "love", "you"]
x = torch.tensor([[vocab[w] for w in sentence]])  # (batch=1, seq_len=3)
vocab_size = 3
d_model = 4
num_heads = 2
d_ff = 8

embedding = EmbeddingLayer(vocab_size, d_model)
pos_enc = PositionalEncodingLayer(d_model, max_len=10)
encoder_block = EncoderBlock(d_model, num_heads, d_ff)

# 실제 플로우
print(f'''입력: {x}''')
h = embedding(x)
h = pos_enc(h)
h = encoder_block(h)
