from myTransformer import *

import torch
import torch.nn as nn

# 1. Vocab, 입력, 세그먼트
vocab = {"[PAD]":0, "[CLS]":1, "[SEP]":2, "I":3, "love":4, "you":5, "How":6, "are":7, "?":8}
inv_vocab = {v:k for k,v in vocab.items()}
vocab_size = len(vocab)

sentence_a = ["I", "love", "you"]
sentence_b = ["How", "are", "you", "?"]
is_next = 1  # 1=연속문장, 0=랜덤

tokens = ["[CLS]"] + sentence_a + ["[SEP]"] + sentence_b + ["[SEP]"]
token_ids = torch.tensor([[vocab[w] for w in tokens]])  # (1, seq_len)
segment_ids = torch.tensor([[0]* (len(sentence_a)+2) + [1]*(len(sentence_b)+1)])  # (1, seq_len)

print("입력:", tokens)
print("segment_ids:", segment_ids)

# 2. 임베딩+세그먼트+포지셔널 인코딩
d_model = 4
num_heads = 2
d_ff = 8
num_layers = 2

embedding = EmbeddingLayer(vocab_size, d_model)
segment_embedding = nn.Embedding(2, d_model)
pos_enc = PositionalEncodingLayer(d_model, max_len=20)
encoder_blocks = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])

token_emb = embedding(token_ids)
seg_emb = segment_embedding(segment_ids)
h = token_emb + seg_emb
h = pos_enc(h)

for block in encoder_blocks:
    h = block(h)

# 3. NSP 헤드: [CLS] 위치 벡터만 뽑기
cls_vec = h[:, 0, :]  # (batch, d_model)
nsp_linear = nn.Linear(d_model, 2)
nsp_logits = nsp_linear(cls_vec)  # (batch, 2)
nsp_probs = nn.Softmax(dim=-1)(nsp_logits)
pred = nsp_probs.argmax(dim=-1).item()

print("NSP 예측 확률:", nsp_probs.detach().numpy())
print("NSP 예측 결과:", "IsNext" if pred==1 else "NotNext", f"(정답: {'IsNext' if is_next==1 else 'NotNext'})")

# 4. NSP Loss 계산 (학습시)
label = torch.tensor([is_next])
loss = nn.CrossEntropyLoss()(nsp_logits, label)
print("NSP Loss:", loss.item())
