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


# === 인코더 파트는 이미 완성 ===
print(f"인코더 입력: {x}")
h = embedding(x)                # (batch, src_seq_len, d_model)
h = pos_enc(h)                  # (batch, src_seq_len, d_model)
enc_out = encoder_block(h)      # (batch, src_seq_len, d_model)


# === 디코더 입력(예: 'I', 'love') ===
# 일반적으로 디코더 입력은 [BOS, I, love]처럼, 오른쪽으로 한 칸 shift된 시퀀스를 사용
# 예시: "I love" (batch=1, tgt_seq_len=2)
tgt_sentence = ["I", "love"]
tgt_x = torch.tensor([[vocab[w] for w in tgt_sentence]])  # (batch=1, tgt_seq_len=2)
print(f"디코더 입력: {tgt_x}")

# === 임베딩, 포지셔널 인코딩 ===
tgt_embedding = EmbeddingLayer(vocab_size, d_model)
tgt_pos_enc = PositionalEncodingLayer(d_model, max_len=10)
decoder_block = DecoderBlock(d_model, num_heads, d_ff)

tgt_h = tgt_embedding(tgt_x)          # (batch, tgt_seq_len, d_model)
tgt_h = tgt_pos_enc(tgt_h)            # (batch, tgt_seq_len, d_model)

# === 마스킹(디코더용, 미래정보 차단) ===
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    return mask

tgt_seq_len = tgt_x.size(1)
self_attn_mask = generate_square_subsequent_mask(tgt_seq_len)
dec_out = decoder_block(tgt_h, enc_out, self_attn_mask=self_attn_mask)

# === 최종 Linear 레이어 및 Softmax ===
output_linear = nn.Linear(d_model, vocab_size)   # (d_model → vocab_size)
softmax = nn.Softmax(dim=-1)

# 디코더 출력: (batch, tgt_seq_len, d_model)
logits = output_linear(dec_out)                  # (batch, tgt_seq_len, vocab_size)
probs = softmax(logits)                          # (batch, tgt_seq_len, vocab_size)

print(f"== Linear 변환 결과 (logits) ==\n{logits}\n")
print(f"== Softmax 확률 분포 ==\n{probs}\n")

# === 예측 토큰(가장 확률 높은 것 선택) ===
# 각 위치별로 argmax
pred_token_ids = probs.argmax(dim=-1)   # (batch, tgt_seq_len)
print(f"== 예측 토큰 인덱스 ==\n{pred_token_ids}\n")

# 예측 결과 해석(문자열로 변환)
inv_vocab = {v:k for k,v in vocab.items()}
pred_tokens = [[inv_vocab[idx.item()] for idx in seq] for seq in pred_token_ids]
print(f"== 예측 결과(단어) ==\n{pred_tokens}\n")

