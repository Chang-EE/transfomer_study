from myTransformer import *

import torch
import torch.nn as nn
import math

# --- 2. Vocab/데이터/하이퍼파라미터 ---
vocab = {"[PAD]":0, "[BOS]":1, "[EOS]":2, "I":3, "love":4, "you":5}
inv_vocab = {v:k for k,v in vocab.items()}
vocab_size = len(vocab)

sentence = ["[BOS]", "I", "love"]    # 입력(4개 토큰)
target = ["I", "love", "you"]      # 오른쪽 한 칸 shift(학습용 정답)

x = torch.tensor([[vocab[w] for w in sentence]])   # (1, 4)
tgt = torch.tensor([[vocab[w] for w in target]])   # (1, 4)

d_model = 8
num_heads = 2
d_ff = 16
num_layers = 2

# --- 3. 모델 구성 ---
embedding = EmbeddingLayer(vocab_size, d_model)
pos_enc = PositionalEncodingLayer(d_model, max_len=20)
decoder_blocks = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])

# 미래 마스킹 함수 (트라이앵글)
def generate_future_mask(sz):
    # PyTorch 2.0 이상 버전
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

# --- 4. Forward + 학습 ---
output_linear = nn.Linear(d_model, vocab_size)
softmax = nn.Softmax(dim=-1)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    list(embedding.parameters()) +
    list(pos_enc.parameters()) +
    list(output_linear.parameters()) +
    [p for block in decoder_blocks for p in block.parameters()]
)

for epoch in range(100):
    optimizer.zero_grad()
    h = embedding(x)          # (1, 4, d_model)
    h = pos_enc(h)            # (1, 4, d_model)
    mask = generate_future_mask(x.size(1))  # (4, 4)

    for block in decoder_blocks:
        # GPT 구조에서는 cross-attn 미사용, enc_out=None
        h = block(h, enc_out=None, self_attn_mask=mask, enc_key_padding_mask=None)

    logits = output_linear(h)                # (1, 4, vocab_size)
    loss = loss_fn(logits.view(-1, vocab_size), tgt.view(-1))
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# --- 5. 예측 결과 확인 ---
with torch.no_grad():
    h = embedding(x)
    h = pos_enc(h)
    mask = generate_future_mask(x.size(1))
    for block in decoder_blocks:
        h = block(h, enc_out=None, self_attn_mask=mask, enc_key_padding_mask=None)
    logits = output_linear(h)
    probs = softmax(logits)
    pred_token_ids = probs.argmax(dim=-1)    # (1, 4)
    pred_tokens = [inv_vocab[idx.item()] for idx in pred_token_ids[0]]
    print(f"입력: {sentence}")
    print(f"정답: {target}")
    print(f"모델 예측: {pred_tokens}")
