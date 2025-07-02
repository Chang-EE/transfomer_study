from myTransformer import *
import torch
import torch.nn as nn
import math


# ----------------------- 1. Vocab/입출력 데이터 -----------------------
vocab = {
    "[PAD]":0, "[BOS]":1, "[EOS]":2, "translate":3, "I":4, "love":5, "you":6,
    "J":7, "'":8, "aime":9, "tu":10,
    "나":11, "는":12, "너":13, "를":14, "사랑":15, "해":16
}
inv_vocab = {v:k for k,v in vocab.items()}
vocab_size = len(vocab)

src_sentence = ["translate", "I", "love", "you"]
tgt_sentence = ["[BOS]", "나", "는", "너", "를", "사랑", "해"]
target = ["나", "는", "너", "를", "사랑", "해", "[EOS]"]

src = torch.tensor([[vocab[w] for w in src_sentence]])     # (1, 4)
tgt_in = torch.tensor([[vocab[w] for w in tgt_sentence]])  # (1, 5)
tgt_out = torch.tensor([[vocab[w] for w in target]])       # (1, 5)

# ----------------------- 2. 하이퍼파라미터 및 모델 -----------------------
d_model = 8
num_heads = 2
d_ff = 16
num_layers = 2

embedding = EmbeddingLayer(vocab_size, d_model)
pos_enc = PositionalEncodingLayer(d_model, max_len=20)

encoder_blocks = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
decoder_blocks = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])

output_linear = nn.Linear(d_model, vocab_size)
softmax = nn.Softmax(dim=-1)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    list(embedding.parameters()) +
    list(pos_enc.parameters()) +
    list(output_linear.parameters()) +
    [p for block in encoder_blocks for p in block.parameters()] +
    [p for block in decoder_blocks for p in block.parameters()]
)

# ----------------------- 3. 마스킹 함수 (디코더용) -----------------------
def generate_future_mask(sz):
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

# ----------------------- 4. 학습 루프 -----------------------
for epoch in range(100):
    optimizer.zero_grad()
    # --- 인코더 ---
    src_emb = embedding(src)           # (1, 4, d_model)
    src_emb = pos_enc(src_emb)         # (1, 4, d_model)
    h_enc = src_emb
    for block in encoder_blocks:
        h_enc = block(h_enc)
    # --- 디코더 ---
    tgt_emb = embedding(tgt_in)        # (1, 5, d_model)
    tgt_emb = pos_enc(tgt_emb)         # (1, 5, d_model)
    mask = generate_future_mask(tgt_in.size(1))   # (5, 5)
    h_dec = tgt_emb
    for block in decoder_blocks:
        h_dec = block(h_dec, enc_out=h_enc, self_attn_mask=mask)
    logits = output_linear(h_dec)                           # (1, 5, vocab_size)
    loss = loss_fn(logits.view(-1, vocab_size), tgt_out.view(-1))
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ----------------------- 5. 예측 결과 (Teacher Forcing) -----------------------
with torch.no_grad():
    src_emb = embedding(src)
    src_emb = pos_enc(src_emb)
    h_enc = src_emb
    for block in encoder_blocks:
        h_enc = block(h_enc)
    tgt_emb = embedding(tgt_in)
    tgt_emb = pos_enc(tgt_emb)
    mask = generate_future_mask(tgt_in.size(1))
    h_dec = tgt_emb
    for block in decoder_blocks:
        h_dec = block(h_dec, enc_out=h_enc, self_attn_mask=mask)
    logits = output_linear(h_dec)
    probs = softmax(logits)
    pred_token_ids = probs.argmax(dim=-1)         # (1, 5)
    pred_tokens = [inv_vocab[idx.item()] for idx in pred_token_ids[0]]
    print(f"입력: {src_sentence}")
    print(f"정답: {target}")
    print(f"모델 예측: {pred_tokens}")
