from myTransformer import *

import torch
import torch.nn as nn
import torch.optim as optim

# ============ 1. Vocab 및 입력 데이터 ============
vocab = {
    "[PAD]":0, "[CLS]":1, "[SEP]":2, "I":3, "love":4, "you":5, 
    "How":6, "are":7, "?":8, "[MASK]":9
}
inv_vocab = {v:k for k,v in vocab.items()}
vocab_size = len(vocab)

# 입력 시퀀스 예시: [CLS] I [MASK] you [SEP] How are you ? [SEP] (길이 10)
tokens = ["[CLS]", "I", "[MASK]", "you", "[SEP]", "How", "are", "you", "?", "[SEP]"]
token_ids = torch.tensor([[vocab[w] for w in tokens]])       # (1, 10)
segment_ids = torch.tensor([[0]*5 + [1]*5])                  # (1, 10) - 문장A:0, 문장B:1

# 정답 시퀀스 (MLM): [CLS] love you you [SEP] How are you ? [SEP]
mlm_labels = torch.tensor([
    [vocab["[CLS]"], vocab["love"], vocab["you"], vocab["you"], vocab["[SEP]"], 
     vocab["How"], vocab["are"], vocab["you"], vocab["?"], vocab["[SEP]"]]
]) # (1, 10)

# 마스크 ([MASK] 위치만 1, 나머지는 0)
mlm_mask = torch.tensor([
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
]) # (1, 10) - 두 번째 위치([MASK])만 1

is_next = 1  # NSP 정답: 실제 다음문장(1) 또는 무작위(0)

# ============ 2. 모델 정의 ============
d_model = 8
num_heads = 2
d_ff = 16
num_layers = 2

# 임베딩/포지션/세그먼트 임베딩
embedding = nn.Embedding(vocab_size, d_model)
segment_embedding = nn.Embedding(2, d_model)
pos_enc = nn.Embedding(20, d_model)   # 포지션 임베딩 (간단 구현)
encoder_blocks = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])

mlm_linear = nn.Linear(d_model, vocab_size)
nsp_linear = nn.Linear(d_model, 2)

# ============ 3. Optimizer/Loss ============
params = (
    list(embedding.parameters()) +
    list(segment_embedding.parameters()) +
    list(pos_enc.parameters()) +
    list(mlm_linear.parameters()) +
    list(nsp_linear.parameters()) +
    [p for block in encoder_blocks for p in block.parameters()]
)
optimizer = optim.Adam(params)
mlm_loss_fn = nn.CrossEntropyLoss()
nsp_loss_fn = nn.CrossEntropyLoss()

# ============ 4. 학습 루프 ============
for epoch in range(100):
    optimizer.zero_grad()

    # 임베딩+세그먼트+포지션
    tok_emb = embedding(token_ids)                      # (1, 10, d_model)
    seg_emb = segment_embedding(segment_ids)            # (1, 10, d_model)
    pos_ids = torch.arange(0, token_ids.size(1)).unsqueeze(0)
    pos_emb = pos_enc(pos_ids)                          # (1, 10, d_model)
    h = tok_emb + seg_emb + pos_emb                     # (1, 10, d_model)
    for block in encoder_blocks:
        h = block(h)

    # --- MLM Loss ---
    mlm_logits = mlm_linear(h)                          # (1, 10, vocab_size)
    # Flatten for mask-indexing
    mlm_logits_flat = mlm_logits.view(-1, vocab_size)   # (10, vocab_size)
    mlm_labels_flat = mlm_labels.view(-1)               # (10,)
    mlm_mask_flat = mlm_mask.view(-1).bool()            # (10,)

    mlm_logits_masked = mlm_logits_flat[mlm_mask_flat]  # (n_masked, vocab_size)
    mlm_labels_masked = mlm_labels_flat[mlm_mask_flat]  # (n_masked,)

    if mlm_logits_masked.shape[0] > 0:
        mlm_loss = mlm_loss_fn(mlm_logits_masked, mlm_labels_masked)
    else:
        mlm_loss = torch.tensor(0.0)

    # --- NSP Loss ---
    cls_vec = h[:, 0, :]                             # (1, d_model)
    nsp_logits = nsp_linear(cls_vec)                 # (1, 2)
    nsp_label = torch.tensor([is_next])
    nsp_loss = nsp_loss_fn(nsp_logits, nsp_label)

    total_loss = mlm_loss + nsp_loss
    total_loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | MLM loss: {mlm_loss.item():.4f}, NSP loss: {nsp_loss.item():.4f}")

# ============ 5. 학습 후 예측 ============
with torch.no_grad():
    tok_emb = embedding(token_ids)
    seg_emb = segment_embedding(segment_ids)
    pos_ids = torch.arange(0, token_ids.size(1)).unsqueeze(0)
    pos_emb = pos_enc(pos_ids)
    h = tok_emb + seg_emb + pos_emb
    for block in encoder_blocks:
        h = block(h)
    mlm_logits = mlm_linear(h)
    nsp_logits = nsp_linear(h[:, 0, :])

    print(tokens)

    # [MASK] 위치 MLM 예측
    mask_pos = mlm_mask.view(-1).nonzero(as_tuple=True)[0].item()  # [MASK] 위치 인덱스
    pred_token_id = mlm_logits.view(-1, vocab_size)[mask_pos].argmax().item()
    print(f"[MASK] 위치 예측: {inv_vocab[pred_token_id]} (정답: {inv_vocab[mlm_labels.view(-1)[mask_pos].item()]})")

    # NSP 예측
    nsp_pred = nsp_logits.argmax(dim=-1).item()
    print(f"NSP 예측: {'IsNext' if nsp_pred==1 else 'NotNext'} (정답: {'IsNext' if is_next==1 else 'NotNext'})")



# 학습하는 것들
# 토큰 임베딩 가중치
# 포지션 임베딩 가중치
# 세그먼트(문장구분) 임베딩 가중치
# 인코더 블록 내부(어텐션, FFN 등) 모든 가중치
# MLM Linear 레이어 가중치
# NSP Linear 레이어 가중치