from myTransformer import *

import torch
import torch.nn as nn
import torch.optim as optim


# 1. Vocab 및 입력 예시
vocab = {"[PAD]":0, "[CLS]":1, "[SEP]":2, "I":3, "love":4, "you":5, "[MASK]":6}
inv_vocab = {v:k for k,v in vocab.items()}
vocab_size = len(vocab)

# 2. 입력 문장: [CLS] I [MASK] you [SEP]
sentence = ["[CLS]", "I", "[MASK]", "you", "[SEP]"]
input_ids = torch.tensor([[vocab[w] for w in sentence]])  # (batch=1, seq_len=5)
print(f'''입력: {input_ids}''')

# 3. 정답(타겟) 시퀀스: [CLS] I love you [SEP]
target_ids = torch.tensor([[vocab[w] for w in ["[CLS]", "I", "love", "you", "[SEP]"]]])

# 4. 모델 구성
d_model = 4
num_heads = 2
d_ff = 8
num_layers = 2

embedding = EmbeddingLayer(vocab_size, d_model)
pos_enc = PositionalEncodingLayer(d_model, max_len=10)
encoder_blocks = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])

# 5. 인코더 통과
h = embedding(input_ids)
h = pos_enc(h)
for block in encoder_blocks:
    h = block(h)  # (batch, seq_len, d_model)

# 6. MLM용 Linear + Softmax
mlm_linear = nn.Linear(d_model, vocab_size)
softmax = nn.Softmax(dim=-1)
logits = mlm_linear(h)     # (batch, seq_len, vocab_size)
probs = softmax(logits)    # (batch, seq_len, vocab_size)

# 7. [MASK] 위치만 확인
mask_idx = sentence.index("[MASK]")   # 예시: 2
mask_pred_dist = probs[0, mask_idx]   # (vocab_size, )
print(f"[MASK] 위치 softmax 분포:\n{mask_pred_dist}\n")
print(f"가장 높은 토큰 인덱스: {mask_pred_dist.argmax().item()} → {inv_vocab[mask_pred_dist.argmax().item()]}")

# 8. (선택) CrossEntropyLoss 계산
loss_fn = nn.CrossEntropyLoss()
mask_label = target_ids[0, mask_idx].unsqueeze(0)   # 정답: love
mask_logits = logits[0, mask_idx].unsqueeze(0)      # shape: (1, vocab_size)
loss = loss_fn(mask_logits, mask_label)
print(f"CrossEntropyLoss (MLM): {loss.item()}")

# (전체 시퀀스에 대해 MLM loss를 모두 계산할 수도 있음)


#학습
optimizer = optim.Adam(list(embedding.parameters()) +
                       list(pos_enc.parameters()) +
                       list(mlm_linear.parameters()) +
                       list([p for b in encoder_blocks for p in b.parameters()]))

loss_fn = nn.CrossEntropyLoss()

for epoch in range(100):  # 에폭 반복
    optimizer.zero_grad()

    # 1. 모델 실행
    h = embedding(input_ids)
    h = pos_enc(h)
    for block in encoder_blocks:
        h = block(h)
    logits = mlm_linear(h)     # (batch, seq_len, vocab_size)

    # 2. [MASK] 위치만 학습(=BERT 학습방식, 보통 전체 시퀀스가 아니라 [MASK] 위치만 loss)
    mask_idx = sentence.index("[MASK]")
    mask_logits = logits[0, mask_idx].unsqueeze(0)   # (1, vocab_size)
    mask_label = target_ids[0, mask_idx].unsqueeze(0) # (1, )

    loss = loss_fn(mask_logits, mask_label)

    # 3. 역전파 및 가중치 업데이트
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    
    # 학습 종료 후, 새로운 입력(여전히 [MASK] 포함)에 대해
with torch.no_grad():
    h = embedding(input_ids)
    h = pos_enc(h)
    for block in encoder_blocks:
        h = block(h)
    logits = mlm_linear(h)
    probs = torch.softmax(logits, dim=-1)
    mask_idx = sentence.index("[MASK]")
    pred_token_id = probs[0, mask_idx].argmax().item()
    pred_token = inv_vocab[pred_token_id]
    print(f"학습 전 예측: {mask_pred_dist.argmax().item()} → {inv_vocab[mask_pred_dist.argmax().item()]}")
    print(f"[MASK] 예측 결과: {pred_token}")
