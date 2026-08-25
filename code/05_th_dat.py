"""
TH-DAT v4: Redesigned - Global Attention + Domain-Aware Pooling + Trimester + Gated Fusion
Strategy: Start from TabTransformer's strong base, add TH-DAT novel components on top
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
import numpy as np
import pandas as pd
import math

X_tr = np.load('/content/X_tr.npy')
X_val = np.load('/content/X_val.npy')
X_te = np.load('/content/X_te.npy')
y_tr = np.load('/content/y_tr.npy')
y_val = np.load('/content/y_val.npy')
y_te = np.load('/content/y_te.npy')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)
N = X_tr.shape[1]
print("Features:", N)

DEMO_IDX = [0, 6, 7, 8, 11, 13]
OBST_IDX = [1, 2, 3, 4, 5, 10, 14]
PSYC_IDX = [9, 12, 15]
assigned = set(DEMO_IDX + OBST_IDX + PSYC_IDX)
for i in range(N):
    if i not in assigned:
        OBST_IDX.append(i)
domain_indices = [DEMO_IDX, OBST_IDX, PSYC_IDX]
print(f"Domains: Demo={len(DEMO_IDX)}, Obst={len(OBST_IDX)}, Psyc={len(PSYC_IDX)}")

# Domain labels for each feature position (for domain-aware positional embedding)
domain_labels = torch.zeros(N, dtype=torch.long)
for i in DEMO_IDX:
    domain_labels[i] = 0
for i in OBST_IDX:
    domain_labels[i] = 1
for i in PSYC_IDX:
    domain_labels[i] = 2
domain_labels = domain_labels.to(device)

# Trimester IDs
gest_all = np.concatenate([X_tr[:, 1], X_val[:, 1], X_te[:, 1]])
q33, q66 = np.percentile(gest_all, 33), np.percentile(gest_all, 66)
tri_tr_id = np.digitize(X_tr[:, 1], bins=[q33, q66]).astype(np.int64)
tri_val_id = np.digitize(X_val[:, 1], bins=[q33, q66]).astype(np.int64)
tri_te_id = np.digitize(X_te[:, 1], bins=[q33, q66]).astype(np.int64)
print(f"Trimester: T1={sum(tri_tr_id==0)}, T2={sum(tri_tr_id==1)}, T3={sum(tri_tr_id==2)}")


class THDAT_v4(nn.Module):
    def __init__(self, num_feat, dom_idx, dim=64, heads=8, layers=4, drop=0.1):
        super(THDAT_v4, self).__init__()
        self.num_feat = num_feat
        self.dom_idx = dom_idx
        self.dim = dim

        # Component 1: Per-feature tokenizer
        self.feat_embed = nn.ModuleList([nn.Linear(1, dim) for _ in range(num_feat)])
        self.feat_norm = nn.LayerNorm(dim)

        # Domain-aware positional embedding (novel: encodes which domain each feature belongs to)
        self.domain_embed = nn.Embedding(3, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_feat, dim) * 0.02)

        # Trimester conditioning embedding
        self.tri_embed = nn.Embedding(3, dim)

        # Component 2: Global Transformer (processes ALL features with domain awareness)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim*4,
            dropout=drop, batch_first=True, activation='gelu'
        )
        self.global_transformer = nn.TransformerEncoder(enc_layer, num_layers=layers)

        # Component 3: Domain-aware attention pooling (extracts per-domain summary)
        self.domain_query = nn.ParameterList([nn.Parameter(torch.randn(1, 1, dim) * 0.02) for _ in range(3)])
        self.domain_attn = nn.ModuleList([
            nn.MultiheadAttention(dim, heads // 2, dropout=drop, batch_first=True) for _ in range(3)
        ])
        self.domain_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])

        # Component 4: Trimester cross-attention on domain summaries
        self.tri_cross_attn = nn.MultiheadAttention(dim, heads // 2, dropout=drop, batch_first=True)
        self.tri_norm = nn.LayerNorm(dim)
        self.tri_ff = nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(), nn.Dropout(drop), nn.Linear(dim*2, dim))
        self.tri_norm2 = nn.LayerNorm(dim)

        # Component 5: Gated domain fusion with entropy regularization
        self.gate_net = nn.Sequential(nn.Linear(dim*3, dim), nn.GELU(), nn.Dropout(drop), nn.Linear(dim, 3))
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)

        # Classifier with skip connection
        self.skip_proj = nn.Linear(num_feat, dim)
        self.classifier = nn.Sequential(
            nn.Linear(dim * 3, dim), nn.GELU(), nn.BatchNorm1d(dim), nn.Dropout(drop),
            nn.Linear(dim, dim // 2), nn.GELU(), nn.Dropout(drop * 0.5),
            nn.Linear(dim // 2, 1)
        )

        # Projection head for contrastive learning
        self.proj_head = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 32))

    def encode(self, x, tri_ids):
        B = x.shape[0]

        # Step 1: Per-feature embedding + domain positional encoding
        tokens = []
        for i in range(self.num_feat):
            tokens.append(self.feat_embed[i](x[:, i:i+1]).unsqueeze(1))
        tokens = torch.cat(tokens, dim=1)  # (B, N, D)
        tokens = self.feat_norm(tokens)
        tokens = tokens + self.pos_embed + self.domain_embed(domain_labels).unsqueeze(0)

        # Step 2: Add trimester conditioning to all tokens
        tri_emb = self.tri_embed(tri_ids).unsqueeze(1)  # (B, 1, D)
        tokens = tokens + 0.1 * tri_emb

        # Step 3: Global transformer (all features interact, domain-aware)
        attended = self.global_transformer(tokens)  # (B, N, D)

        # Step 4: Domain-aware attention pooling
        domain_sums = []
        for d in range(3):
            idx = self.dom_idx[d]
            dom_tokens = attended[:, idx, :]  # (B, N_d, D)
            query = self.domain_query[d].expand(B, -1, -1)  # (B, 1, D)
            pooled, _ = self.domain_attn[d](query, dom_tokens, dom_tokens)
            pooled = self.domain_norm[d](query + pooled).squeeze(1)  # (B, D)
            domain_sums.append(pooled)
        domain_stack = torch.stack(domain_sums, dim=1)  # (B, 3, D)

        # Step 5: Trimester cross-attention on domain summaries
        tri_q = self.tri_embed(tri_ids).unsqueeze(1)  # (B, 1, D)
        tri_out, tri_w = self.tri_cross_attn(tri_q, domain_stack, domain_stack)
        tri_out = self.tri_norm(tri_q + tri_out)
        tri_out = self.tri_norm2(tri_out + self.tri_ff(tri_out)).squeeze(1)  # (B, D)

        # Step 6: Gated domain fusion
        gate_input = domain_stack.reshape(B, 3 * self.dim)
        gate_w = F.softmax(self.gate_net(gate_input), dim=-1)
        gated = (domain_stack * gate_w.unsqueeze(-1)).sum(dim=1)  # (B, D)

        # Step 7: Skip connection from raw features
        skip = F.gelu(self.skip_proj(x))  # (B, D)

        # Combine: trimester-aware + gated + skip
        combined = torch.cat([tri_out, gated, skip], dim=-1)  # (B, 3*D)
        return combined, gate_w, tri_w

    def forward(self, x, tri_ids):
        combined, gw, tw = self.encode(x, tri_ids)
        return self.classifier(combined).squeeze(-1), gw, tw

    def project(self, x, tri_ids):
        combined, _, _ = self.encode(x, tri_ids)
        return self.proj_head(combined)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
        return (((1 - pt) ** self.gamma) * ce).mean()


class SupConLoss(nn.Module):
    def __init__(self, temp=0.1):
        super(SupConLoss, self).__init__()
        self.temp = temp

    def forward(self, proj, labels):
        z = F.normalize(proj, dim=1)
        B = z.shape[0]
        sim = torch.matmul(z, z.T) / self.temp
        diag = torch.eye(B, device=sim.device).bool()
        pos = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() * (~diag).float()
        neg_mask = (~diag).float()
        exp_s = torch.exp(sim) * neg_mask
        log_p = sim - torch.log(exp_s.sum(1, keepdim=True) + 1e-8)
        return -(pos * log_p).sum(1).div(pos.sum(1) + 1e-8).mean()


def get_scores(prob):
    pred = (np.array(prob) >= 0.5).astype(int)
    return {
        'Accuracy': round(accuracy_score(y_te, pred), 4),
        'F1': round(f1_score(y_te, pred), 4),
        'AUC-ROC': round(roc_auc_score(y_te, prob), 4),
        'AUC-PR': round(average_precision_score(y_te, prob), 4)
    }


# Tensors
Xtr = torch.tensor(X_tr, dtype=torch.float32)
Xva = torch.tensor(X_val, dtype=torch.float32).to(device)
Xte = torch.tensor(X_te, dtype=torch.float32).to(device)
ytr = torch.tensor(y_tr, dtype=torch.float32)
tri_tr = torch.tensor(tri_tr_id, dtype=torch.long)
tri_va = torch.tensor(tri_val_id, dtype=torch.long).to(device)
tri_te = torch.tensor(tri_te_id, dtype=torch.long).to(device)
dl = DataLoader(TensorDataset(Xtr, ytr, tri_tr), batch_size=256, shuffle=True)

model = THDAT_v4(N, domain_indices, dim=64, heads=8, layers=4, drop=0.1).to(device)
params = sum(p.numel() for p in model.parameters())
print(f"TH-DAT v4 params: {params:,}")

# ═══ PHASE 1: Masked Pretraining ═══
print("\n" + "=" * 50)
print("PHASE 1: Masked Feature Pretraining (50 epochs)")
print("=" * 50)
opt1 = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, 50)

for ep in range(50):
    model.train()
    total = 0
    for xb, yb, tb in dl:
        xb, tb = xb.to(device), tb.to(device)
        mask = torch.rand_like(xb) < 0.20
        xm = xb.clone()
        xm[mask] = 0.0
        opt1.zero_grad()
        # Pretrain: reconstruct through full encode path
        comb, _, _ = model.encode(xm, tb)
        # Use first dim features of combined for reconstruction
        loss = F.mse_loss(comb[:, :N], xb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt1.step()
        total += loss.item()
    sch1.step()
    if (ep + 1) % 10 == 0:
        print(f"  Pretrain {ep+1}/50 | Loss: {total/len(dl):.4f}")
print("Phase 1 DONE!")

# ═══ PHASE 2: Supervised Fine-tuning ═══
print("\n" + "=" * 50)
print("PHASE 2: Supervised Fine-tuning (150 epochs)")
print("=" * 50)
focal = FocalLoss(gamma=2.0)
supcon = SupConLoss(temp=0.1)
opt2 = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

def lr_fn(ep):
    warmup = 15
    if ep < warmup:
        return (ep + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (ep - warmup) / (150 - warmup)))

sch2 = torch.optim.lr_scheduler.LambdaLR(opt2, lr_fn)
best_auc, best_w = 0, None
patience, no_improve = 30, 0

for ep in range(150):
    model.train()
    total = 0
    for xb, yb, tb in dl:
        xb, yb, tb = xb.to(device), yb.to(device), tb.to(device)
        opt2.zero_grad()
        logits, gw, _ = model(xb, tb)
        l_focal = focal(logits, yb)

        # Contrastive loss (only every 3rd epoch to reduce overhead)
        if (ep + 1) % 3 == 0:
            proj = model.project(xb, tb)
            l_con = supcon(proj, yb.long())
        else:
            l_con = torch.tensor(0.0, device=device)

        # Gate entropy regularization
        mean_gw = gw.mean(dim=0)
        l_ent = (mean_gw * torch.log(mean_gw + 1e-8)).sum()

        loss = l_focal + 0.03 * l_con + 0.01 * l_ent
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt2.step()
        total += loss.item()
    sch2.step()

    model.eval()
    with torch.no_grad():
        vl, _, _ = model(Xva, tri_va)
        va = roc_auc_score(y_val, torch.sigmoid(vl).cpu().numpy())
    if va > best_auc:
        best_auc = va
        best_w = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1

    if (ep + 1) % 10 == 0:
        print(f"  Epoch {ep+1}/150 | Loss: {total/len(dl):.4f} | Val AUC: {va:.4f} | Best: {best_auc:.4f}")

    if no_improve >= patience and ep > 60:
        print(f"  Early stopping at epoch {ep+1}")
        break

print(f"\nBest Val AUC: {best_auc:.4f}")

# ═══ Test ═══
model.load_state_dict({k: v.to(device) for k, v in best_w.items()})
model.eval()
with torch.no_grad():
    tl, gw, tw = model(Xte, tri_te)
    tp = torch.sigmoid(tl).cpu().numpy()
thdat_scores = get_scores(tp)

print("\n" + "=" * 50)
print("TH-DAT v4 RESULTS")
print("=" * 50)
print(thdat_scores)
gw_np = gw.cpu().numpy()
print(f"\nDomain Gate Weights:")
print(f"  Demographic:  {gw_np[:, 0].mean():.4f} +/- {gw_np[:, 0].std():.4f}")
print(f"  Obstetric:    {gw_np[:, 1].mean():.4f} +/- {gw_np[:, 1].std():.4f}")
print(f"  Psychosocial: {gw_np[:, 2].mean():.4f} +/- {gw_np[:, 2].std():.4f}")

results = {
    'Random Forest':  {'Accuracy': 0.9247, 'F1': 0.9246, 'AUC-ROC': 0.9670, 'AUC-PR': 0.9628},
    'XGBoost':        {'Accuracy': 0.8139, 'F1': 0.8170, 'AUC-ROC': 0.9072, 'AUC-PR': 0.9031},
    'ANN':            {'Accuracy': 0.7735, 'F1': 0.7579, 'AUC-ROC': 0.8514, 'AUC-PR': 0.8601},
    'Autoencoder':    {'Accuracy': 0.8194, 'F1': 0.8187, 'AUC-ROC': 0.8898, 'AUC-PR': 0.8954},
    'TabTransformer': {'Accuracy': 0.8946, 'F1': 0.8942, 'AUC-ROC': 0.9527, 'AUC-PR': 0.9513},
    'FT-Transformer': {'Accuracy': 0.7669, 'F1': 0.7768, 'AUC-ROC': 0.8568, 'AUC-PR': 0.8493},
    'SAINT':          {'Accuracy': 0.7614, 'F1': 0.7598, 'AUC-ROC': 0.8543, 'AUC-PR': 0.8497},
    'BERT':           {'Accuracy': 0.5899, 'F1': 0.6031, 'AUC-ROC': 0.6760, 'AUC-PR': 0.7820},
    'RoBERTa':        {'Accuracy': 0.5852, 'F1': 0.6311, 'AUC-ROC': 0.6469, 'AUC-PR': 0.7645},
    'GPT-2':          {'Accuracy': 0.6342, 'F1': 0.6940, 'AUC-ROC': 0.6797, 'AUC-PR': 0.7805},
    'T5':             {'Accuracy': 0.5514, 'F1': 0.6226, 'AUC-ROC': 0.5631, 'AUC-PR': 0.6940},
    'TH-DAT (Ours)':  thdat_scores,
}
print("\n" + "=" * 60)
print("    COMPLETE RESULTS - ALL 12 MODELS")
print("=" * 60)
print(pd.DataFrame(results).T.sort_values('AUC-ROC', ascending=False).to_string())
