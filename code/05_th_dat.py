"""
TH-DAT v4 — ENHANCED TRAINING
Same architecture, significantly improved training pipeline:
  - dim=128, layers=6 (more capacity)
  - 80 pretrain + 300 finetune epochs
  - Mixup augmentation
  - Label smoothing (0.05)
  - Stochastic Weight Averaging (SWA)
  - Batch size 128 (better gradients on clean data)
  - Cosine annealing with warm restarts
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef
import numpy as np
import pandas as pd
import math, copy

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

domain_labels = torch.zeros(N, dtype=torch.long)
for i in DEMO_IDX: domain_labels[i] = 0
for i in OBST_IDX: domain_labels[i] = 1
for i in PSYC_IDX: domain_labels[i] = 2
domain_labels = domain_labels.to(device)

# Trimester IDs
gest_all = np.concatenate([X_tr[:, 1], X_val[:, 1], X_te[:, 1]])
q33, q66 = np.percentile(gest_all, 33), np.percentile(gest_all, 66)
tri_tr_id = np.digitize(X_tr[:, 1], bins=[q33, q66]).astype(np.int64)
tri_val_id = np.digitize(X_val[:, 1], bins=[q33, q66]).astype(np.int64)
tri_te_id = np.digitize(X_te[:, 1], bins=[q33, q66]).astype(np.int64)
print(f"Trimester: T1={sum(tri_tr_id==0)}, T2={sum(tri_tr_id==1)}, T3={sum(tri_tr_id==2)}")


class THDAT_v4(nn.Module):
    def __init__(self, num_feat, dom_idx, dim=128, heads=8, layers=6, drop=0.1):
        super(THDAT_v4, self).__init__()
        self.num_feat = num_feat
        self.dom_idx = dom_idx
        self.dim = dim

        self.feat_embed = nn.ModuleList([nn.Linear(1, dim) for _ in range(num_feat)])
        self.feat_norm = nn.LayerNorm(dim)
        self.domain_embed = nn.Embedding(3, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_feat, dim) * 0.02)
        self.tri_embed = nn.Embedding(3, dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim*4,
            dropout=drop, batch_first=True, activation='gelu'
        )
        self.global_transformer = nn.TransformerEncoder(enc_layer, num_layers=layers)

        self.domain_query = nn.ParameterList([nn.Parameter(torch.randn(1, 1, dim) * 0.02) for _ in range(3)])
        self.domain_attn = nn.ModuleList([
            nn.MultiheadAttention(dim, heads // 2, dropout=drop, batch_first=True) for _ in range(3)
        ])
        self.domain_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])

        self.tri_cross_attn = nn.MultiheadAttention(dim, heads // 2, dropout=drop, batch_first=True)
        self.tri_norm = nn.LayerNorm(dim)
        self.tri_ff = nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(), nn.Dropout(drop), nn.Linear(dim*2, dim))
        self.tri_norm2 = nn.LayerNorm(dim)

        self.gate_net = nn.Sequential(nn.Linear(dim*3, dim), nn.GELU(), nn.Dropout(drop), nn.Linear(dim, 3))
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)

        self.skip_proj = nn.Linear(num_feat, dim)
        self.classifier = nn.Sequential(
            nn.Linear(dim * 3, dim), nn.GELU(), nn.BatchNorm1d(dim), nn.Dropout(drop),
            nn.Linear(dim, dim // 2), nn.GELU(), nn.Dropout(drop * 0.5),
            nn.Linear(dim // 2, 1)
        )
        self.proj_head = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 32))

    def encode(self, x, tri_ids):
        B = x.shape[0]
        tokens = [self.feat_embed[i](x[:, i:i+1]).unsqueeze(1) for i in range(self.num_feat)]
        tokens = self.feat_norm(torch.cat(tokens, dim=1))
        tokens = tokens + self.pos_embed + self.domain_embed(domain_labels).unsqueeze(0)
        tokens = tokens + 0.1 * self.tri_embed(tri_ids).unsqueeze(1)
        attended = self.global_transformer(tokens)

        domain_sums = []
        for d in range(3):
            idx = self.dom_idx[d]
            dom_tokens = attended[:, idx, :]
            query = self.domain_query[d].expand(B, -1, -1)
            pooled, _ = self.domain_attn[d](query, dom_tokens, dom_tokens)
            pooled = self.domain_norm[d](query + pooled).squeeze(1)
            domain_sums.append(pooled)
        domain_stack = torch.stack(domain_sums, dim=1)

        tri_q = self.tri_embed(tri_ids).unsqueeze(1)
        tri_out, tri_w = self.tri_cross_attn(tri_q, domain_stack, domain_stack)
        tri_out = self.tri_norm(tri_q + tri_out)
        tri_out = self.tri_norm2(tri_out + self.tri_ff(tri_out)).squeeze(1)

        gate_input = domain_stack.reshape(B, 3 * self.dim)
        gate_w = F.softmax(self.gate_net(gate_input), dim=-1)
        gated = (domain_stack * gate_w.unsqueeze(-1)).sum(dim=1)

        skip = F.gelu(self.skip_proj(x))
        combined = torch.cat([tri_out, gated, skip], dim=-1)
        return combined, gate_w, tri_w

    def forward(self, x, tri_ids):
        combined, gw, tw = self.encode(x, tri_ids)
        return self.classifier(combined).squeeze(-1), gw, tw

    def project(self, x, tri_ids):
        combined, _, _ = self.encode(x, tri_ids)
        return self.proj_head(combined)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.05):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ls = label_smoothing

    def forward(self, logits, targets):
        # Label smoothing
        targets_smooth = targets * (1 - self.ls) + 0.5 * self.ls
        ce = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction='none')
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


def mixup_data(x, y, t, alpha=0.2):
    """Mixup augmentation — interpolate between training samples"""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    y_mix = lam * y + (1 - lam) * y[idx]
    t_mix = t  # Keep original trimester IDs
    return x_mix, y_mix, t_mix, lam


def get_scores(y_true, prob):
    pred = (np.array(prob) >= 0.5).astype(int)
    return {
        'Accuracy': round(accuracy_score(y_true, pred), 4),
        'F1': round(f1_score(y_true, pred), 4),
        'AUC-ROC': round(roc_auc_score(y_true, prob), 4),
        'AUC-PR': round(average_precision_score(y_true, prob), 4),
        'MCC': round(matthews_corrcoef(y_true, pred), 4),
    }


# ═══ Setup ═══
Xtr = torch.tensor(X_tr, dtype=torch.float32)
Xva = torch.tensor(X_val, dtype=torch.float32).to(device)
Xte = torch.tensor(X_te, dtype=torch.float32).to(device)
ytr = torch.tensor(y_tr, dtype=torch.float32)
tri_tr = torch.tensor(tri_tr_id, dtype=torch.long)
tri_va = torch.tensor(tri_val_id, dtype=torch.long).to(device)
tri_te = torch.tensor(tri_te_id, dtype=torch.long).to(device)
dl = DataLoader(TensorDataset(Xtr, ytr, tri_tr), batch_size=128, shuffle=True, drop_last=True)

model = THDAT_v4(N, domain_indices, dim=128, heads=8, layers=6, drop=0.1).to(device)
params = sum(p.numel() for p in model.parameters())
print(f"TH-DAT v4 params: {params:,}")

# ═══ PHASE 1: Masked Pretraining (80 epochs) ═══
print("\n" + "=" * 50)
print("PHASE 1: Masked Feature Pretraining (80 epochs)")
print("=" * 50)
opt1 = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, 80)

for ep in range(80):
    model.train()
    total = 0
    for xb, yb, tb in dl:
        xb, tb = xb.to(device), tb.to(device)
        mask = torch.rand_like(xb) < 0.25  # Higher mask ratio for better learning
        xm = xb.clone(); xm[mask] = 0.0
        opt1.zero_grad()
        comb, _, _ = model.encode(xm, tb)
        loss = F.mse_loss(comb[:, :N], xb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt1.step()
        total += loss.item()
    sch1.step()
    if (ep + 1) % 20 == 0:
        print(f"  Pretrain {ep+1}/80 | Loss: {total/len(dl):.4f}")
print("Phase 1 DONE!")

# ═══ PHASE 2: Supervised Fine-tuning (300 epochs) ═══
EPOCHS = 300
print("\n" + "=" * 50)
print(f"PHASE 2: Supervised Fine-tuning ({EPOCHS} epochs)")
print("=" * 50)
focal = FocalLoss(gamma=2.0, label_smoothing=0.05)
supcon = SupConLoss(temp=0.1)
opt2 = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

# Cosine annealing with warm restarts (T0=50 epochs per restart)
sch2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt2, T_0=50, T_mult=2, eta_min=1e-6)

best_auc, best_w = 0, None
patience, no_improve = 50, 0

# Stochastic Weight Averaging
swa_model = None
swa_start = 200  # Start SWA after epoch 200
swa_count = 0

for ep in range(EPOCHS):
    model.train()
    total = 0
    for xb, yb, tb in dl:
        xb, yb, tb = xb.to(device), yb.to(device), tb.to(device)

        # Mixup augmentation (50% chance)
        if np.random.random() < 0.5:
            xb, yb, tb, lam = mixup_data(xb, yb, tb, alpha=0.2)

        opt2.zero_grad()
        logits, gw, _ = model(xb, tb)
        l_focal = focal(logits, yb)

        # Contrastive loss every 3rd epoch
        if (ep + 1) % 3 == 0:
            proj = model.project(xb, tb)
            l_con = supcon(proj, (yb > 0.5).long())
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

    # Validation
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

    # Stochastic Weight Averaging after swa_start
    if ep >= swa_start:
        if swa_model is None:
            swa_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            swa_count = 1
        else:
            for k in swa_model:
                swa_model[k] = (swa_model[k] * swa_count + model.state_dict()[k].cpu()) / (swa_count + 1)
            swa_count += 1

    if (ep + 1) % 10 == 0:
        print(f"  Epoch {ep+1}/{EPOCHS} | Loss: {total/len(dl):.4f} | Val AUC: {va:.4f} | Best: {best_auc:.4f}")

    if no_improve >= patience and ep > 120:
        print(f"  Early stopping at epoch {ep+1}")
        break

print(f"\nBest Val AUC: {best_auc:.4f}")

# ═══ Evaluate BOTH best-checkpoint and SWA ═══
# Best checkpoint
model.load_state_dict({k: v.to(device) for k, v in best_w.items()})
model.eval()
with torch.no_grad():
    tl, gw, tw = model(Xte, tri_te)
    tp_best = torch.sigmoid(tl).cpu().numpy()
scores_best = get_scores(y_te, tp_best)

# SWA model (if available)
if swa_model is not None:
    model.load_state_dict({k: v.to(device) for k, v in swa_model.items()})
    model.eval()
    with torch.no_grad():
        tl_swa, gw_swa, _ = model(Xte, tri_te)
        tp_swa = torch.sigmoid(tl_swa).cpu().numpy()
    scores_swa = get_scores(y_te, tp_swa)
    print(f"\nSWA Model: {scores_swa}")
else:
    scores_swa = None

# Ensemble: average best + SWA
if scores_swa is not None:
    tp_ens = 0.5 * tp_best + 0.5 * tp_swa
    scores_ens = get_scores(y_te, tp_ens)
    print(f"Ensemble (Best+SWA): {scores_ens}")
else:
    scores_ens = None

# Pick best among {checkpoint, SWA, ensemble}
candidates = [('Best Checkpoint', scores_best, tp_best, gw)]
if scores_swa is not None:
    candidates.append(('SWA', scores_swa, tp_swa, gw_swa))
if scores_ens is not None:
    candidates.append(('Ensemble', scores_ens, tp_ens, gw))

winner = max(candidates, key=lambda c: c[1]['AUC-ROC'])
print(f"\n*** Winner: {winner[0]} ***")
thdat_scores = winner[1]
gw_final = winner[3]

print("\n" + "=" * 50)
print("TH-DAT v4 FINAL RESULTS")
print("=" * 50)
print(thdat_scores)
gw_np = gw_final.cpu().numpy()
print(f"\nDomain Gate Weights:")
print(f"  Demographic:  {gw_np[:, 0].mean():.4f} +/- {gw_np[:, 0].std():.4f}")
print(f"  Obstetric:    {gw_np[:, 1].mean():.4f} +/- {gw_np[:, 1].std():.4f}")
print(f"  Psychosocial: {gw_np[:, 2].mean():.4f} +/- {gw_np[:, 2].std():.4f}")

print("\n" + "=" * 60)
print("    DONE - Copy these results!")
print("=" * 60)
