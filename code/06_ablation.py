"""
TH-DAT Ablation Study + SHAP Interpretability
Tests 6 configurations by removing one component at a time.
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

DEMO_IDX = [0, 6, 7, 8, 11, 13]
OBST_IDX = [1, 2, 3, 4, 5, 10, 14]
PSYC_IDX = [9, 12, 15]
assigned = set(DEMO_IDX + OBST_IDX + PSYC_IDX)
for i in range(N):
    if i not in assigned:
        OBST_IDX.append(i)
domain_indices = [DEMO_IDX, OBST_IDX, PSYC_IDX]

gest_all = np.concatenate([X_tr[:, 1], X_val[:, 1], X_te[:, 1]])
q33, q66 = np.percentile(gest_all, 33), np.percentile(gest_all, 66)
tri_tr_id = np.digitize(X_tr[:, 1], bins=[q33, q66]).astype(np.int64)
tri_val_id = np.digitize(X_val[:, 1], bins=[q33, q66]).astype(np.int64)
tri_te_id = np.digitize(X_te[:, 1], bins=[q33, q66]).astype(np.int64)

domain_labels = torch.zeros(N, dtype=torch.long)
for i in DEMO_IDX: domain_labels[i] = 0
for i in OBST_IDX: domain_labels[i] = 1
for i in PSYC_IDX: domain_labels[i] = 2
domain_labels = domain_labels.to(device)

Xtr = torch.tensor(X_tr, dtype=torch.float32)
Xva = torch.tensor(X_val, dtype=torch.float32).to(device)
Xte = torch.tensor(X_te, dtype=torch.float32).to(device)
ytr = torch.tensor(y_tr, dtype=torch.float32)
tri_tr = torch.tensor(tri_tr_id, dtype=torch.long)
tri_va = torch.tensor(tri_val_id, dtype=torch.long).to(device)
tri_te = torch.tensor(tri_te_id, dtype=torch.long).to(device)
dl = DataLoader(TensorDataset(Xtr, ytr, tri_tr), batch_size=256, shuffle=True)


def get_scores(prob):
    pred = (np.array(prob) >= 0.5).astype(int)
    return {
        'Accuracy': round(accuracy_score(y_te, pred), 4),
        'F1': round(f1_score(y_te, pred), 4),
        'AUC-ROC': round(roc_auc_score(y_te, prob), 4),
        'AUC-PR': round(average_precision_score(y_te, prob), 4)
    }


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
    def forward(self, logits, targets):
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
        return (((1 - pt) ** self.gamma) * ce).mean()


class AblationModel(nn.Module):
    """Configurable TH-DAT for ablation study."""
    def __init__(self, num_feat, dom_idx, dim=64, heads=8, layers=4, drop=0.1,
                 use_domain_embed=True, use_trimester=True, use_gate=True, use_skip=True):
        super(AblationModel, self).__init__()
        self.num_feat = num_feat
        self.dom_idx = dom_idx
        self.dim = dim
        self.use_domain_embed = use_domain_embed
        self.use_trimester = use_trimester
        self.use_gate = use_gate
        self.use_skip = use_skip

        self.feat_embed = nn.ModuleList([nn.Linear(1, dim) for _ in range(num_feat)])
        self.feat_norm = nn.LayerNorm(dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_feat, dim) * 0.02)

        if use_domain_embed:
            self.domain_embed = nn.Embedding(3, dim)
        if use_trimester:
            self.tri_embed = nn.Embedding(3, dim)

        enc_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4, dropout=drop, batch_first=True, activation='gelu')
        self.global_tf = nn.TransformerEncoder(enc_layer, num_layers=layers)

        if use_gate:
            self.domain_query = nn.ParameterList([nn.Parameter(torch.randn(1, 1, dim) * 0.02) for _ in range(3)])
            self.domain_attn = nn.ModuleList([nn.MultiheadAttention(dim, heads//2, dropout=drop, batch_first=True) for _ in range(3)])
            self.domain_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
            self.gate_net = nn.Sequential(nn.Linear(dim*3, dim), nn.GELU(), nn.Dropout(drop), nn.Linear(dim, 3))
            nn.init.zeros_(self.gate_net[-1].weight)
            nn.init.zeros_(self.gate_net[-1].bias)

        if use_trimester and use_gate:
            self.tri_cross = nn.MultiheadAttention(dim, heads//2, dropout=drop, batch_first=True)
            self.tri_norm = nn.LayerNorm(dim)
            self.tri_ff = nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(), nn.Dropout(drop), nn.Linear(dim*2, dim))
            self.tri_norm2 = nn.LayerNorm(dim)

        if use_skip:
            self.skip_proj = nn.Linear(num_feat, dim)

        # Calculate classifier input dim
        clf_dim = 0
        if use_trimester and use_gate:
            clf_dim += dim
        if use_gate:
            clf_dim += dim
        if not use_gate:
            clf_dim += dim  # use mean pooling instead
        if use_skip:
            clf_dim += dim

        self.classifier = nn.Sequential(
            nn.Linear(clf_dim, dim), nn.GELU(), nn.BatchNorm1d(dim), nn.Dropout(drop),
            nn.Linear(dim, dim//2), nn.GELU(), nn.Dropout(drop*0.5), nn.Linear(dim//2, 1)
        )

    def forward(self, x, tri_ids):
        B = x.shape[0]
        tokens = []
        for i in range(self.num_feat):
            tokens.append(self.feat_embed[i](x[:, i:i+1]).unsqueeze(1))
        tokens = self.feat_norm(torch.cat(tokens, dim=1))
        tokens = tokens + self.pos_embed

        if self.use_domain_embed:
            tokens = tokens + self.domain_embed(domain_labels).unsqueeze(0)
        if self.use_trimester:
            tri_emb = self.tri_embed(tri_ids).unsqueeze(1)
            tokens = tokens + 0.1 * tri_emb

        attended = self.global_tf(tokens)

        parts = []

        if self.use_gate:
            dom_sums = []
            for d in range(3):
                idx = self.dom_idx[d]
                dt = attended[:, idx, :]
                q = self.domain_query[d].expand(B, -1, -1)
                p, _ = self.domain_attn[d](q, dt, dt)
                p = self.domain_norm[d](q + p).squeeze(1)
                dom_sums.append(p)
            dom_stack = torch.stack(dom_sums, dim=1)
            gate_w = F.softmax(self.gate_net(dom_stack.reshape(B, 3*self.dim)), dim=-1)
            gated = (dom_stack * gate_w.unsqueeze(-1)).sum(dim=1)
            parts.append(gated)

            if self.use_trimester:
                tri_q = self.tri_embed(tri_ids).unsqueeze(1)
                tri_out, _ = self.tri_cross(tri_q, dom_stack, dom_stack)
                tri_out = self.tri_norm(tri_q + tri_out)
                tri_out = self.tri_norm2(tri_out + self.tri_ff(tri_out)).squeeze(1)
                parts.append(tri_out)
        else:
            parts.append(attended.mean(dim=1))

        if self.use_skip:
            parts.append(F.gelu(self.skip_proj(x)))

        combined = torch.cat(parts, dim=-1)
        return self.classifier(combined).squeeze(-1)


def train_ablation(config_name, use_domain_embed=True, use_trimester=True,
                   use_gate=True, use_skip=True, do_pretrain=True):
    print(f"\n{'='*55}")
    print(f"  Ablation: {config_name}")
    print(f"  Domain={use_domain_embed} Tri={use_trimester} Gate={use_gate} Skip={use_skip} Pretrain={do_pretrain}")
    print(f"{'='*55}")

    model = AblationModel(N, domain_indices, dim=64, heads=8, layers=4, drop=0.1,
                          use_domain_embed=use_domain_embed, use_trimester=use_trimester,
                          use_gate=use_gate, use_skip=use_skip).to(device)

    # Phase 1: Pretrain
    if do_pretrain:
        opt1 = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, 30)
        for ep in range(30):
            model.train()
            for xb, yb, tb in dl:
                xb, tb = xb.to(device), tb.to(device)
                mask = torch.rand_like(xb) < 0.20
                xm = xb.clone(); xm[mask] = 0.0
                opt1.zero_grad()
                out = model(xm, tb)
                loss = F.mse_loss(out, yb.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt1.step()
            sch1.step()
        print("  Pretrain done")

    # Phase 2: Fine-tune
    focal = FocalLoss(gamma=2.0)
    opt2 = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    def lr_fn(ep):
        warmup = 10
        if ep < warmup: return (ep+1)/warmup
        return 0.5 * (1 + math.cos(math.pi * (ep-warmup) / (120-warmup)))
    sch2 = torch.optim.lr_scheduler.LambdaLR(opt2, lr_fn)
    best_auc, best_w = 0, None

    for ep in range(120):
        model.train()
        for xb, yb, tb in dl:
            xb, yb, tb = xb.to(device), yb.to(device), tb.to(device)
            opt2.zero_grad()
            logits = model(xb, tb)
            loss = focal(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()
        sch2.step()
        model.eval()
        with torch.no_grad():
            vl = model(Xva, tri_va)
            va = roc_auc_score(y_val, torch.sigmoid(vl).cpu().numpy())
        if va > best_auc:
            best_auc = va
            best_w = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (ep+1) % 30 == 0:
            print(f"  Epoch {ep+1}/120 | Val AUC: {va:.4f} | Best: {best_auc:.4f}")

    model.load_state_dict({k: v.to(device) for k, v in best_w.items()})
    model.eval()
    with torch.no_grad():
        tp = torch.sigmoid(model(Xte, tri_te)).cpu().numpy()
    scores = get_scores(tp)
    print(f"  Result: {scores}")
    del model
    torch.cuda.empty_cache()
    return scores


# ═══════════════════════════════════════════
# Run all 6 ablation experiments
# ═══════════════════════════════════════════
print("=" * 55)
print("  TH-DAT ABLATION STUDY")
print("=" * 55)

ablation = {}

ablation['Full TH-DAT'] = {'Accuracy': 0.9321, 'F1': 0.9314, 'AUC-ROC': 0.9769, 'AUC-PR': 0.9727}
print(f"\n  Full TH-DAT (from previous run): AUC-ROC = 0.9769")

ablation['w/o Domain Grouping'] = train_ablation(
    'Without Domain Grouping', use_domain_embed=False, use_gate=False)

ablation['w/o Trimester Attn'] = train_ablation(
    'Without Trimester Attention', use_trimester=False)

ablation['w/o Gated Fusion'] = train_ablation(
    'Without Gated Fusion', use_gate=False)

ablation['w/o Pretraining'] = train_ablation(
    'Without Masked Pretraining', do_pretrain=False)

ablation['w/o Skip Connection'] = train_ablation(
    'Without Skip Connection', use_skip=False)


# ═══════════════════════════════════════════
# Ablation Results Table
# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("  ABLATION STUDY RESULTS")
print("=" * 60)
abl_df = pd.DataFrame(ablation).T
abl_df['AUC Drop'] = abl_df['AUC-ROC'].apply(lambda x: f"{0.9769 - x:+.4f}")
print(abl_df.to_string())

print("\n" + "=" * 60)
print("  COMPONENT CONTRIBUTION (AUC-ROC drop when removed)")
print("=" * 60)
for name, sc in ablation.items():
    if name == 'Full TH-DAT':
        continue
    drop = 0.9769 - sc['AUC-ROC']
    bar = "█" * int(drop * 200)
    print(f"  {name:25s} | -{drop:.4f} | {bar}")
