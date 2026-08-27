"""
Uganda EPDS Dataset - External Validation Pipeline
Runs key models: RF, XGBoost, ANN, TabTransformer, TH-DAT
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import numpy as np
import pandas as pd
import math
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# ═══════════════════════════════════════════
# Load + Preprocess Uganda EPDS
# ═══════════════════════════════════════════
df = pd.read_csv('/content/records.csv')
print(f"Raw shape: {df.shape}")

# Derive Gestational Age from dates
df['evaluation_date'] = pd.to_datetime(df['evaluation_date'], format='%d-%m-%Y', errors='coerce')
df['dueDate'] = pd.to_datetime(df['dueDate'], format='%d-%m-%Y', errors='coerce')
df['gest_age_weeks'] = 40 - ((df['dueDate'] - df['evaluation_date']).dt.days / 7)
df['gest_age_weeks'] = df['gest_age_weeks'].clip(1, 42)
print(f"Gestational Age range: {df['gest_age_weeks'].min():.0f} - {df['gest_age_weeks'].max():.0f} weeks")

# Derive Trimester
df['Trimester'] = np.where(df['gest_age_weeks'] <= 13, 0,
                  np.where(df['gest_age_weeks'] <= 26, 1, 2))
print(f"Trimester: T1={sum(df['Trimester']==0)}, T2={sum(df['Trimester']==1)}, T3={sum(df['Trimester']==2)}")

# Depression label (EPDS >= 13)
df['Label'] = (df['Total Score'] >= 13).astype(int)
print(f"Depressed: {df['Label'].sum()} ({df['Label'].mean()*100:.1f}%)")
print(f"Not Depressed: {(1-df['Label']).sum()} ({(1-df['Label'].mean())*100:.1f}%)")

# Convert age to numeric
df['age'] = pd.to_numeric(df['age'], errors='coerce')

# Select features
feature_cols = ['age', 'occupation', 'education', 'relationshipStatus',
                'spouseOccupation', 'polygamy', 'otherChildren',
                'numStillBorn', 'numMissCarry', 'hospital',
                'gest_age_weeks', 'Trimester', 'Anxiety']

# Encode categoricals
le = LabelEncoder()
cat_cols = ['occupation', 'education', 'relationshipStatus', 'spouseOccupation', 'polygamy', 'hospital']
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

X = SimpleImputer(strategy='median').fit_transform(df[feature_cols].values)
y = df['Label'].values
N = X.shape[1]

print(f"\nFeatures ({N}):")
for i, c in enumerate(feature_cols):
    print(f"  {i}: {c}")

# ============================================================
# CORRECT PIPELINE: Split FIRST, then SMOTE on training ONLY
# ============================================================
# Step 1: Stratified split on ORIGINAL data
X_tr, X_temp, y_tr, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_te, y_val, y_te = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

print(f"\nBefore SMOTE:")
print(f"  Train: {X_tr.shape[0]} (Dep={sum(y_tr==1)}, Not={sum(y_tr==0)})")
print(f"  Val:   {X_val.shape[0]} (ORIGINAL)")
print(f"  Test:  {X_te.shape[0]} (ORIGINAL)")

# Step 2: SMOTE ONLY on training set
X_tr, y_tr = SMOTE(random_state=42).fit_resample(X_tr, y_tr)
print(f"\nAfter SMOTE (training only):")
print(f"  Train: {X_tr.shape[0]} (Dep={sum(y_tr==1)}, Not={sum(y_tr==0)})")

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_val = scaler.transform(X_val)
X_te = scaler.transform(X_te)

print(f"X_tr: {X_tr.shape} | X_val: {X_val.shape} | X_te: {X_te.shape}")
print("*** SMOTE applied to TRAINING SET ONLY ***")

# Domain indices for Uganda
# Demographic: age(0), occupation(1), education(2), hospital(9)
# Obstetric: otherChildren(6), numStillBorn(7), numMissCarry(8), gest_age(10), trimester(11)
# Psychosocial: relationshipStatus(3), spouseOccupation(4), polygamy(5), Anxiety(12)
DEMO_IDX = [0, 1, 2, 9]
OBST_IDX = [6, 7, 8, 10, 11]
PSYC_IDX = [3, 4, 5, 12]
domain_indices = [DEMO_IDX, OBST_IDX, PSYC_IDX]
print(f"Domains: Demo={len(DEMO_IDX)}, Obst={len(OBST_IDX)}, Psyc={len(PSYC_IDX)}")


def get_scores(y_true, prob):
    pred = (np.array(prob) >= 0.5).astype(int)
    return {
        'Accuracy': round(accuracy_score(y_true, pred), 4),
        'F1': round(f1_score(y_true, pred), 4),
        'AUC-ROC': round(roc_auc_score(y_true, prob), 4),
        'AUC-PR': round(average_precision_score(y_true, prob), 4)
    }


results = {}

# ═══════════════════════════════════════════
# 1. Random Forest
# ═══════════════════════════════════════════
print("\n>>> Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
rf_prob = rf.predict_proba(X_te)[:, 1]
results['Random Forest'] = get_scores(y_te, rf_prob)
print(f"  RF: {results['Random Forest']}")

# ═══════════════════════════════════════════
# 2. XGBoost
# ═══════════════════════════════════════════
print("\n>>> XGBoost...")
xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                               use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_tr, y_tr)
xgb_prob = xgb_model.predict_proba(X_te)[:, 1]
results['XGBoost'] = get_scores(y_te, xgb_prob)
print(f"  XGBoost: {results['XGBoost']}")

# ═══════════════════════════════════════════
# 3. ANN
# ═══════════════════════════════════════════
print("\n>>> ANN...")
Xtr_t = torch.tensor(X_tr, dtype=torch.float32)
Xva_t = torch.tensor(X_val, dtype=torch.float32).to(device)
Xte_t = torch.tensor(X_te, dtype=torch.float32).to(device)
ytr_t = torch.tensor(y_tr, dtype=torch.float32)
dl = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=256, shuffle=True)
bce = nn.BCELoss()

ann = nn.Sequential(nn.Linear(N, 128), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(64, 1), nn.Sigmoid()).to(device)
opt = torch.optim.Adam(ann.parameters(), lr=1e-3)
best_a, best_wa = 0, None
for ep in range(50):
    ann.train()
    for xb, yb in dl:
        opt.zero_grad(); bce(ann(xb.to(device)).squeeze(), yb.to(device)).backward(); opt.step()
    ann.eval()
    with torch.no_grad():
        va = roc_auc_score(y_val, ann(Xva_t).squeeze().cpu().numpy())
    if va > best_a: best_a = va; best_wa = {k: v.clone() for k, v in ann.state_dict().items()}
ann.load_state_dict(best_wa); ann.eval()
with torch.no_grad():
    ann_prob = ann(Xte_t).squeeze().cpu().numpy()
results['ANN'] = get_scores(y_te, ann_prob)
print(f"  ANN: {results['ANN']}")

# ═══════════════════════════════════════════
# 4. TabTransformer
# ═══════════════════════════════════════════
print("\n>>> TabTransformer...")

class TabTF(nn.Module):
    def __init__(self, num_feat):
        super(TabTF, self).__init__()
        self.emb = nn.Linear(1, 32)
        self.tf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True),
            num_layers=3)
        self.clf = nn.Sequential(nn.Linear(num_feat*32, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1), nn.Sigmoid())
    def forward(self, x):
        B = x.shape[0]
        e = self.tf(self.emb(x.unsqueeze(-1)))
        return self.clf(e.reshape(B, -1)).squeeze(-1)

m_tab = TabTF(N).to(device)
opt_tab = torch.optim.Adam(m_tab.parameters(), lr=1e-3, weight_decay=1e-4)
best_t, best_wt = 0, None
for ep in range(50):
    m_tab.train()
    for xb, yb in dl:
        opt_tab.zero_grad(); bce(m_tab(xb.to(device)), yb.to(device)).backward(); opt_tab.step()
    m_tab.eval()
    with torch.no_grad():
        va = roc_auc_score(y_val, m_tab(Xva_t).cpu().numpy())
    if va > best_t: best_t = va; best_wt = {k: v.clone() for k, v in m_tab.state_dict().items()}
    if (ep+1)%10==0: print(f"  Epoch {ep+1}/50 | Val AUC: {va:.4f}")
m_tab.load_state_dict(best_wt); m_tab.eval()
with torch.no_grad():
    tab_prob = m_tab(Xte_t).cpu().numpy()
results['TabTransformer'] = get_scores(y_te, tab_prob)
print(f"  TabTransformer: {results['TabTransformer']}")


# ═══════════════════════════════════════════
# 5. TH-DAT (Proposed)
# ═══════════════════════════════════════════
print("\n>>> TH-DAT (Proposed)...")

# Trimester IDs
gest_all = np.concatenate([X_tr[:, 10], X_val[:, 10], X_te[:, 10]])
q33, q66 = np.percentile(gest_all, 33), np.percentile(gest_all, 66)
tri_tr_id = torch.tensor(np.digitize(X_tr[:, 10], [q33, q66]), dtype=torch.long)
tri_va_id = torch.tensor(np.digitize(X_val[:, 10], [q33, q66]), dtype=torch.long).to(device)
tri_te_id = torch.tensor(np.digitize(X_te[:, 10], [q33, q66]), dtype=torch.long).to(device)

domain_labels_ug = torch.zeros(N, dtype=torch.long)
for i in DEMO_IDX: domain_labels_ug[i] = 0
for i in OBST_IDX: domain_labels_ug[i] = 1
for i in PSYC_IDX: domain_labels_ug[i] = 2
domain_labels_ug = domain_labels_ug.to(device)

class THDAT(nn.Module):
    def __init__(self, nf, dom_idx, dlabels, dim=64, heads=8, layers=4, drop=0.1):
        super(THDAT, self).__init__()
        self.nf = nf; self.dom_idx = dom_idx; self.dim = dim; self.dlabels = dlabels
        self.feat_embed = nn.ModuleList([nn.Linear(1, dim) for _ in range(nf)])
        self.feat_norm = nn.LayerNorm(dim)
        self.domain_embed = nn.Embedding(3, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, nf, dim)*0.02)
        self.tri_embed = nn.Embedding(3, dim)
        self.global_tf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4, dropout=drop, batch_first=True, activation='gelu'),
            num_layers=layers)
        self.dom_q = nn.ParameterList([nn.Parameter(torch.randn(1,1,dim)*0.02) for _ in range(3)])
        self.dom_attn = nn.ModuleList([nn.MultiheadAttention(dim, heads//2, dropout=drop, batch_first=True) for _ in range(3)])
        self.dom_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.tri_cross = nn.MultiheadAttention(dim, heads//2, dropout=drop, batch_first=True)
        self.tri_n1 = nn.LayerNorm(dim)
        self.tri_ff = nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(), nn.Dropout(drop), nn.Linear(dim*2, dim))
        self.tri_n2 = nn.LayerNorm(dim)
        self.gate = nn.Sequential(nn.Linear(dim*3, dim), nn.GELU(), nn.Dropout(drop), nn.Linear(dim, 3))
        nn.init.zeros_(self.gate[-1].weight); nn.init.zeros_(self.gate[-1].bias)
        self.skip = nn.Linear(nf, dim)
        self.head = nn.Sequential(nn.Linear(dim*3, dim), nn.GELU(), nn.BatchNorm1d(dim), nn.Dropout(drop),
                                  nn.Linear(dim, dim//2), nn.GELU(), nn.Dropout(drop*0.5), nn.Linear(dim//2, 1))

    def forward(self, x, tri):
        B = x.shape[0]
        tok = [self.feat_embed[i](x[:,i:i+1]).unsqueeze(1) for i in range(self.nf)]
        tok = self.feat_norm(torch.cat(tok, dim=1)) + self.pos_embed + self.domain_embed(self.dlabels).unsqueeze(0)
        tok = tok + 0.1 * self.tri_embed(tri).unsqueeze(1)
        att = self.global_tf(tok)
        ds = []
        for d in range(3):
            idx = self.dom_idx[d]; dt = att[:, idx, :]
            q = self.dom_q[d].expand(B,-1,-1)
            p, _ = self.dom_attn[d](q, dt, dt)
            ds.append(self.dom_norm[d](q+p).squeeze(1))
        dstack = torch.stack(ds, dim=1)
        tq = self.tri_embed(tri).unsqueeze(1)
        to2, _ = self.tri_cross(tq, dstack, dstack)
        to2 = self.tri_n1(tq+to2)
        to2 = self.tri_n2(to2+self.tri_ff(to2)).squeeze(1)
        gw = F.softmax(self.gate(dstack.reshape(B,3*self.dim)), dim=-1)
        gated = (dstack * gw.unsqueeze(-1)).sum(dim=1)
        skip = F.gelu(self.skip(x))
        return self.head(torch.cat([to2, gated, skip], dim=-1)).squeeze(-1), gw

dl2 = DataLoader(TensorDataset(Xtr_t, ytr_t, tri_tr_id), batch_size=256, shuffle=True)
model = THDAT(N, domain_indices, domain_labels_ug, dim=64, heads=8, layers=4, drop=0.1).to(device)

# Phase 1: Pretrain
opt1 = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, 50)
for ep in range(50):
    model.train()
    for xb, yb, tb in dl2:
        xb, tb = xb.to(device), tb.to(device)
        mask = torch.rand_like(xb) < 0.20; xm = xb.clone(); xm[mask] = 0.0
        opt1.zero_grad()
        out, _ = model(xm, tb)
        F.mse_loss(out, yb.to(device)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt1.step()
    sch1.step()
    if (ep+1)%25==0: print(f"  Pretrain {ep+1}/50")
print("  Pretrain done!")

# Phase 2: Fine-tune
class FocalLoss(nn.Module):
    def __init__(self, g=2.0):
        super().__init__(); self.g = g
    def forward(self, l, t):
        ce = F.binary_cross_entropy_with_logits(l, t, reduction='none')
        pt = torch.sigmoid(l)*t + (1-torch.sigmoid(l))*(1-t)
        return (((1-pt)**self.g)*ce).mean()

focal = FocalLoss()
opt2 = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
def lr_fn(ep):
    w=15
    if ep<w: return (ep+1)/w
    return 0.5*(1+math.cos(math.pi*(ep-w)/(150-w)))
sch2 = torch.optim.lr_scheduler.LambdaLR(opt2, lr_fn)
best_v, best_wm = 0, None

for ep in range(150):
    model.train()
    for xb, yb, tb in dl2:
        xb, yb, tb = xb.to(device), yb.to(device), tb.to(device)
        opt2.zero_grad()
        logits, gw = model(xb, tb)
        loss = focal(logits, yb)
        mw = gw.mean(0); loss = loss + 0.01*(mw*torch.log(mw+1e-8)).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt2.step()
    sch2.step()
    model.eval()
    with torch.no_grad():
        vl, _ = model(Xva_t, tri_va_id)
        va = roc_auc_score(y_val, torch.sigmoid(vl).cpu().numpy())
    if va > best_v: best_v = va; best_wm = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if (ep+1)%30==0: print(f"  Epoch {ep+1}/150 | Val AUC: {va:.4f} | Best: {best_v:.4f}")

model.load_state_dict({k: v.to(device) for k, v in best_wm.items()})
model.eval()
with torch.no_grad():
    tl, gw = model(Xte_t, tri_te_id)
    tp = torch.sigmoid(tl).cpu().numpy()
results['TH-DAT (Ours)'] = get_scores(y_te, tp)
print(f"  TH-DAT: {results['TH-DAT (Ours)']}")

gw_np = gw.cpu().numpy()
print(f"\n  Gate Weights: Demo={gw_np[:,0].mean():.3f} Obst={gw_np[:,1].mean():.3f} Psyc={gw_np[:,2].mean():.3f}")

# ═══════════════════════════════════════════
# Final Results
# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("  UGANDA EPDS - EXTERNAL VALIDATION RESULTS")
print("=" * 60)
print(pd.DataFrame(results).T.sort_values('AUC-ROC', ascending=False).to_string())

print("\n" + "=" * 60)
print("  CROSS-DATASET COMPARISON")
print("=" * 60)
peri = {'Random Forest': 0.9670, 'XGBoost': 0.9072, 'ANN': 0.8514,
        'TabTransformer': 0.9527, 'TH-DAT (Ours)': 0.9769}
# NOTE: Update these PERI_DEP values after rerunning with corrected SMOTE pipeline
print(f"\n{'Model':<20} {'PERI_DEP':>10} {'Uganda':>10} {'Avg':>10}")
print("-" * 52)
for name in results:
    p = peri.get(name, 'N/A')
    u = results[name]['AUC-ROC']
    avg = f"{(p+u)/2:.4f}" if isinstance(p, float) else "N/A"
    p_str = f"{p:.4f}" if isinstance(p, float) else p
    print(f"{name:<20} {p_str:>10} {u:>10.4f} {avg:>10}")
