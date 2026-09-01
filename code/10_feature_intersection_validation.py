"""
10_feature_intersection_validation.py (v3 FIXED)
=================================================
Exact column names from actual datasets.
Fixed gate tensor shape mismatch.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                             matthews_corrcoef, average_precision_score)
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader, TensorDataset
import warnings; warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ═══════════════════════════════════════════
# 1. LOAD BOTH DATASETS
# ═══════════════════════════════════════════
print("\n" + "="*60)
print("  LOADING DATASETS")
print("="*60)

peri = pd.read_csv('/content/dataset.csv')
peri['dep'] = (peri['Labelling'] == 'Depressed').astype(int)
peri['Trimester'] = np.where(peri['Gestational Age'] <= 13, 0,
                    np.where(peri['Gestational Age'] <= 26, 1, 2))
print(f"PERI_DEP: {peri.shape}, Columns: {list(peri.columns)}")

ug = pd.read_csv('/content/records.csv')
# Clean bad values
for col in ug.columns:
    if ug[col].dtype == 'object':
        mask = ug[col].isin(['#VALUE!', '#REF!', '#N/A', '#DIV/0!', 'nan', ''])
        if mask.any():
            print(f"  Cleaned {mask.sum()} bad values in '{col}'")
            ug.loc[mask, col] = np.nan

ug['age'] = pd.to_numeric(ug['age'], errors='coerce')
ug['otherChildren'] = pd.to_numeric(ug['otherChildren'], errors='coerce')
ug['numStillBorn'] = pd.to_numeric(ug['numStillBorn'], errors='coerce')
ug['numMissCarry'] = pd.to_numeric(ug['numMissCarry'], errors='coerce')
ug['evaluation_date'] = pd.to_datetime(ug['evaluation_date'], format='%d-%m-%Y', errors='coerce')
ug['dueDate'] = pd.to_datetime(ug['dueDate'], format='%d-%m-%Y', errors='coerce')
ug['gest_age_weeks'] = (40 - (ug['dueDate'] - ug['evaluation_date']).dt.days / 7).clip(1, 42)
ug['Trimester'] = np.where(ug['gest_age_weeks'] <= 13, 0,
                  np.where(ug['gest_age_weeks'] <= 26, 1, 2))
ug['dep'] = (ug['Total Score'] >= 13).astype(int)
print(f"Uganda:   {ug.shape}")

# ═══════════════════════════════════════════
# 2. SEMANTIC FEATURE MAPPING (EXACT column names)
# ═══════════════════════════════════════════
# PERI_DEP columns (from actual CSV):
#   'Age', 'Gestational Age', 'Number of sons ', 'Number of daughters',
#   'Total Number of Children', 'Gravida', 'Previous Miscarriage', ...
# Uganda columns:
#   'age', 'otherChildren', 'numStillBorn', 'numMissCarry',
#   'gest_age_weeks', 'Trimester', ...

SHARED = [
    # (name, peri_col, ug_col)
    ('Age',            'Age',                       'age'),
    ('GestAge',        'Gestational Age',           'gest_age_weeks'),
    ('Trimester',      'Trimester',                 'Trimester'),
    ('Children',       'Total Number of Children',  'otherChildren'),
    ('Miscarriage',    'Previous Miscarriage',      'numMissCarry'),
    ('Gravida',        'Gravida',                   'otherChildren'),
]

# Validate
valid = []
for name, pc, uc in SHARED:
    # Handle trailing spaces in PERI_DEP column names
    pc_match = None
    for c in peri.columns:
        if c.strip() == pc.strip():
            pc_match = c
            break
    uc_match = uc if uc in ug.columns else None

    if pc_match and uc_match:
        valid.append((name, pc_match, uc_match))
        print(f"  OK: {name} -> PERI:'{pc_match}' | UG:'{uc_match}'")
    else:
        print(f"  SKIP: {name} (PERI:{pc_match is not None}, UG:{uc_match is not None})")

n_feat = len(valid)
print(f"\nShared features: {n_feat}")

# ═══════════════════════════════════════════
# 3. BUILD FEATURE MATRICES
# ═══════════════════════════════════════════
le = LabelEncoder()
for _, pc, uc in valid:
    if peri[pc].dtype == 'object':
        peri[pc] = le.fit_transform(peri[pc].astype(str))
    if ug[uc].dtype == 'object':
        ug[uc] = le.fit_transform(ug[uc].astype(str))

X_peri = np.column_stack([pd.to_numeric(peri[pc], errors='coerce').values for _, pc, _ in valid]).astype(np.float32)
X_ug   = np.column_stack([pd.to_numeric(ug[uc], errors='coerce').values   for _, _, uc in valid]).astype(np.float32)
y_peri = peri['dep'].values
y_ug   = ug['dep'].values

imp = SimpleImputer(strategy='median')
X_peri = imp.fit_transform(X_peri)
X_ug   = imp.transform(X_ug)

print(f"\nPERI_DEP: {X_peri.shape[0]} x {n_feat}, Dep={y_peri.sum()} ({y_peri.mean()*100:.1f}%)")
print(f"Uganda:   {X_ug.shape[0]} x {n_feat}, Dep={y_ug.sum()} ({y_ug.mean()*100:.1f}%)")

# ═══════════════════════════════════════════
# 4. SPLIT + SMOTE
# ═══════════════════════════════════════════
X_tr, X_temp, y_tr, y_temp = train_test_split(X_peri, y_peri, test_size=0.30, stratify=y_peri, random_state=42)
X_val, X_te, y_val, y_te = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
print(f"\nSplit: Train={len(y_tr)}, Val={len(y_val)}, Test={len(y_te)}")
X_tr, y_tr = SMOTE(random_state=42).fit_resample(X_tr, y_tr)
print(f"After SMOTE: Train={len(y_tr)}")

scaler = StandardScaler()
X_tr  = scaler.fit_transform(X_tr)
X_val = scaler.transform(X_val)
X_te  = scaler.transform(X_te)
X_ug_s = scaler.transform(X_ug)

# ═══════════════════════════════════════════
# 5. TH-DAT MODEL
# ═══════════════════════════════════════════
# Domains: 0=Age(DEMO), 1=GestAge(OBST), 2=Trim(OBST), 3=Children(OBST), 4=Miscarriage(OBST), 5=Gravida(OBST)
DEMO_IDX = [0]
OBST_IDX = [i for i in range(1, n_feat)]

class THDAT_Harm(nn.Module):
    def __init__(self, nf, dim=128, heads=4, layers=6, drop=0.15):
        super().__init__()
        self.nf, self.dim = nf, dim
        self.feat_embed = nn.Linear(1, dim)
        self.pos_enc = nn.Embedding(nf + 3, dim)
        enc = nn.TransformerEncoderLayer(dim, heads, dim*4, drop, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, layers)
        self.demo_pool = nn.MultiheadAttention(dim, heads, drop, batch_first=True)
        self.obst_pool = nn.MultiheadAttention(dim, heads, drop, batch_first=True)
        self.tri_query = nn.Embedding(3, dim)
        self.tri_cross = nn.MultiheadAttention(dim, heads, drop, batch_first=True)
        # Gate on raw domain summaries (dim*2 input)
        self.gate = nn.Sequential(nn.Linear(dim * 2, 2), nn.Softmax(dim=-1))
        # Classifier: fused (dim) + raw features (nf)
        self.classifier = nn.Sequential(
            nn.Linear(dim + nf, dim), nn.GELU(), nn.Dropout(drop), nn.Linear(dim, 1))

    def forward(self, x, tri=None):
        B, F = x.shape
        raw = x.clone()
        tok = self.feat_embed(x.unsqueeze(-1))
        tok = tok + self.pos_enc(torch.arange(F, device=x.device).unsqueeze(0).expand(B, -1))
        h = self.transformer(tok)

        def pool(attn, idx):
            if not idx:
                return torch.zeros(B, 1, self.dim, device=x.device)
            s = h[:, idx, :]
            q = s.mean(1, keepdim=True)
            o, _ = attn(q, s, s)
            return o

        d = pool(self.demo_pool, DEMO_IDX)   # (B, 1, dim)
        o = pool(self.obst_pool, OBST_IDX)   # (B, 1, dim)

        # Gate from raw domain summaries BEFORE tri_cross
        gate_in = torch.cat([d.squeeze(1), o.squeeze(1)], dim=-1)  # (B, dim*2)
        g = self.gate(gate_in)  # (B, 2)

        # Weighted fusion
        fused = g[:, 0:1] * d.squeeze(1) + g[:, 1:2] * o.squeeze(1)  # (B, dim)

        # Add trimester context
        if tri is not None:
            dc = torch.cat([d, o], dim=1)  # (B, 2, dim)
            tq = self.tri_query(tri.long()).unsqueeze(1)  # (B, 1, dim)
            tri_ctx, _ = self.tri_cross(tq, dc, dc)  # (B, 1, dim)
            fused = fused + tri_ctx.squeeze(1)

        out = self.classifier(torch.cat([fused, raw], dim=-1)).squeeze(-1)
        return out, g

# ═══════════════════════════════════════════
# 6. TRAIN
# ═══════════════════════════════════════════
def make_loader(X, y, bs=128, shuf=True):
    # Trimester from column 2 (Trimester is index 2 in our feature list)
    if X.shape[1] > 2:
        tri = np.clip(X[:, 2].round().astype(int), 0, 2)
    else:
        tri = np.zeros(len(X), dtype=int)
    return DataLoader(TensorDataset(
        torch.FloatTensor(X), torch.FloatTensor(y), torch.LongTensor(tri)
    ), bs, shuf)

model = THDAT_Harm(n_feat).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2)
crit = nn.BCEWithLogitsLoss()

params = sum(p.numel() for p in model.parameters())
print(f"\nParams: {params:,}")
print(f"\n{'='*60}")
print(f"  TRAINING TH-DAT ({n_feat} shared features)")
print(f"{'='*60}")

best_auc, best_st = 0, None
for ep in range(1, 201):
    model.train()
    for xb, yb, tb in make_loader(X_tr, y_tr):
        xb, yb, tb = xb.to(device), yb.to(device), tb.to(device)
        logits, _ = model(xb, tb)
        loss = crit(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()

    if ep % 10 == 0:
        model.eval()
        ap, ay = [], []
        with torch.no_grad():
            for xb, yb, tb in make_loader(X_val, y_val, shuf=False):
                xb, tb = xb.to(device), tb.to(device)
                logits, _ = model(xb, tb)
                ap.extend(torch.sigmoid(logits).cpu().numpy())
                ay.extend(yb.numpy())
        auc = roc_auc_score(ay, ap)
        if auc > best_auc:
            best_auc = auc
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 50 == 0:
            print(f"  Epoch {ep}/200 | Val AUC: {auc:.4f} | Best: {best_auc:.4f}")

model.load_state_dict(best_st)
model.eval()

# ═══════════════════════════════════════════
# 7. EVALUATE
# ═══════════════════════════════════════════
def evaluate(model, X, y, name):
    loader = make_loader(X, y, shuf=False)
    ap, ay, ag = [], [], []
    with torch.no_grad():
        for xb, yb, tb in loader:
            xb, tb = xb.to(device), tb.to(device)
            logits, gates = model(xb, tb)
            ap.extend(torch.sigmoid(logits).cpu().numpy())
            ay.extend(yb.numpy())
            ag.extend(gates.cpu().numpy())
    probs = np.array(ap)
    preds = (probs >= 0.5).astype(int)
    yt = np.array(ay)
    r = {
        'Accuracy': accuracy_score(yt, preds),
        'F1': f1_score(yt, preds),
        'AUC-ROC': roc_auc_score(yt, probs),
        'AUC-PR': average_precision_score(yt, probs),
        'MCC': matthews_corrcoef(yt, preds)
    }
    print(f"\n  {name}:")
    for k, v in r.items():
        print(f"    {k}: {v:.4f}")
    ga = np.array(ag)
    if ga.ndim == 2:
        for i in range(ga.shape[1]):
            lbl = ['Demo', 'Obst'][i] if i < 2 else f'D{i}'
            print(f"    Gate {lbl}: {ga[:,i].mean():.3f}")
    return r

print(f"\n{'='*60}")
print(f"  RESULTS: {n_feat}-FEATURE HARMONIZED TH-DAT")
print(f"{'='*60}")

r1 = evaluate(model, X_te, y_te, f"PERI_DEP ({n_feat}-feat, internal test)")
r2 = evaluate(model, X_ug_s, y_ug, f"Uganda ({n_feat}-feat, DIRECT external, NO zero-padding)")

delta = abs(r1['AUC-ROC'] - r2['AUC-ROC'])
avg = (r1['AUC-ROC'] + r2['AUC-ROC']) / 2

print(f"\n{'='*60}")
print(f"  CROSS-DATASET COMPARISON")
print(f"{'='*60}")
print(f"  Full 19-feat (zero-padded): PERI_DEP=0.9440, Uganda=0.9026, Delta=0.041")
print(f"  Harmonized {n_feat}-feat:     PERI_DEP={r1['AUC-ROC']:.4f}, Uganda={r2['AUC-ROC']:.4f}, Delta={delta:.3f}")
print(f"  Average AUC (harmonized):  {avg:.4f}")

print(f"\n  >>> COPY THESE FOR THE PAPER <<<")
print(f"  Harmonized PERI_DEP AUC: {r1['AUC-ROC']:.4f}")
print(f"  Harmonized Uganda AUC:   {r2['AUC-ROC']:.4f}")
print(f"  Harmonized Delta:        {delta:.4f}")
print(f"  Harmonized F1 (PERI):    {r1['F1']:.4f}")
print(f"  Harmonized F1 (Uganda):  {r2['F1']:.4f}")
