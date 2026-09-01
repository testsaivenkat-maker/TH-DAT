"""
11_real_predictions_and_curves.py
==================================
Trains RF, XGBoost, TabTransformer, TH-DAT on PERI_DEP.
Generates ROC and PR curves from ACTUAL model predictions.
NO synthetic/approximate curves.

Run in Colab (T4 GPU, ~20 min):
  !pip install imbalanced-learn xgboost
  !python 11_real_predictions_and_curves.py
"""
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, f1_score, accuracy_score,
                             matthews_corrcoef, brier_score_loss, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import warnings; warnings.filterwarnings('ignore')
import json, os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ═══════════════════════════════════════════
# 1. PREPROCESSING (exact same as 01_preprocessing)
# ═══════════════════════════════════════════
print("Loading data...")
df = pd.read_csv('/content/dataset.csv')
df['Label'] = (df['Labelling'] == 'Depressed').astype(int)
df['Trimester'] = np.where(df['Gestational Age'] <= 13, 0,
                  np.where(df['Gestational Age'] <= 26, 1, 2))

le = LabelEncoder()
cat_cols = [c for c in df.select_dtypes(include='object').columns if c != 'Labelling']
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

phq9 = ['Little interest or pleasure in doing things',
        'Feeling down, depressed, or hopeless',
        'Trouble falling or staying sleep or sleeping too much',
        'Feeling tired or having little energy',
        'Poor appetite or overeating',
        'Feeling badabout yourself that you are failure or have let yourself or your family down',
        'Trouble concentrating on things, such as reading the newspaper or watching television',
        'Moving or speaking so slowly that other people could have Noticed.',
        'Thoughts that you would be better off dead, or of hurting yourself']

# Match exact column names (some have trailing spaces)
phq9_matched = []
for p in phq9:
    for c in df.columns:
        if c.strip() == p.strip():
            phq9_matched.append(c)
            break

feat_cols = [c for c in df.columns if c not in ['Labelling', 'Scalling', 'Label'] + phq9_matched]
X = SimpleImputer(strategy='median').fit_transform(df[feat_cols].values)
y = df['Label'].values
N = X.shape[1]
print(f"Features: {N}")
for i, c in enumerate(feat_cols):
    print(f"  {i}: {c}")

# Split first, then SMOTE
X_tr, X_temp, y_tr, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_te, y_val, y_te = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
print(f"\nBefore SMOTE: Train={len(y_tr)}, Val={len(y_val)}, Test={len(y_te)}")
X_tr, y_tr = SMOTE(random_state=42).fit_resample(X_tr, y_tr)
print(f"After SMOTE:  Train={len(y_tr)}")

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_val = scaler.transform(X_val)
X_te = scaler.transform(X_te)

# ═══════════════════════════════════════════
# 2. RANDOM FOREST
# ═══════════════════════════════════════════
print("\n" + "="*50)
print("  Training Random Forest...")
rf = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=5,
                            min_samples_leaf=2, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
rf_probs = rf.predict_proba(X_te)[:, 1]
print(f"  RF AUC: {roc_auc_score(y_te, rf_probs):.4f}")

# ═══════════════════════════════════════════
# 3. XGBOOST
# ═══════════════════════════════════════════
print("  Training XGBoost...")
xgb_clf = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                              eval_metric='logloss', random_state=42, use_label_encoder=False)
xgb_clf.fit(X_tr, y_tr)
xgb_probs = xgb_clf.predict_proba(X_te)[:, 1]
print(f"  XGB AUC: {roc_auc_score(y_te, xgb_probs):.4f}")

# ═══════════════════════════════════════════
# 4. TABTRANSFORMER
# ═══════════════════════════════════════════
print("  Training TabTransformer...")

class TabTransformer(nn.Module):
    def __init__(self, n_feat, dim=64, heads=4, layers=3, drop=0.1):
        super().__init__()
        self.embed = nn.Linear(1, dim)
        self.pos = nn.Embedding(n_feat, dim)
        enc = nn.TransformerEncoderLayer(dim, heads, dim*4, drop, batch_first=True)
        self.tf = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim),
                                  nn.GELU(), nn.Dropout(drop), nn.Linear(dim, 1))

    def forward(self, x):
        B, F = x.shape
        t = self.embed(x.unsqueeze(-1)) + self.pos(torch.arange(F, device=x.device).unsqueeze(0).expand(B, -1))
        t = self.tf(t)
        return self.head(t.mean(dim=1)).squeeze(-1)

def make_dl(X, y, bs=128, shuf=True):
    return DataLoader(TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y)), bs, shuf)

tabtf = TabTransformer(N).to(device)
opt_t = torch.optim.AdamW(tabtf.parameters(), lr=1e-3, weight_decay=1e-4)
crit = nn.BCEWithLogitsLoss()
best_tabtf_auc, best_tabtf_st = 0, None

for ep in range(1, 151):
    tabtf.train()
    for xb, yb in make_dl(X_tr, y_tr):
        xb, yb = xb.to(device), yb.to(device)
        loss = crit(tabtf(xb), yb)
        opt_t.zero_grad(); loss.backward(); opt_t.step()
    if ep % 10 == 0:
        tabtf.eval()
        ap, ay = [], []
        with torch.no_grad():
            for xb, yb in make_dl(X_val, y_val, shuf=False):
                ap.extend(torch.sigmoid(tabtf(xb.to(device))).cpu().numpy())
                ay.extend(yb.numpy())
        a = roc_auc_score(ay, ap)
        if a > best_tabtf_auc:
            best_tabtf_auc = a
            best_tabtf_st = {k: v.cpu().clone() for k, v in tabtf.state_dict().items()}
        if ep % 50 == 0:
            print(f"    TabTF Epoch {ep} | Val AUC: {a:.4f} | Best: {best_tabtf_auc:.4f}")

tabtf.load_state_dict(best_tabtf_st); tabtf.eval()
tabtf_probs_list = []
with torch.no_grad():
    for xb, yb in make_dl(X_te, y_te, shuf=False):
        tabtf_probs_list.extend(torch.sigmoid(tabtf(xb.to(device))).cpu().numpy())
tabtf_probs = np.array(tabtf_probs_list)
print(f"  TabTF AUC: {roc_auc_score(y_te, tabtf_probs):.4f}")

# ═══════════════════════════════════════════
# 5. TH-DAT (exact architecture from paper)
# ═══════════════════════════════════════════
print("  Training TH-DAT...")

DEMO_IDX = [0, 6, 7, 8, 11, 13]
OBST_IDX = [1, 2, 3, 4, 5, 10, 14]
PSYC_IDX = [9, 12, 15]
assigned = set(DEMO_IDX + OBST_IDX + PSYC_IDX)
for i in range(N):
    if i not in assigned:
        OBST_IDX.append(i)

class THDAT(nn.Module):
    def __init__(self, nf, dim=128, heads=4, layers=6, drop=0.15):
        super().__init__()
        self.nf, self.dim = nf, dim
        self.feat_embed = nn.Linear(1, dim)
        self.pos_enc = nn.Embedding(nf + 3, dim)
        enc = nn.TransformerEncoderLayer(dim, heads, dim*4, drop, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, layers)
        self.demo_pool = nn.MultiheadAttention(dim, heads, drop, batch_first=True)
        self.obst_pool = nn.MultiheadAttention(dim, heads, drop, batch_first=True)
        self.psyc_pool = nn.MultiheadAttention(dim, heads, drop, batch_first=True)
        self.tri_query = nn.Embedding(3, dim)
        self.tri_cross = nn.MultiheadAttention(dim, heads, drop, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(dim * 3, 3), nn.Softmax(dim=-1))
        self.classifier = nn.Sequential(
            nn.Linear(dim + nf, dim), nn.GELU(), nn.Dropout(drop), nn.Linear(dim, 1))

    def forward(self, x, tri=None):
        B, F = x.shape
        raw = x.clone()
        tok = self.feat_embed(x.unsqueeze(-1))
        tok = tok + self.pos_enc(torch.arange(F, device=x.device).unsqueeze(0).expand(B, -1))
        h = self.transformer(tok)
        def pool(attn, idx):
            s = h[:, idx, :]; q = s.mean(1, keepdim=True); o, _ = attn(q, s, s); return o
        d, o, p = pool(self.demo_pool, DEMO_IDX), pool(self.obst_pool, OBST_IDX), pool(self.psyc_pool, PSYC_IDX)
        gate_in = torch.cat([d.squeeze(1), o.squeeze(1), p.squeeze(1)], dim=-1)
        g = self.gate(gate_in)
        fused = g[:,0:1]*d.squeeze(1) + g[:,1:2]*o.squeeze(1) + g[:,2:3]*p.squeeze(1)
        if tri is not None:
            dc = torch.cat([d, o, p], dim=1)
            tq = self.tri_query(tri.long()).unsqueeze(1)
            tri_ctx, _ = self.tri_cross(tq, dc, dc)
            fused = fused + tri_ctx.squeeze(1)
        return self.classifier(torch.cat([fused, raw], dim=-1)).squeeze(-1), g

# Trimester IDs
gest_col = 1  # Gestational Age is column index 1
def get_tri(X):
    tri = np.zeros(len(X), dtype=int)
    for i in range(len(X)):
        v = X[i, gest_col]
        if v <= -0.5: tri[i] = 0
        elif v <= 0.5: tri[i] = 1
        else: tri[i] = 2
    return tri

def make_dl_tri(X, y, bs=128, shuf=True):
    tri = get_tri(X)
    return DataLoader(TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y),
                                    torch.LongTensor(tri)), bs, shuf)

model = THDAT(N).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2)

best_auc, best_st = 0, None
for ep in range(1, 201):
    model.train()
    for xb, yb, tb in make_dl_tri(X_tr, y_tr):
        xb, yb, tb = xb.to(device), yb.to(device), tb.to(device)
        logits, _ = model(xb, tb)
        loss = crit(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()
    if ep % 10 == 0:
        model.eval()
        ap, ay = [], []
        with torch.no_grad():
            for xb, yb, tb in make_dl_tri(X_val, y_val, shuf=False):
                xb, tb = xb.to(device), tb.to(device)
                logits, _ = model(xb, tb)
                ap.extend(torch.sigmoid(logits).cpu().numpy()); ay.extend(yb.numpy())
        a = roc_auc_score(ay, ap)
        if a > best_auc:
            best_auc = a
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 50 == 0:
            print(f"    TH-DAT Epoch {ep} | Val AUC: {a:.4f} | Best: {best_auc:.4f}")

model.load_state_dict(best_st); model.eval()

# Get TH-DAT predictions
thdat_probs_list, thdat_gates_list = [], []
with torch.no_grad():
    for xb, yb, tb in make_dl_tri(X_te, y_te, shuf=False):
        xb, tb = xb.to(device), tb.to(device)
        logits, gates = model(xb, tb)
        thdat_probs_list.extend(torch.sigmoid(logits).cpu().numpy())
        thdat_gates_list.extend(gates.cpu().numpy())
thdat_probs = np.array(thdat_probs_list)
thdat_gates = np.array(thdat_gates_list)
print(f"  TH-DAT AUC: {roc_auc_score(y_te, thdat_probs):.4f}")
print(f"  Gates: Demo={thdat_gates[:,0].mean():.4f}, Obst={thdat_gates[:,1].mean():.4f}, Psyc={thdat_gates[:,2].mean():.4f}")

# ═══════════════════════════════════════════
# 6. COMPREHENSIVE METRICS (all from REAL predictions)
# ═══════════════════════════════════════════
print("\n" + "="*60)
print("  ALL METRICS FROM REAL PREDICTIONS")
print("="*60)

THRESHOLD = 0.50  # Fixed threshold

all_models = [
    ('TH-DAT', thdat_probs),
    ('Random Forest', rf_probs),
    ('XGBoost', xgb_probs),
    ('TabTransformer', tabtf_probs),
]

results = {}
for name, probs in all_models:
    preds = (probs >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, preds).ravel()
    r = {
        'AUC-ROC': roc_auc_score(y_te, probs),
        'AUC-PR': average_precision_score(y_te, probs),
        'Accuracy': accuracy_score(y_te, preds),
        'F1': f1_score(y_te, preds),
        'MCC': matthews_corrcoef(y_te, preds),
        'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'Brier': brier_score_loss(y_te, probs),
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
    }
    results[name] = r
    print(f"\n  {name} (threshold={THRESHOLD}):")
    for k, v in r.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

# Save results
with open('/content/real_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\n  Saved: /content/real_metrics.json")

# ═══════════════════════════════════════════
# 7. PLOT: ROC + PR CURVES (FROM REAL PREDICTIONS)
# ═══════════════════════════════════════════
print("\nGenerating ROC and PR curves from REAL predictions...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

colors = {'TH-DAT': '#FF4444', 'Random Forest': '#4488CC',
          'XGBoost': '#44AA88', 'TabTransformer': '#FF8844'}
styles = {'TH-DAT': '-', 'Random Forest': '--',
          'XGBoost': '-.', 'TabTransformer': ':'}
widths = {'TH-DAT': 2.5, 'Random Forest': 2.0,
          'XGBoost': 1.8, 'TabTransformer': 1.8}

for name, probs in all_models:
    auc_val = roc_auc_score(y_te, probs)
    fpr, tpr, _ = roc_curve(y_te, probs)
    ax1.plot(fpr, tpr, color=colors[name], linestyle=styles[name],
             linewidth=widths[name], label=f'{name} (AUC={auc_val:.4f})')

    pr_val = average_precision_score(y_te, probs)
    prec, rec, _ = precision_recall_curve(y_te, probs)
    ax2.plot(rec, prec, color=colors[name], linestyle=styles[name],
             linewidth=widths[name], label=f'{name} (AP={pr_val:.4f})')

ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
ax1.set_xlabel('False Positive Rate', fontsize=13)
ax1.set_ylabel('True Positive Rate', fontsize=13)
ax1.set_title('ROC Curves (from actual test-set predictions)', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right', fontsize=10)
ax1.set_xlim([-0.02, 1.02]); ax1.set_ylim([-0.02, 1.02])
ax1.grid(True, alpha=0.3)

prev = y_te.mean()
ax2.axhline(y=prev, color='gray', linestyle='--', alpha=0.3, label=f'Prevalence ({prev:.3f})')
ax2.set_xlabel('Recall', fontsize=13)
ax2.set_ylabel('Precision', fontsize=13)
ax2.set_title('Precision-Recall Curves (from actual test-set predictions)', fontsize=14, fontweight='bold')
ax2.legend(loc='lower left', fontsize=10)
ax2.set_xlim([-0.02, 1.02]); ax2.set_ylim([-0.02, 1.02])
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/content/plot7_roc_pr_real.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: /content/plot7_roc_pr_real.png")

# ═══════════════════════════════════════════
# 8. CALIBRATION PLOT (FROM REAL PREDICTIONS)
# ═══════════════════════════════════════════
print("Generating calibration plot from REAL predictions...")

fig, ax = plt.subplots(figsize=(8, 8))
for name, probs in all_models:
    prob_true, prob_pred = calibration_curve(y_te, probs, n_bins=10, strategy='uniform')
    ax.plot(prob_pred, prob_true, color=colors[name], linestyle=styles[name],
            linewidth=widths[name], marker='o', markersize=5, label=name)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Perfect')
ax.set_xlabel('Mean Predicted Probability', fontsize=13)
ax.set_ylabel('Fraction of Positives', fontsize=13)
ax.set_title('Calibration Plot (from actual test-set predictions)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
ax.grid(True, alpha=0.3); ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('/content/plot9_calibration_real.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: /content/plot9_calibration_real.png")

# ═══════════════════════════════════════════
# 9. CONFUSION MATRIX PLOT (FROM REAL PREDICTIONS)
# ═══════════════════════════════════════════
print("Generating confusion matrices from REAL predictions...")

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for idx, (name, probs) in enumerate(all_models):
    preds = (probs >= THRESHOLD).astype(int)
    cm = confusion_matrix(y_te, preds)
    ax = axes[idx]
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Not Dep', 'Dep']); ax.set_yticklabels(['Not Dep', 'Dep'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=14, fontweight='bold',
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.suptitle(f'Confusion Matrices (threshold=0.50, N={len(y_te):,})',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/content/plot8_confusion_real.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: /content/plot8_confusion_real.png")

# ═══════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════
print("\n" + "="*60)
print("  ALL PLOTS GENERATED FROM REAL PREDICTIONS")
print("="*60)
print("  1. /content/plot7_roc_pr_real.png     - ROC + PR curves")
print("  2. /content/plot9_calibration_real.png - Calibration plot")
print("  3. /content/plot8_confusion_real.png   - Confusion matrices")
print("  4. /content/real_metrics.json          - All metrics")
print("\n  ALL curves are from actual model predictions.")
print("  NO synthetic/approximate distributions used.")
print("\n  Download all 3 PNGs and send them back!")
