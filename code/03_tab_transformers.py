import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
import numpy as np
import pandas as pd
import pickle

# Load saved data
X_tr = np.load('X_tr.npy')
X_val = np.load('X_val.npy')
X_te = np.load('X_te.npy')
y_tr = np.load('y_tr.npy')
y_val = np.load('y_val.npy')
y_te = np.load('y_te.npy')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)
print("X_tr:", X_tr.shape)

Xtr_t = torch.tensor(X_tr, dtype=torch.float32)
Xva_t = torch.tensor(X_val, dtype=torch.float32).to(device)
Xte_t = torch.tensor(X_te, dtype=torch.float32).to(device)
ytr_t = torch.tensor(y_tr, dtype=torch.float32)
dl = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=256, shuffle=True)
N = X_tr.shape[1]
bce = nn.BCELoss()


def get_scores(prob):
    pred = (prob >= 0.5).astype(int)
    return {
        'Accuracy': round(accuracy_score(y_te, pred), 4),
        'F1': round(f1_score(y_te, pred), 4),
        'AUC-ROC': round(roc_auc_score(y_te, prob), 4),
        'AUC-PR': round(average_precision_score(y_te, prob), 4)
    }


class TabTF(nn.Module):
    def __init__(self, num_feat):
        super(TabTF, self).__init__()
        self.emb = nn.Linear(1, 32)
        self.tf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=32, nhead=4, dim_feedforward=128,
                dropout=0.1, batch_first=True
            ),
            num_layers=3
        )
        self.clf = nn.Sequential(
            nn.Linear(num_feat * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B = x.shape[0]
        e = self.emb(x.unsqueeze(-1))
        e = self.tf(e)
        return self.clf(e.reshape(B, -1)).squeeze(-1)


class FTTF(nn.Module):
    def __init__(self, num_feat):
        super(FTTF, self).__init__()
        self.emb = nn.Linear(1, 64)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 64))
        self.pos_embed = nn.Parameter(torch.randn(1, num_feat + 1, 64))
        self.tf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=64, nhead=8, dim_feedforward=256,
                dropout=0.1, batch_first=True
            ),
            num_layers=4
        )
        self.clf = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B = x.shape[0]
        e = self.emb(x.unsqueeze(-1))
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, e], dim=1) + self.pos_embed
        out = self.tf(seq)
        return self.clf(out[:, 0]).squeeze(-1)


# ═══ Train TabTransformer ═══
print("\n>>> Training TabTransformer...")
m1 = TabTF(N).to(device)
o1 = torch.optim.Adam(m1.parameters(), lr=1e-3, weight_decay=1e-4)
best1, w1 = 0, None

for ep in range(50):
    m1.train()
    for xb, yb in dl:
        o1.zero_grad()
        bce(m1(xb.to(device)), yb.to(device)).backward()
        o1.step()
    m1.eval()
    with torch.no_grad():
        a = roc_auc_score(y_val, m1(Xva_t).cpu().numpy())
    if a > best1:
        best1 = a
        w1 = {k: v.clone() for k, v in m1.state_dict().items()}
    if (ep + 1) % 10 == 0:
        print(f"  Epoch {ep+1}/50 | Val AUC: {a:.4f}")

m1.load_state_dict(w1)
m1.eval()
with torch.no_grad():
    p1 = m1(Xte_t).cpu().numpy()
tab_scores = get_scores(p1)
print("TabTransformer:", tab_scores)


# ═══ Train FT-Transformer ═══
print("\n>>> Training FT-Transformer...")
m2 = FTTF(N).to(device)
o2 = torch.optim.Adam(m2.parameters(), lr=5e-4, weight_decay=1e-4)
best2, w2 = 0, None

for ep in range(50):
    m2.train()
    for xb, yb in dl:
        o2.zero_grad()
        bce(m2(xb.to(device)), yb.to(device)).backward()
        o2.step()
    m2.eval()
    with torch.no_grad():
        a2 = roc_auc_score(y_val, m2(Xva_t).cpu().numpy())
    if a2 > best2:
        best2 = a2
        w2 = {k: v.clone() for k, v in m2.state_dict().items()}
    if (ep + 1) % 10 == 0:
        print(f"  Epoch {ep+1}/50 | Val AUC: {a2:.4f}")

m2.load_state_dict(w2)
m2.eval()
with torch.no_grad():
    p2 = m2(Xte_t).cpu().numpy()
ft_scores = get_scores(p2)
print("FT-Transformer:", ft_scores)


# ═══ Print All Results ═══
results = {
    'Random Forest':  {'Accuracy': 0.9247, 'F1': 0.9246, 'AUC-ROC': 0.9670, 'AUC-PR': 0.9628},
    'XGBoost':        {'Accuracy': 0.8139, 'F1': 0.8170, 'AUC-ROC': 0.9072, 'AUC-PR': 0.9031},
    'ANN':            {'Accuracy': 0.7735, 'F1': 0.7579, 'AUC-ROC': 0.8514, 'AUC-PR': 0.8601},
    'Autoencoder':    {'Accuracy': 0.8194, 'F1': 0.8187, 'AUC-ROC': 0.8898, 'AUC-PR': 0.8954},
    'TabTransformer': tab_scores,
    'FT-Transformer': ft_scores,
}

print("\n========== RESULTS SO FAR ==========")
print(pd.DataFrame(results).T.to_string())
