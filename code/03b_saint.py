import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
import numpy as np
import pandas as pd

X_tr = np.load('/content/X_tr.npy')
X_val = np.load('/content/X_val.npy')
X_te = np.load('/content/X_te.npy')
y_tr = np.load('/content/y_tr.npy')
y_val = np.load('/content/y_val.npy')
y_te = np.load('/content/y_te.npy')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

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


class SAINTBlock(nn.Module):
    def __init__(self, dim, heads=4):
        super(SAINTBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff1 = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)
        self.inter_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.ff2 = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(dim * 2, dim))
        self.norm4 = nn.LayerNorm(dim)

    def forward(self, x):
        sa_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + sa_out)
        x = self.norm2(x + self.ff1(x))
        B, NF, D = x.shape
        x_perm = x.permute(1, 0, 2)
        inter_results = []
        for i in range(NF):
            tokens = x_perm[i].unsqueeze(1)
            ia_out, _ = self.inter_attn(tokens, tokens, tokens)
            inter_results.append(ia_out)
        ia_combined = torch.cat(inter_results, dim=1)
        x = self.norm3(x + ia_combined)
        x = self.norm4(x + self.ff2(x))
        return x


class SAINT(nn.Module):
    def __init__(self, num_feat, dim=64, heads=4, num_blocks=3):
        super(SAINT, self).__init__()
        self.feat_embed = nn.Linear(1, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_feat + 1, dim))
        self.blocks = nn.ModuleList([SAINTBlock(dim, heads) for _ in range(num_blocks)])
        self.classifier = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        B = x.shape[0]
        e = self.feat_embed(x.unsqueeze(-1))
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, e], dim=1) + self.pos_embed
        for block in self.blocks:
            seq = block(seq)
        return self.classifier(seq[:, 0]).squeeze(-1)


print("\n>>> Training SAINT...")
model = SAINT(N, dim=64, heads=4, num_blocks=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 60)
best_auc, best_weights = 0, None

for ep in range(60):
    model.train()
    for xb, yb in dl:
        optimizer.zero_grad()
        bce(model(xb.to(device)), yb.to(device)).backward()
        optimizer.step()
    scheduler.step()
    model.eval()
    with torch.no_grad():
        val_auc = roc_auc_score(y_val, model(Xva_t).cpu().numpy())
    if val_auc > best_auc:
        best_auc = val_auc
        best_weights = {k: v.clone() for k, v in model.state_dict().items()}
    if (ep + 1) % 10 == 0:
        print(f"  Epoch {ep+1}/60 | Val AUC: {val_auc:.4f}")

model.load_state_dict(best_weights)
model.eval()
with torch.no_grad():
    prob = model(Xte_t).cpu().numpy()
saint_scores = get_scores(prob)
print("SAINT:", saint_scores)

results = {
    'Random Forest':  {'Accuracy': 0.9247, 'F1': 0.9246, 'AUC-ROC': 0.9670, 'AUC-PR': 0.9628},
    'XGBoost':        {'Accuracy': 0.8139, 'F1': 0.8170, 'AUC-ROC': 0.9072, 'AUC-PR': 0.9031},
    'ANN':            {'Accuracy': 0.7735, 'F1': 0.7579, 'AUC-ROC': 0.8514, 'AUC-PR': 0.8601},
    'Autoencoder':    {'Accuracy': 0.8194, 'F1': 0.8187, 'AUC-ROC': 0.8898, 'AUC-PR': 0.8954},
    'TabTransformer': {'Accuracy': 0.8946, 'F1': 0.8942, 'AUC-ROC': 0.9527, 'AUC-PR': 0.9513},
    'FT-Transformer': {'Accuracy': 0.7669, 'F1': 0.7768, 'AUC-ROC': 0.8568, 'AUC-PR': 0.8493},
    'SAINT':          saint_scores,
}
print("\n========== RESULTS SO FAR ==========")
print(pd.DataFrame(results).T.to_string())
