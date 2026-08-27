import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             average_precision_score, matthews_corrcoef,
                             precision_score, recall_score, confusion_matrix)
import os
import pandas as pd

# Load Data
def load_data(data_dir='/content/'):
    X_train = np.load(os.path.join(data_dir, 'X_tr.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_tr.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_te.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_te.npy'))
    return X_train, y_train, X_val, y_val, X_test, y_test

class ANN(nn.Module):
    def __init__(self, input_dim):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.drop(self.relu(self.fc1(x)))
        x = self.drop(self.relu(self.fc2(x)))
        return self.sigmoid(self.fc3(x))

class AutoencoderBaseline(nn.Module):
    def __init__(self, input_dim):
        super(AutoencoderBaseline, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid()
        )

    def forward(self, x):
        emb = self.encoder(x)
        return self.classifier(emb)

def train_nn(model, X_train, y_train, X_val, y_val, epochs=80):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_v = torch.FloatTensor(X_val)
    
    best_auc, best_w = 0, None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_prob = model(X_v).numpy().squeeze()
            va = roc_auc_score(y_val, val_prob)
        if va > best_auc:
            best_auc = va
            best_w = {k: v.clone() for k, v in model.state_dict().items()}
    
    if best_w:
        model.load_state_dict(best_w)
    return best_auc

def full_metrics(y_true, y_prob, name="Model"):
    pred = (y_prob >= 0.5).astype(int)
    metrics = {
        'Accuracy': round(accuracy_score(y_true, pred), 4),
        'Precision': round(precision_score(y_true, pred, zero_division=0), 4),
        'Recall': round(recall_score(y_true, pred, zero_division=0), 4),
        'F1': round(f1_score(y_true, pred), 4),
        'AUC-ROC': round(roc_auc_score(y_true, y_prob), 4),
        'AUC-PR': round(average_precision_score(y_true, y_prob), 4),
        'MCC': round(matthews_corrcoef(y_true, pred), 4),
    }
    print(f"{name}: Acc={metrics['Accuracy']}, F1={metrics['F1']}, AUC={metrics['AUC-ROC']}, "
          f"PR={metrics['AUC-PR']}, MCC={metrics['MCC']}")
    return metrics

def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    print(f"Data loaded. Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    print(f"Test class dist: Dep={sum(y_test==1)}, Not={sum(y_test==0)}")
    
    results = {}
    
    # 1. Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict_proba(X_test)[:, 1]
    results['Random Forest'] = full_metrics(y_test, rf_preds, "Random Forest")
    
    # 2. XGBoost
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                         eval_metric='logloss', random_state=42)
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict_proba(X_test)[:, 1]
    results['XGBoost'] = full_metrics(y_test, xgb_preds, "XGBoost")
    
    # 3. ANN
    ann = ANN(input_dim=X_train.shape[1])
    best_v = train_nn(ann, X_train, y_train, X_val, y_val, epochs=80)
    print(f"ANN best val AUC: {best_v:.4f}")
    ann.eval()
    with torch.no_grad():
        ann_preds = ann(torch.FloatTensor(X_test)).numpy().squeeze()
    results['ANN'] = full_metrics(y_test, ann_preds, "ANN")
    
    # 4. Autoencoder
    ae = AutoencoderBaseline(input_dim=X_train.shape[1])
    best_v = train_nn(ae, X_train, y_train, X_val, y_val, epochs=80)
    print(f"Autoencoder best val AUC: {best_v:.4f}")
    ae.eval()
    with torch.no_grad():
        ae_preds = ae(torch.FloatTensor(X_test)).numpy().squeeze()
    results['Autoencoder'] = full_metrics(y_test, ae_preds, "Autoencoder")
    
    print("\n" + "=" * 70)
    print("  BASELINE RESULTS (ALL METRICS)")
    print("=" * 70)
    print(pd.DataFrame(results).T.to_string())

if __name__ == '__main__':
    main()
