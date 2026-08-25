import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import os

# Load Data
def load_data(data_dir='/content/'):
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    return X_train, y_train, X_val, y_val, X_test, y_test

class ANN(nn.Module):
    def __init__(self, input_dim):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))

class AutoencoderBaseline(nn.Module):
    def __init__(self, input_dim):
        super(AutoencoderBaseline, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        emb = self.encoder(x)
        return self.classifier(emb)

def train_nn(model, X_train, y_train, X_val, y_val, epochs=50):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train).unsqueeze(1)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()

def evaluate(y_true, y_pred_prob, name="Model"):
    auc = roc_auc_score(y_true, y_pred_prob)
    acc = accuracy_score(y_true, (y_pred_prob > 0.5).astype(int))
    print(f"{name} - AUC: {auc:.4f}, ACC: {acc:.4f}")
    return auc, acc

def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    print(f"Data loaded. Train size: {X_train.shape[0]}")
    
    # 1. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict_proba(X_test)[:, 1]
    evaluate(y_test, rf_preds, "Random Forest")
    
    # 2. XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict_proba(X_test)[:, 1]
    evaluate(y_test, xgb_preds, "XGBoost")
    
    # 3. ANN
    ann = ANN(input_dim=X_train.shape[1])
    train_nn(ann, X_train, y_train, X_val, y_val)
    ann.eval()
    with torch.no_grad():
        ann_preds = ann(torch.FloatTensor(X_test)).numpy().squeeze()
    evaluate(y_test, ann_preds, "ANN")
    
    # 4. Autoencoder
    ae = AutoencoderBaseline(input_dim=X_train.shape[1])
    train_nn(ae, X_train, y_train, X_val, y_val)
    ae.eval()
    with torch.no_grad():
        ae_preds = ae(torch.FloatTensor(X_test)).numpy().squeeze()
    evaluate(y_test, ae_preds, "Autoencoder")

if __name__ == '__main__':
    main()
