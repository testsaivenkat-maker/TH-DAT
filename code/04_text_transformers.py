import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# ═══════════════════════════════════════════
# Load original dataset and prepare text
# ═══════════════════════════════════════════
df = pd.read_csv('/content/drive/MyDrive/dataset.csv')
df['Label'] = (df['Labelling'] == 'Depressed').astype(int)
df['Trimester'] = np.where(df['Gestational Age'] <= 13, 'First',
                  np.where(df['Gestational Age'] <= 26, 'Second', 'Third'))

phq9_cols = [
    'Little interest or pleasure in doing things',
    'Feeling down, depressed, or hopeless',
    'Trouble falling or staying sleep or sleeping too much',
    'Feeling tired or having little energy',
    'Poor appetite or overeating',
    'Feeling badabout yourself that you are failure or have let yourself or your family down',
    'Trouble concentrating on things, such as reading the newspaper or watching television',
    'Moving or speaking so slowly that other people could have Noticed.',
    'Thoughts that you would be better off dead, or of hurting yourself'
]

feature_cols = [c for c in df.columns if c not in ['Labelling', 'Scalling', 'Label'] + phq9_cols]

# Convert each row to text
def row_to_text(row):
    parts = []
    for col in feature_cols:
        parts.append(f"{col} is {row[col]}")
    return ". ".join(parts) + "."

texts = [row_to_text(row) for _, row in df.iterrows()]
labels = df['Label'].values

print(f"Total samples: {len(texts)}")
print(f"Sample text: {texts[0][:200]}...")

# Split 70/15/15 (same random_state as numeric pipeline)
t_tr, t_temp, l_tr, l_temp = train_test_split(
    texts, labels, test_size=0.30, stratify=labels, random_state=42)
t_val, t_te, l_val, l_te = train_test_split(
    t_temp, l_temp, test_size=0.50, stratify=l_temp, random_state=42)

print(f"Train: {len(t_tr)} | Val: {len(t_val)} | Test: {len(t_te)}")

# Class weights for imbalance
n_pos = sum(l_tr)
n_neg = len(l_tr) - n_pos
weight = torch.tensor([n_pos / len(l_tr), n_neg / len(l_tr)], dtype=torch.float32).to(device)
print(f"Class weights: {weight}")


# ═══════════════════════════════════════════
# Dataset class
# ═══════════════════════════════════════════
class TextDS(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.enc = tokenizer(texts, truncation=True, padding='max_length',
                             max_length=max_len, return_tensors='pt')
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        item['labels'] = self.labels[idx]
        return item


def get_scores(probs, y_true):
    preds = (np.array(probs) >= 0.5).astype(int)
    return {
        'Accuracy': round(accuracy_score(y_true, preds), 4),
        'F1': round(f1_score(y_true, preds), 4),
        'AUC-ROC': round(roc_auc_score(y_true, probs), 4),
        'AUC-PR': round(average_precision_score(y_true, probs), 4)
    }


# ═══════════════════════════════════════════
# Shared training function
# ═══════════════════════════════════════════
def train_text_model(model_name, display_name, epochs=4, batch_size=32, lr=2e-5):
    print(f"\n{'='*50}")
    print(f"Training: {display_name}")
    print(f"{'='*50}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True
    ).to(device)

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    train_ds = TextDS(t_tr, l_tr, tokenizer)
    val_ds = TextDS(t_val, l_val, tokenizer)
    test_ds = TextDS(t_te, l_te, tokenizer)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(weight=weight)

    best_val_auc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            labs = batch.pop('labels')
            opt.zero_grad()
            outputs = model(**batch)
            loss = loss_fn(outputs.logits, labs)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        model.eval()
        val_probs = []
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                batch.pop('labels')
                logits = model(**batch).logits
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                val_probs.extend(probs)
        val_auc = roc_auc_score(l_val, val_probs)
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_dl):.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    test_probs = []
    with torch.no_grad():
        for batch in test_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch.pop('labels')
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            test_probs.extend(probs)

    scores = get_scores(test_probs, l_te)
    print(f"  {display_name}: {scores}")

    del model, tokenizer, train_ds, val_ds, test_ds
    torch.cuda.empty_cache()

    return scores


# ═══════════════════════════════════════════
# Train all 4 text transformers
# ═══════════════════════════════════════════
bert_scores = train_text_model('bert-base-uncased', 'BERT')
roberta_scores = train_text_model('roberta-base', 'RoBERTa')
gpt2_scores = train_text_model('gpt2', 'GPT-2', lr=1e-5)
t5_scores = train_text_model('google-t5/t5-small', 'T5', lr=1e-4)

# ═══════════════════════════════════════════
# Full results table
# ═══════════════════════════════════════════
results = {
    'Random Forest':  {'Accuracy': 0.9247, 'F1': 0.9246, 'AUC-ROC': 0.9670, 'AUC-PR': 0.9628},
    'XGBoost':        {'Accuracy': 0.8139, 'F1': 0.8170, 'AUC-ROC': 0.9072, 'AUC-PR': 0.9031},
    'ANN':            {'Accuracy': 0.7735, 'F1': 0.7579, 'AUC-ROC': 0.8514, 'AUC-PR': 0.8601},
    'Autoencoder':    {'Accuracy': 0.8194, 'F1': 0.8187, 'AUC-ROC': 0.8898, 'AUC-PR': 0.8954},
    'TabTransformer': {'Accuracy': 0.8946, 'F1': 0.8942, 'AUC-ROC': 0.9527, 'AUC-PR': 0.9513},
    'FT-Transformer': {'Accuracy': 0.7669, 'F1': 0.7768, 'AUC-ROC': 0.8568, 'AUC-PR': 0.8493},
    'SAINT':          {'Accuracy': 0.7614, 'F1': 0.7598, 'AUC-ROC': 0.8543, 'AUC-PR': 0.8497},
    'BERT':           bert_scores,
    'RoBERTa':        roberta_scores,
    'GPT-2':          gpt2_scores,
    'T5':             t5_scores,
}
print("\n========== ALL 11 BASELINES COMPLETE ==========")
print(pd.DataFrame(results).T.to_string())
print("\n>>> NEXT: TH-DAT (your proposed model!) <<<")
