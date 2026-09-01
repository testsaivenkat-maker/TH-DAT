"""
TH-DAT Preprocessing — PERI_DEP Dataset
FIXED: Split FIRST, then SMOTE on training set ONLY
Validation and test sets contain ONLY original (non-synthetic) samples.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

df = pd.read_csv('/content/drive/MyDrive/dataset.csv')
print("Loaded:", df.shape)

df['Label'] = (df['Labelling'] == 'Depressed').astype(int)
df['Trimester'] = np.where(df['Gestational Age'] <= 13, 0, np.where(df['Gestational Age'] <= 26, 1, 2))

cat_cols = [c for c in df.select_dtypes(include='object').columns if c != 'Labelling']
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

phq9_cols = ['Little interest or pleasure in doing things', 'Feeling down, depressed, or hopeless', 'Trouble falling or staying sleep or sleeping too much', 'Feeling tired or having little energy', 'Poor appetite or overeating', 'Feeling badabout yourself that you are failure or have let yourself or your family down', 'Trouble concentrating on things, such as reading the newspaper or watching television', 'Moving or speaking so slowly that other people could have Noticed.', 'Thoughts that you would be better off dead, or of hurting yourself']

feature_cols_clean = [c for c in df.columns if c not in ['Labelling', 'Scalling', 'Label'] + phq9_cols]
print("Features:", len(feature_cols_clean))
for i, c in enumerate(feature_cols_clean):
    print(f"  {i}: {c}")

X = SimpleImputer(strategy='median').fit_transform(df[feature_cols_clean].values)
y = df['Label'].values

print(f"\nOriginal data: {X.shape[0]} samples")
print(f"  Depressed: {sum(y==1)} ({sum(y==1)/len(y)*100:.1f}%)")
print(f"  Not depressed: {sum(y==0)} ({sum(y==0)/len(y)*100:.1f}%)")

# ============================================================
# CORRECT PIPELINE: Split FIRST, then SMOTE on training ONLY
# ============================================================
# Step 1: Stratified split on ORIGINAL data (no SMOTE yet)
X_tr, X_temp, y_tr, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_te, y_val, y_te = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

print(f"\nBefore SMOTE:")
print(f"  Train: {X_tr.shape[0]} (Dep={sum(y_tr==1)}, Not={sum(y_tr==0)})")
print(f"  Val:   {X_val.shape[0]} (Dep={sum(y_val==1)}, Not={sum(y_val==0)})")
print(f"  Test:  {X_te.shape[0]} (Dep={sum(y_te==1)}, Not={sum(y_te==0)})")

# Step 2: SMOTE ONLY on training set
X_tr, y_tr = SMOTE(random_state=42).fit_resample(X_tr, y_tr)

print(f"\nAfter SMOTE (training only):")
print(f"  Train: {X_tr.shape[0]} (Dep={sum(y_tr==1)}, Not={sum(y_tr==0)})")
print(f"  Val:   {X_val.shape[0]} (ORIGINAL, untouched)")
print(f"  Test:  {X_te.shape[0]} (ORIGINAL, untouched)")

# Step 3: Scale (fit on training, transform val/test)
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_val = scaler.transform(X_val)
X_te = scaler.transform(X_te)

# Step 4: Save
np.save('/content/X_tr.npy', X_tr)
np.save('/content/X_val.npy', X_val)
np.save('/content/X_te.npy', X_te)
np.save('/content/y_tr.npy', y_tr)
np.save('/content/y_val.npy', y_val)
np.save('/content/y_te.npy', y_te)

print(f"\nFinal shapes:")
print(f"  X_tr: {X_tr.shape}")
print(f"  X_val: {X_val.shape}")
print(f"  X_te: {X_te.shape}")
print("\n.npy files saved to /content/")
print("\n*** SMOTE applied to TRAINING SET ONLY ***")
print("*** Validation and test sets are 100% original data ***")
