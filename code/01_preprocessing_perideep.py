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

X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
X_tr, X_temp, y_tr, y_temp = train_test_split(X_res, y_res, test_size=0.30, stratify=y_res, random_state=42)
X_val, X_te, y_val, y_te = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_val = scaler.transform(X_val)
X_te = scaler.transform(X_te)

np.save('/content/X_tr.npy', X_tr)
np.save('/content/X_val.npy', X_val)
np.save('/content/X_te.npy', X_te)
np.save('/content/y_tr.npy', y_tr)
np.save('/content/y_val.npy', y_val)
np.save('/content/y_te.npy', y_te)

print("\nX_tr:", X_tr.shape)
print("X_val:", X_val.shape)
print("X_te:", X_te.shape)
print("\n.npy files saved to /content/")
