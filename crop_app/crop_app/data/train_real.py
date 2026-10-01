"""
train_real.py
Trains IDBN on the real Kaggle Crop Recommendation dataset (22 Indian crops).
Saves model + preprocessor for the Flask app.
"""
import sys, os, pickle
sys.path.insert(0, '/home/claude/crop_prediction')   # reuse IDBN

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib, warnings
warnings.filterwarnings('ignore')

from models.idbn import IDBN

# ── 1. Build / load dataset ───────────────────────────────────
from data.build_real_dataset import build
df = build()
print(f"Dataset: {df.shape}  crops={df['label'].nunique()}")

FEATURES = ['N','P','K','temperature','humidity','ph','rainfall']
X_raw = df[FEATURES].values
le    = LabelEncoder()
y     = le.fit_transform(df['label'])

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ── 2. Split ──────────────────────────────────────────────────
Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
Xtr,Xv, ytr,yv  = train_test_split(Xtr,ytr,test_size=0.15,stratify=ytr,random_state=42)
print(f"Train {len(Xtr)} | Val {len(Xv)} | Test {len(Xte)}")

# ── 3. Train IDBN ─────────────────────────────────────────────
# Yield proxy — use N+P+K as a simple yield surrogate for regression head
yr_tr = df.iloc[:len(X)]['N'].values + df.iloc[:len(X)]['P'].values  # proxy
yr_tr_tr = yr_tr[:len(ytr)]; yr_v = yr_tr[len(ytr):len(ytr)+len(yv)]

model = IDBN(n_in=7, n_cls=len(le.classes_),
             hidden=(256,128,64), dropout=0.4, lr=0.001, wd=0.001)

history = model.fit(
    Xtr, ytr, yr_tr_tr,
    Xv,  yv,  yr_v,
    epochs=120, bs=32, pt_epochs=50, verbose=True
)

# ── 4. Evaluate ───────────────────────────────────────────────
preds = model.predict(Xte)
acc   = accuracy_score(yte, preds)*100
print(f"\nTest Accuracy: {acc:.2f}%")
print(classification_report(yte, preds, target_names=le.classes_))

# ── 5. Save ───────────────────────────────────────────────────
os.makedirs('/home/claude/crop_app/models', exist_ok=True)
joblib.dump({'features': FEATURES, 'le': le, 'scaler': scaler},
            '/home/claude/crop_app/models/preprocessor.pkl')
with open('/home/claude/crop_app/models/idbn_model.pkl','wb') as f:
    pickle.dump(model, f)

print("\n✅  Model saved to /home/claude/crop_app/models/")
print(f"    Classes: {list(le.classes_)}")
