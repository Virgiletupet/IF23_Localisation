"""
07_alt1_reproduction.py
=======================
Reproduit fidèlement le pipeline ALT1 (jalon février) : dataset_unified.csv,
6 BSSID communs, pivot (Room,Time), Random Forest, split aléatoire 75/25.
Puis applique le protocole HONNÊTE (split temporel) pour mesurer la vraie
performance d'ALT1. Sert à documenter le jalon 1 avec de vrais chiffres.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval_utils import build_alt1_dataset, temporal_mask, time_blocks  # noqa: E402

OUT = ROOT / "reports" / "figures"
RS = 42


def rf():
    return RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5,
                                  min_samples_leaf=2, random_state=RS, n_jobs=-1)


def main():
    X, y_room = build_alt1_dataset()
    le = LabelEncoder(); y = le.fit_transform(y_room.to_numpy())
    print(f"ALT1 : {X.shape[0]} snapshots x {X.shape[1]} BSSID | {len(le.classes_)} zones")

    # (A) Biaisé — reproduit l'historique (split aléatoire 75/25)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RS)
    m = rf().fit(Xtr, ytr); pa = m.predict(Xte)
    accA = accuracy_score(yte, pa); f1A = f1_score(yte, pa, average="macro")

    # (C1) Honnête — holdout temporel 70/30
    mask = temporal_mask(X.index, 0.7)
    m2 = rf().fit(X.iloc[mask], y[mask]); pc = m2.predict(X.iloc[~mask])
    accC1 = accuracy_score(y[~mask], pc)
    balC1 = balanced_accuracy_score(y[~mask], pc)
    f1C1 = f1_score(y[~mask], pc, average="macro")

    # (C2) Honnête — GroupKFold temporel
    groups = time_blocks(X.index, 5)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RS)
    accs = []
    for tr, te in sgkf.split(X, y, groups=groups):
        f = rf().fit(X.iloc[tr], y[tr]); accs.append(accuracy_score(y[te], f.predict(X.iloc[te])))
    accC2 = float(np.mean(accs))

    df = pd.DataFrame([{
        "jalon": "ALT1 (6 BSSID, 16 zones, RF)",
        "biaise_aleatoire": round(accA, 3), "biaise_f1": round(f1A, 3),
        "honnete_holdout_temp": round(accC1, 3),
        "honnete_holdout_bal": round(balC1, 3), "honnete_holdout_f1": round(f1C1, 3),
        "honnete_groupkfold": round(accC2, 3),
    }])
    print("\n=== ALT1 — biaisé vs honnête ===")
    print(df.T.to_string(header=False))
    df.to_csv(OUT / "alt1_reproduction.csv", index=False)
    print(f"\nCSV -> {OUT / 'alt1_reproduction.csv'}")


if __name__ == "__main__":
    main()
