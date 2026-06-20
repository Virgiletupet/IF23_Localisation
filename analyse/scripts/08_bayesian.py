"""
08_bayesian.py — Évaluation honnête du modèle bayésien du cours.
Compare le GaussianBayesLocalizer (présence Bernoulli + RSSI gaussien, MAP)
au GaussianNB de scikit-learn, en holdout temporel et GroupKFold, scope SSID
(plus adapté au modèle gaussien : moins de features dégénérées).
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
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval_utils import build_dataset, temporal_mask, time_blocks  # noqa: E402
from bayesian_localization import GaussianBayesLocalizer  # noqa: E402

OUT = ROOT / "reports" / "figures"
RS = 42


def eval_model(name, model, X, y):
    # holdout temporel
    m = temporal_mask(X.index, 0.7)
    fitted = clone(model).fit(X.iloc[m], y[m])
    pred = fitted.predict(X.iloc[~m])
    accH = accuracy_score(y[~m], pred)
    balH = balanced_accuracy_score(y[~m], pred)
    f1H = f1_score(y[~m], pred, average="macro")
    # groupkfold
    groups = time_blocks(X.index, 5)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RS)
    accs = []
    for tr, te in sgkf.split(X, y, groups=groups):
        f = clone(model).fit(X.iloc[tr], y[tr])
        accs.append(accuracy_score(y[te], f.predict(X.iloc[te])))
    return {"modele": name, "holdout_acc": round(accH, 3), "holdout_bal": round(balH, 3),
            "holdout_f1": round(f1H, 3), "groupkfold_acc": round(float(np.mean(accs)), 3),
            "groupkfold_std": round(float(np.std(accs)), 3)}


def main(scope="ssid"):
    X, y_room, builder, _ = build_dataset(scope=scope)
    le = LabelEncoder(); y = le.fit_transform(y_room.to_numpy())
    print(f"Scope={scope} : {X.shape[0]} snapshots x {X.shape[1]} features | {len(le.classes_)} zones\n")

    models = {
        "Bayésien gaussien (cours, custom)": GaussianBayesLocalizer(var_floor=4.0),
        "GaussianNB (sklearn)": Pipeline([("m", GaussianNB())]),
    }
    rows = [eval_model(n, m, X, y) for n, m in models.items()]
    df = pd.DataFrame(rows)
    print("=" * 80)
    print(f"MODÈLE BAYÉSIEN — évaluation honnête (scope {scope.upper()})")
    print("=" * 80)
    print(df.to_string(index=False))
    df.to_csv(OUT / f"bayesian_{scope}.csv", index=False)
    print(f"\nCSV -> {OUT / f'bayesian_{scope}.csv'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ssid")
