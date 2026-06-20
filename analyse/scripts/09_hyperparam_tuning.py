"""
09_hyperparam_tuning.py — Optimisation des hyperparamètres (group-aware).
Recherche par grille avec validation croisée StratifiedGroupKFold (anti-fuite)
pour SVM (C, gamma), k-NN (k, pondération) et ExtraTrees (arbres, profondeur).
On compare le score honnête par défaut au score honnête après réglage.
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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval_utils import build_dataset, time_blocks  # noqa: E402

OUT = ROOT / "reports" / "figures"
RS = 42


def tune(name, pipe, grid, default_pipe, X, y, groups, cv):
    base = cross_val_score(default_pipe, X, y, groups=groups, cv=cv, scoring="accuracy").mean()
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=-1)
    gs.fit(X, y, groups=groups)
    best = {k.split("__")[-1]: v for k, v in gs.best_params_.items()}
    print(f"\n{name}")
    print(f"  défaut  (GroupKFold) : {base:.3f}")
    print(f"  optimisé(GroupKFold) : {gs.best_score_:.3f}   | meilleurs params : {best}")
    return {"modele": name, "defaut": round(base, 3), "optimise": round(gs.best_score_, 3),
            "gain": round(gs.best_score_ - base, 3), "params": str(best)}


def main(scope="bssid"):
    X, y_room, _, _ = build_dataset(scope=scope)
    le = LabelEncoder(); y = le.fit_transform(y_room.to_numpy())
    groups = time_blocks(X.index, 5)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RS)
    print(f"Optimisation hyperparamètres (scope {scope}, {X.shape[1]} features) — CV group-aware\n" + "="*70)

    rows = []
    rows.append(tune(
        "SVM (RBF)",
        Pipeline([("sc", StandardScaler()), ("m", SVC(class_weight="balanced", random_state=RS))]),
        {"m__C": [1, 10, 100], "m__gamma": ["scale", 0.01, 0.001]},
        Pipeline([("sc", StandardScaler()), ("m", SVC(kernel="rbf", C=10, gamma="scale",
                  class_weight="balanced", random_state=RS))]),
        X, y, groups, cv))
    rows.append(tune(
        "k-NN",
        Pipeline([("sc", StandardScaler()), ("m", KNeighborsClassifier())]),
        {"m__n_neighbors": [1, 3, 5, 7, 9], "m__weights": ["uniform", "distance"]},
        Pipeline([("sc", StandardScaler()), ("m", KNeighborsClassifier(n_neighbors=5))]),
        X, y, groups, cv))
    rows.append(tune(
        "ExtraTrees",
        Pipeline([("m", ExtraTreesClassifier(class_weight="balanced", random_state=RS, n_jobs=-1))]),
        {"m__n_estimators": [180, 400], "m__max_depth": [24, 40, None],
         "m__min_samples_leaf": [1, 2]},
        Pipeline([("m", ExtraTreesClassifier(n_estimators=180, max_depth=24,
                  class_weight="balanced", random_state=RS, n_jobs=-1))]),
        X, y, groups, cv))

    df = pd.DataFrame(rows)
    print("\n" + "="*70)
    print("SYNTHÈSE — gain du réglage des hyperparamètres (accuracy GroupKFold)")
    print("="*70)
    print(df[["modele", "defaut", "optimise", "gain", "params"]].to_string(index=False))
    df.to_csv(OUT / f"hyperparam_tuning_{scope}.csv", index=False)
    print(f"\nCSV -> {OUT / f'hyperparam_tuning_{scope}.csv'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bssid")
