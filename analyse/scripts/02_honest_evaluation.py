"""
02_honest_evaluation.py
=======================
Cœur scientifique d'ALT3 : démontrer le biais d'évaluation du pipeline initial
puis fournir une estimation HONNÊTE de la performance de généralisation.

Trois protocoles sont comparés sur EXACTEMENT le même jeu de features :

  (A) BIAISÉ   — train/test aléatoire (shuffle) : reproduit le ~99% du rapport ALT2.
  (B) LEAKAGE  — entraînement sur 100% des données puis prédiction sur ces mêmes
                 données : reproduit le "100% live" du rapport (évaluation in-sample).
  (C) HONNÊTE  — séparation TEMPORELLE par blocs (les snapshots consécutifs d'une
                 même session ne peuvent pas être à la fois en train et en test) :
                 1) holdout temporel  : 70% premiers / 30% derniers par zone
                 2) StratifiedGroupKFold sur blocs temporels (validation croisée group-aware)

Pourquoi ? Chaque zone a été scannée en UNE seule session continue. Des scans
espacés de quelques secondes au même endroit sont quasi identiques. Un split
aléatoire place ces quasi-doublons des deux côtés -> fuite -> accuracy optimiste.

Sortie : tableaux console + CSV dans reports/figures/.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Console Windows : forcer UTF-8 pour les caractères accentués / symboles.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robust_localization import (  # noqa: E402
    RobustFeatureBuilder,
    build_snapshot_tables,
    load_raw_wifi_data,
)
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
N_BLOCKS = 5  # nb de blocs temporels par zone pour la validation honnête


def build_dataset(scope: str = "bssid", max_ids: int = 120, min_freq: int = 5):
    """Construit X, y, en choisissant l'identifiant réseau :
    scope='bssid' -> 1 feature par point d'accès physique (modèle vedette, 529 feat).
    scope='ssid'  -> 1 feature par nom de réseau (variante, ~74 feat).
    """
    raw = load_raw_wifi_data(RAW_DIR)
    if scope == "bssid":
        # le builder pivote sur la colonne 'ssid' : on y injecte le bssid.
        raw = raw.copy()
        raw["ssid"] = raw["bssid"]
    per_ssid, snapshot = build_snapshot_tables(raw)
    builder = RobustFeatureBuilder(max_ssids=max_ids, min_ssid_frequency=min_freq)
    X, y = builder.fit_transform(per_ssid, snapshot)
    return X, y, builder


def course_model_zoo(random_state: int = RANDOM_STATE):
    """Modèles vus en cours IF23 + ensembles. Pipelines avec scaler quand utile."""
    return {
        "k-NN (k=5)": Pipeline([("scaler", StandardScaler()),
                                ("model", KNeighborsClassifier(n_neighbors=5))]),
        "NaiveBayes": Pipeline([("model", GaussianNB())]),
        "SVM (RBF)": Pipeline([("scaler", StandardScaler()),
                               ("model", SVC(kernel="rbf", C=10, gamma="scale",
                                             class_weight="balanced", random_state=random_state))]),
        "LogReg": Pipeline([("scaler", StandardScaler()),
                            ("model", LogisticRegression(max_iter=3000, class_weight="balanced",
                                                         random_state=random_state))]),
        "RandomForest": Pipeline([("model", RandomForestClassifier(
            n_estimators=120, max_depth=20, class_weight="balanced_subsample",
            random_state=random_state, n_jobs=-1))]),
        "ExtraTrees": Pipeline([("model", ExtraTreesClassifier(
            n_estimators=180, max_depth=24, class_weight="balanced",
            random_state=random_state, n_jobs=-1))]),
    }


def time_blocks(index: pd.MultiIndex, n_blocks: int) -> np.ndarray:
    """Affecte un identifiant de bloc temporel à chaque snapshot.
    index = MultiIndex (room, time). Pour chaque zone, les snapshots sont triés
    par temps puis découpés en n_blocks tranches contiguës de tailles ~égales.
    Renvoie un tableau de labels de groupe 'room__bk'.
    """
    df = pd.DataFrame({"room": index.get_level_values("room"),
                       "time": index.get_level_values("time")})
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    groups = np.empty(len(df), dtype=object)
    for room, sub in df.groupby("room"):
        order = sub.sort_values("time").index.to_numpy()
        k = min(n_blocks, len(order))
        chunks = np.array_split(order, k)
        for b, idx in enumerate(chunks):
            for pos in idx:
                groups[pos] = f"{room}__b{b}"
    return groups


def eval_biased_holdout(X, y_enc, models):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=RANDOM_STATE
    )
    rows = []
    for name, model in models.items():
        fitted = clone(model).fit(Xtr, ytr)
        pred = fitted.predict(Xte)
        rows.append({"model": name,
                     "accuracy": accuracy_score(yte, pred),
                     "f1_macro": f1_score(yte, pred, average="macro")})
    return pd.DataFrame(rows)


def eval_leakage(X, y_enc, models):
    """Entraîne et teste sur les MÊMES données (ce que faisait l'éval 'live')."""
    rows = []
    for name, model in models.items():
        fitted = clone(model).fit(X, y_enc)
        pred = fitted.predict(X)
        rows.append({"model": name,
                     "accuracy": accuracy_score(y_enc, pred),
                     "f1_macro": f1_score(y_enc, pred, average="macro")})
    return pd.DataFrame(rows)


def eval_honest_temporal_holdout(X, y_enc, y_room, models, frac_train=0.7):
    """Pour chaque zone : premiers frac_train% (temps) -> train, reste -> test."""
    df = pd.DataFrame({"room": X.index.get_level_values("room"),
                       "time": pd.to_datetime(X.index.get_level_values("time"), errors="coerce")})
    df = df.reset_index(drop=True)
    train_mask = np.zeros(len(df), dtype=bool)
    for room, sub in df.groupby("room"):
        order = sub.sort_values("time").index.to_numpy()
        cut = int(round(len(order) * frac_train))
        train_mask[order[:cut]] = True
    Xtr, Xte = X.iloc[train_mask], X.iloc[~train_mask]
    ytr, yte = y_enc[train_mask], y_enc[~train_mask]
    rows = []
    for name, model in models.items():
        fitted = clone(model).fit(Xtr, ytr)
        pred = fitted.predict(Xte)
        rows.append({"model": name,
                     "accuracy": accuracy_score(yte, pred),
                     "f1_macro": f1_score(yte, pred, average="macro")})
    return pd.DataFrame(rows), int(train_mask.sum()), int((~train_mask).sum())


def eval_honest_groupcv(X, y_enc, models, groups, n_splits=5):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in models.items():
        accs, f1s = [], []
        for tr, te in sgkf.split(X, y_enc, groups=groups):
            fitted = clone(model).fit(X.iloc[tr], y_enc[tr])
            pred = fitted.predict(X.iloc[te])
            accs.append(accuracy_score(y_enc[te], pred))
            f1s.append(f1_score(y_enc[te], pred, average="macro"))
        rows.append({"model": name,
                     "accuracy_mean": float(np.mean(accs)),
                     "accuracy_std": float(np.std(accs)),
                     "f1_macro_mean": float(np.mean(f1s)),
                     "f1_macro_std": float(np.std(f1s))})
    return pd.DataFrame(rows)


def main(scope: str = "bssid"):
    print(f"Construction du dataset (scope={scope.upper()})...")
    X, y_room, builder = build_dataset(scope=scope)
    le = LabelEncoder()
    y_enc = le.fit_transform(y_room.to_numpy())
    print(f"  X = {X.shape[0]} snapshots x {X.shape[1]} features | "
          f"{len(le.classes_)} zones | {len(builder.selected_ssids)} identifiants retenus")

    models = course_model_zoo()
    groups = time_blocks(X.index, N_BLOCKS)
    print(f"  Blocs temporels : {len(set(groups))} groupes (~{N_BLOCKS}/zone)\n")

    print("=" * 78)
    print("(A) BIAISE  — holdout aleatoire 80/20 (shuffle)   [reproduit le ~99% ALT2]")
    print("=" * 78)
    a = eval_biased_holdout(X, y_enc, models)
    print(a.to_string(index=False))

    print("\n" + "=" * 78)
    print("(B) LEAKAGE — train ET test sur 100% des donnees  [reproduit le '100% live']")
    print("=" * 78)
    b = eval_leakage(X, y_enc, models)
    print(b.to_string(index=False))

    print("\n" + "=" * 78)
    print("(C1) HONNETE — holdout TEMPOREL 70/30 par zone")
    print("=" * 78)
    c1, ntr, nte = eval_honest_temporal_holdout(X, y_enc, y_room, models)
    print(f"  train={ntr} snapshots | test={nte} snapshots")
    print(c1.to_string(index=False))

    print("\n" + "=" * 78)
    print(f"(C2) HONNETE — StratifiedGroupKFold ({N_BLOCKS} blocs temporels/zone)")
    print("=" * 78)
    c2 = eval_honest_groupcv(X, y_enc, models, groups, n_splits=N_BLOCKS)
    print(c2.to_string(index=False))

    # Sauvegardes brutes
    suf = scope
    a.to_csv(OUT_DIR / f"eval_A_biased_holdout_{suf}.csv", index=False)
    b.to_csv(OUT_DIR / f"eval_B_leakage_{suf}.csv", index=False)
    c1.to_csv(OUT_DIR / f"eval_C1_temporal_holdout_{suf}.csv", index=False)
    c2.to_csv(OUT_DIR / f"eval_C2_group_cv_{suf}.csv", index=False)

    # Tableau de synthese : tous les modeles x protocoles (accuracy) — coeur du rapport
    summary = pd.DataFrame({
        "model": a["model"],
        "A_aleatoire": a["accuracy"].values,
        "B_insample": b["accuracy"].values,
        "C1_holdout_temporel": c1["accuracy"].values,
        "C2_groupkfold": c2["accuracy_mean"].values,
    })
    summary["chute_A_vers_C2"] = summary["A_aleatoire"] - summary["C2_groupkfold"]
    summary.to_csv(OUT_DIR / f"eval_summary_{suf}.csv", index=False)
    print("\n" + "=" * 78)
    print(f"SYNTHESE ({scope.upper()}) — accuracy par protocole — l'ecart biaise vs honnete")
    print("=" * 78)
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(summary.to_string(index=False))
    print(f"\nCSV ecrits dans {OUT_DIR}")
    return summary


if __name__ == "__main__":
    scope = sys.argv[1] if len(sys.argv) > 1 else "bssid"
    main(scope)
