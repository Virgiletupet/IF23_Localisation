"""
04_sanitized_model.py
=====================
Amélioration ALT3 : tester un modèle "assaini" n'utilisant QUE les réseaux
d'infrastructure stables (institutionnels UTT/eduroam...), en retirant les
hotspots personnels / éphémères (téléphones, Direct-*, etc.) qui ne se
généralisent pas d'une session à l'autre.

Compare honnêtement (holdout temporel 70/30 + StratifiedGroupKFold) trois
ensembles de features sur le scope BSSID :
  - ALL          : tous les réseaux (référence honnête)
  - INSTITUTION  : seulement les BSSID dont le SSID est institutionnel
  - PERSO        : seulement les réseaux NON institutionnels
Le but est de montrer (a) la robustesse du modèle assaini et (b) que les
réseaux personnels seuls portent un signal "qui/quand" (preuve du biais).
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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robust_localization import (  # noqa: E402
    RobustFeatureBuilder,
    build_snapshot_tables,
    load_raw_wifi_data,
)

RAW_DIR = ROOT / "data" / "raw"
OUT = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
N_BLOCKS = 5

INSTITUTION_KEYWORDS = ["utt", "eduroam", "ucpa"]


def is_institution(ssid: str) -> bool:
    s = str(ssid).lower()
    return any(k in s for k in INSTITUTION_KEYWORDS)


def time_blocks(index, n_blocks):
    df = pd.DataFrame({"room": index.get_level_values("room"),
                       "time": pd.to_datetime(index.get_level_values("time"), errors="coerce")})
    groups = np.empty(len(df), dtype=object)
    for room, sub in df.groupby("room"):
        order = sub.sort_values("time").index.to_numpy()
        for b, idx in enumerate(np.array_split(order, min(n_blocks, len(order)))):
            for pos in idx:
                groups[pos] = f"{room}__b{b}"
    return groups


def temporal_mask(index, frac=0.7):
    df = pd.DataFrame({"room": index.get_level_values("room"),
                       "time": pd.to_datetime(index.get_level_values("time"), errors="coerce")})
    df = df.reset_index(drop=True)
    mask = np.zeros(len(df), dtype=bool)
    for room, sub in df.groupby("room"):
        order = sub.sort_values("time").index.to_numpy()
        mask[order[:int(round(len(order) * frac))]] = True
    return mask


def build_filtered(kind: str):
    """kind in {all, institution, perso}. Renvoie X, y(room), n_networks."""
    raw = load_raw_wifi_data(RAW_DIR)
    bssid_to_ssid = (raw.groupby("bssid")["ssid"]
                     .agg(lambda s: s.value_counts().index[0]).to_dict())
    raw = raw.copy()
    raw["ssid"] = raw["bssid"]  # scope BSSID
    if kind == "institution":
        keep = {b for b in raw["ssid"].unique() if is_institution(bssid_to_ssid.get(b, ""))}
        raw = raw[raw["ssid"].isin(keep)]
    elif kind == "perso":
        keep = {b for b in raw["ssid"].unique() if not is_institution(bssid_to_ssid.get(b, ""))}
        raw = raw[raw["ssid"].isin(keep)]
    per_ssid, snapshot = build_snapshot_tables(raw)
    builder = RobustFeatureBuilder(max_ssids=200, min_ssid_frequency=5)
    X, y = builder.fit_transform(per_ssid, snapshot)
    return X, y, len(builder.selected_ssids)


def evaluate(kind: str):
    X, y_room, n_net = build_filtered(kind)
    le = LabelEncoder()
    y = le.fit_transform(y_room.to_numpy())

    # Holdout temporel
    m = temporal_mask(X.index, 0.7)
    clf = ExtraTreesClassifier(n_estimators=300, max_depth=24, class_weight="balanced",
                               random_state=RANDOM_STATE, n_jobs=-1).fit(X.iloc[m], y[m])
    pred = clf.predict(X.iloc[~m])
    th_acc = accuracy_score(y[~m], pred)
    th_bal = balanced_accuracy_score(y[~m], pred)
    th_f1 = f1_score(y[~m], pred, average="macro")

    # GroupKFold
    groups = time_blocks(X.index, N_BLOCKS)
    sgkf = StratifiedGroupKFold(n_splits=N_BLOCKS, shuffle=True, random_state=RANDOM_STATE)
    accs = []
    base = ExtraTreesClassifier(n_estimators=300, max_depth=24, class_weight="balanced",
                                random_state=RANDOM_STATE, n_jobs=-1)
    for tr, te in sgkf.split(X, y, groups=groups):
        f = clone(base).fit(X.iloc[tr], y[tr])
        accs.append(accuracy_score(y[te], f.predict(X.iloc[te])))
    return {"jeu_features": kind, "n_features": X.shape[1], "n_reseaux": n_net,
            "holdout_temp_acc": round(th_acc, 3), "holdout_temp_bal": round(th_bal, 3),
            "holdout_temp_macroF1": round(th_f1, 3),
            "groupkfold_acc": round(float(np.mean(accs)), 3),
            "groupkfold_std": round(float(np.std(accs)), 3)}


def main():
    rows = [evaluate(k) for k in ["all", "institution", "perso"]]
    df = pd.DataFrame(rows)
    print("=" * 90)
    print("MODÈLE ASSAINI — ExtraTrees, scope BSSID, protocoles honnêtes")
    print("=" * 90)
    print(df.to_string(index=False))
    df.to_csv(OUT / "sanitized_comparison.csv", index=False)
    print(f"\nCSV -> {OUT / 'sanitized_comparison.csv'}")
    print("\nLecture : 'institution' = modèle déployable robuste (sans téléphones perso) ;")
    print("'perso' seul portant un signal = preuve que le modèle complet exploitait un biais 'qui/quand'.")


if __name__ == "__main__":
    main()
