"""
03_honest_report_artifacts.py
=============================
Produit les artefacts détaillés pour le rapport, sous le protocole HONNÊTE
(holdout temporel 70/30 par zone) :

  - rapport de classification par zone (precision / recall / F1)  [métriques du cours]
  - accuracy, balanced accuracy, macro-F1, top-3 accuracy
  - matrice de confusion HONNÊTE (figure PNG)
  - importance des features (Random Forest / ExtraTrees) + part des réseaux
    ÉPHÉMÈRES (hotspots/personnels) vs STABLES (institutionnels) dans le top.

Usage : python 03_honest_report_artifacts.py [bssid|ssid]
"""
from __future__ import annotations

import sys
import re
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
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

# Mots-clés des réseaux institutionnels STABLES (présents en permanence dans le bâtiment).
STABLE_KEYWORDS = ["utt", "eduroam", "ucpa", "phpwave"]
# Heuristique réseau personnel/éphémère (nom contenant ces motifs => probablement instable).
EPHEMERAL_HINTS = ["iphone", "galaxy", "redmi", "honor", "huawei", "samsung", "android",
                   "pixel", "oneplus", "xiaomi", "hotspot", "airpods", "macbook", "pc de",
                   "portable", "de "]


def build(scope: str):
    raw = load_raw_wifi_data(RAW_DIR)
    # garder une table bssid->ssid pour interpréter l'importance des features
    bssid_to_ssid = (raw.groupby("bssid")["ssid"]
                     .agg(lambda s: s.value_counts().index[0]).to_dict())
    if scope == "bssid":
        raw = raw.copy()
        raw["ssid"] = raw["bssid"]
    per_ssid, snapshot = build_snapshot_tables(raw)
    builder = RobustFeatureBuilder(max_ssids=120, min_ssid_frequency=5)
    X, y = builder.fit_transform(per_ssid, snapshot)
    return X, y, builder, bssid_to_ssid


def temporal_split(X, y_enc, frac_train=0.7):
    df = pd.DataFrame({"room": X.index.get_level_values("room"),
                       "time": pd.to_datetime(X.index.get_level_values("time"), errors="coerce")})
    df = df.reset_index(drop=True)
    mask = np.zeros(len(df), dtype=bool)
    for room, sub in df.groupby("room"):
        order = sub.sort_values("time").index.to_numpy()
        cut = int(round(len(order) * frac_train))
        mask[order[:cut]] = True
    return mask


def top_k_accuracy(model, X, y_true_enc, k=3):
    proba = model.predict_proba(X)
    topk = np.argsort(proba, axis=1)[:, -k:]
    return float(np.mean([y_true_enc[i] in topk[i] for i in range(len(y_true_enc))]))


def classify_network(label: str, bssid_to_ssid: dict, scope: str) -> str:
    name = bssid_to_ssid.get(label, label) if scope == "bssid" else label
    name = str(name).lower()
    if any(k in name for k in STABLE_KEYWORDS):
        return "stable"
    if any(h in name for h in EPHEMERAL_HINTS):
        return "ephemere"
    return "autre"


def main(scope="bssid"):
    print(f"== Artefacts honnêtes (scope={scope}) ==")
    X, y_room, builder, bssid_to_ssid = build(scope)
    le = LabelEncoder()
    y_enc = le.fit_transform(y_room.to_numpy())
    classes = le.classes_

    mask = temporal_split(X, y_enc, 0.7)
    Xtr, Xte = X.iloc[mask], X.iloc[~mask]
    ytr, yte = y_enc[mask], y_enc[~mask]

    clf = ExtraTreesClassifier(n_estimators=300, max_depth=24, class_weight="balanced",
                               random_state=RANDOM_STATE, n_jobs=-1).fit(Xtr, ytr)
    pred = clf.predict(Xte)

    acc = accuracy_score(yte, pred)
    bal = balanced_accuracy_score(yte, pred)
    f1m = f1_score(yte, pred, average="macro")
    t3 = top_k_accuracy(clf, Xte, yte, k=3)
    print(f"Accuracy={acc:.3f} | BalancedAcc={bal:.3f} | macroF1={f1m:.3f} | Top-3={t3:.3f}")

    # Rapport par zone
    rep = classification_report(yte, pred, target_names=classes, output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(rep).transpose()
    rep_df.to_csv(OUT / f"honest_classification_report_{scope}.csv")
    print("\nPar zone (precision/recall/f1):")
    print(rep_df.loc[list(classes)][["precision", "recall", "f1-score", "support"]]
          .round(3).to_string())

    # Matrice de confusion honnête
    cm = confusion_matrix(yte, pred, labels=range(len(classes)))
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Zone prédite"); ax.set_ylabel("Zone réelle")
    ax.set_title(f"Matrice de confusion — protocole HONNÊTE (holdout temporel) — {scope.upper()}\n"
                 f"acc={acc:.2f}, balanced={bal:.2f}, macroF1={f1m:.2f}")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()*0.5 else "black", fontsize=6)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / f"honest_confusion_matrix_{scope}.png", dpi=150)
    plt.close(fig)

    # Importance des features + part stable vs éphémère
    imp = pd.Series(clf.feature_importances_, index=Xtr.columns)
    # n'agréger que les colonnes liées à un identifiant réseau (pattern stat__id)
    rows = []
    for col, val in imp.items():
        m = re.match(r"(presence|rssi_mean|rssi_max|rssi_std|rssi_count)__(.+)", col)
        if m:
            rows.append({"network": m.group(2), "importance": val})
    netimp = (pd.DataFrame(rows).groupby("network")["importance"].sum()
              .sort_values(ascending=False))
    netimp_df = netimp.reset_index()
    netimp_df["type"] = netimp_df["network"].apply(lambda n: classify_network(n, bssid_to_ssid, scope))
    if scope == "bssid":
        netimp_df["ssid"] = netimp_df["network"].map(bssid_to_ssid)
    netimp_df.to_csv(OUT / f"feature_importance_{scope}.csv", index=False)

    part = netimp_df.groupby("type")["importance"].sum()
    part = (part / part.sum() * 100).round(1)
    print("\nPart de l'importance par type de réseau (%):")
    print(part.to_string())
    print("\nTop 12 réseaux les plus discriminants:")
    cols = ["network", "ssid", "importance", "type"] if scope == "bssid" else ["network", "importance", "type"]
    print(netimp_df.head(12)[cols].to_string(index=False))

    pd.DataFrame([{"scope": scope, "accuracy": acc, "balanced_accuracy": bal,
                   "macro_f1": f1m, "top3": t3,
                   "n_train": int(mask.sum()), "n_test": int((~mask).sum())}]
                 ).to_csv(OUT / f"honest_headline_{scope}.csv", index=False)
    print(f"\nArtefacts écrits dans {OUT}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bssid")
