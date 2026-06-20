"""
16_alt_evolution_real.py — Évolution temporelle RÉELLE ALT1(février) -> ALT2(avril).
Les zones ALT1 (sous-découpage A/B) sont fusionnées pour correspondre à ALT2.
Stratégies de la consigne :
  S1 : train ALT1            -> test ALT2 (zones communes)   [drift pur]
  S2 : train ALT1 + 50% ALT2 -> test 50% restant ALT2        [ré-apprentissage]
  S3 : train ALT1 + ALT2     -> test ALT3                     [à venir, ALT3]
Produit reports/figures/evolution_real.csv, evolution_real_perzone.csv,
fig18_evolution.png
"""
from __future__ import annotations

import re
import sys
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
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robust_localization import RobustFeatureBuilder, build_snapshot_tables, load_raw_wifi_data  # noqa

FIG = ROOT / "reports" / "figures"
RS = 42
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def merge_zone(z: str) -> str:
    """Fusionne les sous-zones A/B (P102A -> P102, S103B -> S103)."""
    return z[:-1] if re.match(r"^[PS]\d{3}[AB]$", str(z)) else str(z)


def load_bssid(raw_dir: Path):
    raw = load_raw_wifi_data(raw_dir)
    raw = raw.copy()
    raw["room"] = raw["room"].map(merge_zone)
    raw["ssid"] = raw["bssid"]  # scope BSSID
    return build_snapshot_tables(raw)


def et():
    return ExtraTreesClassifier(n_estimators=300, max_depth=24, class_weight="balanced",
                               random_state=RS, n_jobs=-1)


def main():
    per1, snap1 = load_bssid(ROOT / "data" / "raw")
    builder = RobustFeatureBuilder(max_ssids=200, min_ssid_frequency=5)
    X1, y1 = builder.fit_transform(per1, snap1)
    y1 = y1.to_numpy()
    per2, snap2 = load_bssid(ROOT / "data" / "raw_alt2")
    X2 = builder.transform(per2, snap2)
    y2 = snap2.set_index(["room", "time"]).loc[X2.index].index.get_level_values("room").to_numpy()

    zones1, zones2 = set(y1), set(y2)
    common = sorted(zones1 & zones2)
    print(f"ALT1 : {len(X1)} snapshots, {len(zones1)} zones | ALT2 : {len(X2)} snapshots, {len(zones2)} zones")
    print(f"Zones communes ({len(common)}) : {common}")
    print(f"Nouvelles en ALT2 : {sorted(zones2 - zones1)}")

    # --- Stratégie 1 : train ALT1 -> test ALT2 (zones communes) ---
    m1 = et().fit(X1, y1)
    mask_common = np.isin(y2, common)
    Xte, yte = X2[mask_common], y2[mask_common]
    pred = m1.predict(Xte)
    acc_s1 = accuracy_score(yte, pred)
    f1_s1 = f1_score(yte, pred, average="macro")
    perzone = []
    for z in common:
        mz = yte == z
        perzone.append({"zone": z, "n": int(mz.sum()),
                        "acc_S1": round(accuracy_score(yte[mz], pred[mz]), 3)})
    print(f"\nStratégie 1 (train ALT1 -> test ALT2) : accuracy = {acc_s1:.3f} | F1 = {f1_s1:.3f}")

    # --- Stratégie 2 : train ALT1 + 50% ALT2 -> test reste ALT2 ---
    rng = np.random.default_rng(RS)
    idx = rng.permutation(len(X2)); half = len(X2) // 2
    tr2, te2 = idx[:half], idx[half:]
    Xtr = pd.concat([X1, X2.iloc[tr2]]); ytr = np.concatenate([y1, y2[tr2]])
    m2 = et().fit(Xtr, ytr)
    pred2 = m2.predict(X2.iloc[te2])
    acc_s2 = accuracy_score(y2[te2], pred2)
    f1_s2 = f1_score(y2[te2], pred2, average="macro")
    print(f"Stratégie 2 (train ALT1+50% ALT2 -> reste) : accuracy = {acc_s2:.3f} | F1 = {f1_s2:.3f}")
    print("Stratégie 3 (train ALT1+ALT2 -> ALT3) : EN ATTENTE du dataset ALT3")

    # Sauvegardes
    pd.DataFrame([
        {"strategie": "S1: ALT1 -> ALT2 (drift pur)", "accuracy": round(acc_s1, 3), "f1_macro": round(f1_s1, 3)},
        {"strategie": "S2: ALT1+50%ALT2 -> reste", "accuracy": round(acc_s2, 3), "f1_macro": round(f1_s2, 3)},
        {"strategie": "S3: ALT1+ALT2 -> ALT3", "accuracy": np.nan, "f1_macro": np.nan},
    ]).to_csv(FIG / "evolution_real.csv", index=False)
    pz = pd.DataFrame(perzone)
    pz.to_csv(FIG / "evolution_real_perzone.csv", index=False)
    print("\nPar zone (Stratégie 1) :"); print(pz.to_string(index=False))

    # --- Figure : par zone S1 + barres globales S1/S2 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]})
    colors = ["#d9534f" if a < 0.3 else ("#f0ad4e" if a < 0.7 else "#27ae60") for a in pz["acc_S1"]]
    ax1.bar(pz["zone"], pz["acc_S1"] * 100, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Accuracy (%)"); ax1.set_ylim(0, 105)
    ax1.set_title("Stratégie 1 — train ALT1 → test ALT2, par zone\nCertaines zones s'effondrent (drift WiFi)")
    for i, a in enumerate(pz["acc_S1"]):
        ax1.text(i, a * 100 + 2, f"{a*100:.0f}", ha="center", fontsize=8)
    ax2.bar(["S1\nALT1→ALT2", "S2\n+50% ALT2"], [acc_s1 * 100, acc_s2 * 100],
            color=["#d9534f", "#27ae60"], edgecolor="black", linewidth=0.6)
    for i, v in enumerate([acc_s1 * 100, acc_s2 * 100]):
        ax2.text(i, v + 2, f"{v:.0f}%", ha="center", fontweight="bold")
    ax2.set_ylim(0, 105); ax2.set_ylabel("Accuracy globale (%)")
    ax2.set_title("Drift puis ré-apprentissage")
    fig.tight_layout(); fig.savefig(FIG / "fig18_evolution.png", bbox_inches="tight"); plt.close(fig)
    print("  -> fig18_evolution.png")


if __name__ == "__main__":
    main()
