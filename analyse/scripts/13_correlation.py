"""
13_correlation.py — Étude des jeux ALT1 : corrélation entre réseaux (consigne).
Calcule la corrélation des RSSI entre points d'accès, identifie les réseaux
redondants (fortement corrélés) et la couverture des réseaux par zone.
Sert à justifier la sélection de BSSIDs.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval_utils import build_dataset  # noqa: E402

FIG = ROOT / "reports" / "figures"
plt.rcParams.update({"figure.dpi": 150, "font.size": 9})


def main():
    X, y_room, builder, bssid_to_ssid = build_dataset(scope="bssid")
    rssi_cols = [c for c in X.columns if c.startswith("rssi_mean__")]
    R = X[rssi_cols].copy()
    R.columns = [c.replace("rssi_mean__", "") for c in rssi_cols]

    # Corrélation entre réseaux (sur les RSSI moyens)
    corr = R.corr().fillna(0.0)

    # Couverture : part des snapshots où chaque réseau est présent
    pres_cols = [c for c in X.columns if c.startswith("presence__")]
    coverage = X[pres_cols].mean().sort_values(ascending=False)
    coverage.index = [c.replace("presence__", "") for c in coverage.index]
    n_global = int((coverage > 0.9).sum())
    n_rare = int((coverage < 0.2).sum())
    print(f"{len(rssi_cols)} réseaux | présents partout (>90% snapshots): {n_global} "
          f"| rares (<20%): {n_rare}")

    # Paires fortement corrélées (redondance)
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = corr.iloc[i, j]
            if abs(c) > 0.9:
                pairs.append((cols[i], cols[j], round(float(c), 3)))
    print(f"Paires de réseaux très corrélées (|r|>0.9) : {len(pairs)}")
    for a, b, c in pairs[:8]:
        print(f"   {bssid_to_ssid.get(a,a)[:18]:18s} ~ {bssid_to_ssid.get(b,b)[:18]:18s} r={c}")
    pd.DataFrame(pairs, columns=["bssid_a", "bssid_b", "r"]).to_csv(
        FIG / "correlation_pairs.csv", index=False)

    # Heatmap sur les 25 réseaux les plus couvrants (lisibilité)
    top = coverage.head(25).index.tolist()
    sub = corr.loc[top, top]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(sub.values, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [str(bssid_to_ssid.get(t, t))[:14] for t in top]
    ax.set_xticks(range(len(top))); ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(labels, fontsize=6)
    ax.set_title("Corrélation des RSSI entre réseaux (25 plus couvrants)\n"
                 "Des réseaux très corrélés sont redondants pour la classification")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(FIG / "fig16_correlation.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> fig16_correlation.png")

    # Figure couverture
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(len(coverage)), coverage.values * 100, color="#2c7fb8")
    ax.fill_between(range(len(coverage)), coverage.values * 100, alpha=0.2, color="#2c7fb8")
    ax.set_xlabel("Réseaux (triés par couverture)"); ax.set_ylabel("Présence (% des snapshots)")
    ax.set_title("Couverture des réseaux : quelques réseaux stables, beaucoup d'éphémères")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "fig17_couverture.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> fig17_couverture.png")


if __name__ == "__main__":
    main()
