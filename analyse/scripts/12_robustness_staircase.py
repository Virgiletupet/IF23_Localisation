"""
12_robustness_staircase.py — "Escalier de réalisme" (storytelling central).

On part du test de base (split aléatoire, 99 %) et on ajoute un à un des
facteurs réalistes qui rapprochent l'évaluation des conditions réelles (live) :
  1. test de base (split aléatoire)
  2. + séparation temporelle (validation croisée group-aware, anti-fuite)
  3. + holdout temporel strict (test = fin de session, le plus dur)
  4. + retrait des réseaux personnels (absents/différents en live)
  5. + bruit de mesure RSSI (variabilité appareil/instant)
On montre que l'évaluation honnête atterrit dans la bande de performance
réellement observée en live (~70-80 %) : la méthodologie prédit la réalité.

Produit : reports/figures/staircase.csv, fig13_escalier.png, fig14_bruit.png
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
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval_utils import build_dataset, temporal_mask, time_blocks  # noqa: E402

FIG = ROOT / "reports" / "figures"
RS = 42
LIVE_LOW, LIVE_HIGH = 0.70, 0.80
INSTITUTION_KEYWORDS = ["utt", "eduroam", "ucpa"]
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})


def et():
    return ExtraTreesClassifier(n_estimators=300, max_depth=24, class_weight="balanced",
                               random_state=RS, n_jobs=-1)


def institution_cols(columns, bssid_to_ssid):
    keep = []
    import re
    for c in columns:
        m = re.match(r"(presence|rssi_mean|rssi_max|rssi_std|rssi_count)__(.+)", c)
        if m:
            ssid = str(bssid_to_ssid.get(m.group(2), "")).lower()
            if any(k in ssid for k in INSTITUTION_KEYWORDS):
                keep.append(c)
        else:
            keep.append(c)  # colonnes globales conservées
    return keep


def add_rssi_noise(X, sigma, rng):
    Xn = X.copy()
    for c in Xn.columns:
        if c.startswith(("rssi_mean__", "rssi_max__")):
            col = Xn[c].to_numpy(dtype=float)
            present = col > -100.0
            col[present] = col[present] + rng.normal(0, sigma, present.sum())
            Xn[c] = np.clip(col, -100, -20)
    return Xn


def main():
    X, y_room, builder, bssid_to_ssid = build_dataset(scope="bssid")
    le = LabelEncoder(); y = le.fit_transform(y_room.to_numpy())
    rng = np.random.default_rng(RS)
    print(f"Dataset : {X.shape} | {len(le.classes_)} zones")

    steps = []

    # 1) Test de base : split aléatoire
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RS)
    acc = accuracy_score(yte, et().fit(Xtr, ytr).predict(Xte))
    steps.append(("1. Test de base\n(split aléatoire)", acc))

    # Base honnête : masque temporel
    m = temporal_mask(X.index, 0.7)
    Xtr_h, Xte_h, ytr_h, yte_h = X.iloc[m], X.iloc[~m], y[m], y[~m]

    # 2) Séparation temporelle (GroupKFold)
    groups = time_blocks(X.index, 5)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RS)
    accs = [accuracy_score(y[te], clone(et()).fit(X.iloc[tr], y[tr]).predict(X.iloc[te]))
            for tr, te in sgkf.split(X, y, groups=groups)]
    steps.append(("2. + séparation\ntemporelle (CV)", float(np.mean(accs))))

    # 3) Holdout temporel strict
    model_all = et().fit(Xtr_h, ytr_h)
    acc = accuracy_score(yte_h, model_all.predict(Xte_h))
    steps.append(("3. + holdout temporel\nstrict (fin de session)", acc))

    # 4) + retrait des réseaux personnels (institution-only)
    inst = institution_cols(list(X.columns), bssid_to_ssid)
    model_inst = et().fit(Xtr_h[inst], ytr_h)
    acc = accuracy_score(yte_h, model_inst.predict(Xte_h[inst]))
    steps.append(("4. + retrait réseaux\npersonnels", acc))

    # 5) + bruit de mesure RSSI (sigma=4 dBm) sur le test
    Xte_noisy = add_rssi_noise(Xte_h[inst], sigma=4.0, rng=rng)
    acc = accuracy_score(yte_h, model_inst.predict(Xte_noisy))
    steps.append(("5. + bruit de mesure\n(±4 dBm, ~live)", acc))

    df = pd.DataFrame(steps, columns=["etape", "accuracy"])
    df.to_csv(FIG / "staircase.csv", index=False)
    print("\n=== Escalier de réalisme ===")
    for e, a in steps:
        print(f"  {e.replace(chr(10),' ')}: {a*100:.1f} %")

    # ---- Figure escalier ----
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    xs = np.arange(len(steps))
    vals = [a*100 for _, a in steps]
    colors = ["#d9534f"] + ["#e67e22", "#f0ad4e"] + ["#5b9bd5", "#27ae60"]
    ax.axhspan(LIVE_LOW*100, LIVE_HIGH*100, color="#27ae60", alpha=0.13, zorder=0)
    ax.text(len(steps)-1, (LIVE_HIGH)*100+0.5, "Performance live observée (~70-80 %)",
            ha="right", color="#1e7e34", fontsize=9, style="italic")
    ax.step(np.append(xs, xs[-1]+1)-0.5, np.append(vals, vals[-1]), where="post",
            color="#555", linewidth=1.2, zorder=1)
    bars = ax.bar(xs, vals, color=colors, edgecolor="black", linewidth=0.6, width=0.7, zorder=2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+1, f"{v:.0f}%", ha="center", fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels([e for e, _ in steps], fontsize=8.5)
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 108)
    ax.set_title("L'escalier de réalisme : du test de base (99 %) à la réalité du live (~75 %)\n"
                 "Chaque facteur réaliste ajouté rapproche la mesure des conditions réelles")
    fig.tight_layout(); fig.savefig(FIG / "fig13_escalier.png", bbox_inches="tight"); plt.close(fig)
    print("  -> fig13_escalier.png")

    # ---- Figure : sensibilité au bruit ----
    sigmas = [0, 2, 4, 6, 8, 10]
    accs_noise = []
    for s in sigmas:
        if s == 0:
            accs_noise.append(accuracy_score(yte_h, model_inst.predict(Xte_h[inst])))
        else:
            Xn = add_rssi_noise(Xte_h[inst], sigma=float(s), rng=np.random.default_rng(RS))
            accs_noise.append(accuracy_score(yte_h, model_inst.predict(Xn)))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhspan(LIVE_LOW*100, LIVE_HIGH*100, color="#27ae60", alpha=0.13)
    ax.plot(sigmas, [a*100 for a in accs_noise], "o-", color="#2c7fb8", linewidth=2)
    for s, a in zip(sigmas, accs_noise):
        ax.text(s, a*100+1, f"{a*100:.0f}%", ha="center", fontsize=8)
    ax.set_xlabel("Bruit ajouté sur le RSSI au test (écart-type, dBm)")
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100)
    ax.set_title("Robustesse au bruit de mesure (modèle institutionnel, holdout temporel)\n"
                 "Dégradation progressive cohérente avec la variabilité radio réelle")
    fig.tight_layout(); fig.savefig(FIG / "fig14_bruit.png", bbox_inches="tight"); plt.close(fig)
    print("  -> fig14_bruit.png")


if __name__ == "__main__":
    main()
