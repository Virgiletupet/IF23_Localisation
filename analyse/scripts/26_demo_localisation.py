"""
26_demo_localisation.py — Démo de localisation complète sur un vrai scan ALT2.
Chaîne : scan -> étage + zone prédite (classification) + top-3 -> affichage sur
le croquis d'étage + position (x, y) d'exemple. Illustre la cartographie de bout
en bout sur une mesure réelle.
Produit fig24_demo_localisation.png
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
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib import font_manager
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
sys.path.insert(0, str(ROOT / "src"))
from robust_localization import RobustFeatureBuilder, build_snapshot_tables, load_raw_wifi_data  # noqa

RS = 42
NAVY = "#001E62"; BLUE = "#1F77B4"; LIGHT = "#D6E6F5"; GREEN = "#2E8B57"; RED = "#C0392B"
GREY = "#E9ECEF"; CORR = "#F4D58D"
fams = {f.name for f in font_manager.fontManager.ttflist}
FONT = "Segoe UI" if "Segoe UI" in fams else "DejaVu Sans"
plt.rcParams.update({"figure.dpi": 150, "font.family": FONT})


def merge_zone(z):
    return z[:-1] if re.match(r"^[PS]\d{3}[AB]$", str(z)) else str(z)


def load_bssid(d):
    raw = load_raw_wifi_data(d); raw = raw.copy()
    raw["room"] = raw["room"].map(merge_zone); raw["ssid"] = raw["bssid"]
    return build_snapshot_tables(raw)


def room(ax, x, y, w, hh, label, highlight=False, pos=None):
    fc = "#FBE3DE" if highlight else LIGHT
    ec = RED if highlight else BLUE
    ax.add_patch(FancyBboxPatch((x, y), w, hh, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=2.2 if highlight else 1.2))
    ax.text(x + w / 2, y + hh / 2 + (0.18 if pos is not None else 0), label, ha="center", va="center",
            fontsize=11, fontweight="bold", color=RED if highlight else NAVY)
    if pos is not None:
        px, py = pos
        ax.plot(x + 0.15 + px * (w - 0.3), y + 0.15 + py * (hh - 0.3), "o", color=GREEN, ms=12,
                mec="white", mew=1.5, zorder=5)
        ax.text(x + w / 2, y + hh / 2 - 0.30, "position (x, y)", ha="center", fontsize=8,
                color=GREEN, style="italic")


def draw_floor(ax, section, floor, highlight, pos=None):
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title(f"Étage {floor} — section {section}", color=NAVY, fontsize=11, fontweight="bold")
    ax.add_patch(Rectangle((0.3, 2.5), 9.4, 1.0, fc=CORR, ec="#C9A227", lw=1))
    ax.text(5, 3.0, "couloir", ha="center", va="center", fontsize=8, color="#8a6d1b")
    for lbl, x in zip([f"{section}{floor}01", f"{section}{floor}02"], [0.6, 4.0]):
        room(ax, x, 3.7, 3.0, 1.9, lbl, highlight=(lbl == highlight), pos=pos if lbl == highlight else None)
    for lbl, x in zip([f"{section}{floor}03", f"{section}{floor}04"], [0.6, 4.0]):
        room(ax, x, 0.4, 3.0, 1.9, lbl, highlight=(lbl == highlight), pos=pos if lbl == highlight else None)
    ax.add_patch(FancyBboxPatch((7.7, 0.4), 1.9, 5.2, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=GREY, ec="#9AA0A6", lw=1.2))
    ax.text(8.65, 3.0, f"PALIER\n{floor}", ha="center", va="center", fontsize=8, color="#555")


def main():
    per1, snap1 = load_bssid(ROOT / "data" / "raw")
    builder = RobustFeatureBuilder(max_ssids=200, min_ssid_frequency=5)
    X1, y1 = builder.fit_transform(per1, snap1)
    le = LabelEncoder(); ye = le.fit_transform(y1.to_numpy()); zones = list(le.classes_)
    clf = ExtraTreesClassifier(n_estimators=300, max_depth=24, class_weight="balanced",
                               random_state=RS, n_jobs=-1).fit(X1, ye)
    # un vrai scan ALT2 d'une salle reconnue (ex. S102)
    per2, snap2 = load_bssid(ROOT / "data" / "raw_alt2")
    X2 = builder.transform(per2, snap2)
    y2 = snap2.set_index(["room", "time"]).loc[X2.index].index.get_level_values("room").to_numpy()
    cible = "S102"
    idx = np.where(y2 == cible)[0]
    sample = X2.iloc[[idx[len(idx) // 2]]]
    proba = clf.predict_proba(sample)[0]
    order = np.argsort(proba)[::-1][:3]
    top3 = [(zones[i], proba[i]) for i in order]
    pred = top3[0][0]
    print("Vraie zone:", cible, "| Prédite:", pred, "| Top-3:", [(z, round(p, 2)) for z, p in top3])

    sec, fl = pred[0], int(pred[1])
    fig = plt.figure(figsize=(11, 4.6))
    ax1 = fig.add_axes([0.02, 0.05, 0.62, 0.86])
    draw_floor(ax1, sec, fl, highlight=pred, pos=(0.35, 0.6))
    ax2 = fig.add_axes([0.70, 0.18, 0.27, 0.62])
    names = [z for z, _ in top3][::-1]; vals = [p * 100 for _, p in top3][::-1]
    cols = [GREEN if z == pred else BLUE for z in names]
    ax2.barh(names, vals, color=cols)
    for i, v in enumerate(vals):
        ax2.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=9, color=NAVY)
    ax2.set_xlim(0, 105); ax2.set_xlabel("Confiance"); ax2.set_title("Top-3 zones", color=NAVY, fontsize=10)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Localisation de bout en bout sur un scan réel (ALT2) — zone prédite : {pred}",
                 color=NAVY, fontsize=12, fontweight="bold")
    fig.savefig(FIG / "fig24_demo_localisation.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  -> fig24_demo_localisation.png")


if __name__ == "__main__":
    main()
