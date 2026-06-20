"""
10_hierarchical.py — Localisation hiérarchique étage -> salle (approche hybride).
Première étape de l'approche hybride du cours (classifier la zone grossière puis
affiner). On compare :
  - un classifieur PLAT (24 salles directement),
  - un classifieur HIÉRARCHIQUE : on prédit d'abord l'étage (RDC/Etage1/Etage2)
    puis la salle au sein de l'étage prédit.
Évaluation en holdout temporel honnête. On reporte aussi l'accuracy étage et le
plafond "étage oracle" (salle sachant l'étage réel).
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
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval_utils import build_dataset, floor_of, temporal_mask  # noqa: E402

OUT = ROOT / "reports" / "figures"
RS = 42


def et():
    return ExtraTreesClassifier(n_estimators=300, max_depth=24, class_weight="balanced",
                                random_state=RS, n_jobs=-1)


def main(scope="bssid"):
    X, y_room, _, _ = build_dataset(scope=scope)
    rooms = y_room.to_numpy()
    floors = np.array([floor_of(r) for r in rooms])
    print(f"Scope={scope} | étages: {sorted(set(floors))}")
    for f in sorted(set(floors)):
        print(f"  {f}: {sorted(set(rooms[floors==f]))}")

    m = temporal_mask(X.index, 0.7)
    Xtr, Xte = X.iloc[m], X.iloc[~m]
    rtr, rte = rooms[m], rooms[~m]
    ftr, fte = floors[m], floors[~m]

    # 1) Plat
    flat = et().fit(Xtr, rtr)
    acc_flat = accuracy_score(rte, flat.predict(Xte))

    # 2) Étage
    fclf = et().fit(Xtr, ftr)
    fpred = fclf.predict(Xte)
    acc_floor = accuracy_score(fte, fpred)

    # 3) Sous-classifieurs par étage
    room_clfs = {}
    for f in sorted(set(ftr)):
        idx = ftr == f
        if len(set(rtr[idx])) > 1:
            room_clfs[f] = et().fit(Xtr[idx], rtr[idx])
        else:
            room_clfs[f] = ("const", rtr[idx][0])

    def predict_room(Xrows, fpredictions):
        out = []
        for i in range(len(fpredictions)):
            f = fpredictions[i]
            clf = room_clfs.get(f)
            row = Xrows.iloc[[i]]
            if isinstance(clf, tuple):
                out.append(clf[1])
            elif clf is None:
                out.append(flat.predict(row)[0])
            else:
                out.append(clf.predict(row)[0])
        return np.array(out)

    # Hiérarchique (étage prédit)
    hier = predict_room(Xte, fpred)
    acc_hier = accuracy_score(rte, hier)
    # Plafond (étage oracle)
    oracle = predict_room(Xte, fte)
    acc_oracle = accuracy_score(rte, oracle)

    df = pd.DataFrame([
        {"approche": "Plat (24 salles)", "accuracy": round(acc_flat, 3)},
        {"approche": "Classification d'étage", "accuracy": round(acc_floor, 3)},
        {"approche": "Hiérarchique (étage prédit -> salle)", "accuracy": round(acc_hier, 3)},
        {"approche": "Hiérarchique (étage oracle -> salle)", "accuracy": round(acc_oracle, 3)},
    ])
    print("\n" + "=" * 70)
    print(f"APPROCHE HIÉRARCHIQUE — holdout temporel honnête (scope {scope})")
    print("=" * 70)
    print(df.to_string(index=False))
    df.to_csv(OUT / f"hierarchical_{scope}.csv", index=False)
    print(f"\nCSV -> {OUT / f'hierarchical_{scope}.csv'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bssid")
