# IF23 — Localisation Indoor WiFi par fingerprinting (UTT)

Prédiction automatique de la zone d'un utilisateur dans le bâtiment UTT à partir
des signaux WiFi (RSSI) captés par un terminal mobile.

Projet conduit sur trois périodes :
- **ALT1 (février)** — pipeline exploratoire, peu concluant.
- **ALT2 (avril)** — pipeline robuste multi-SSID, bons scores hors-ligne mais détection
  live décevante.
- **ALT3 (juin)** — audit critique de l'évaluation, **correction du biais méthodologique**
  (fuite de données), ajout des modèles vus en cours (k-NN, Bayésien, SVM) et
  cartographie de la version live.

## Arborescence

```
IF23_Localisation/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/          # CSV bruts par zone (wifi_<zone>.csv) — campagne février 2026
│   ├── raw_avril/    # (à compléter) campagne avril 2026 = test de généralisation
│   └── cleaned/      # CSV nettoyés + dataset_unified.csv
├── src/
│   └── robust_localization.py   # module cœur : features, modèles, prédiction
├── notebooks/        # notebooks d'entraînement / évaluation / live
├── models/           # artefacts : modèle, feature_builder, label_encoder, métriques
├── reports/
│   └── figures/      # figures générées pour le rapport
└── scripts/          # scripts reproductibles (1 par étape)
    └── 02_honest_evaluation.py
```

## Installation

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python -m pip install -r IF23_Localisation\requirements.txt
```

> Python **3.10** requis (les artefacts `.pkl` sont sérialisés en cpython-310 ;
> scikit-learn 1.5.x).

## Reproduire les résultats

```powershell
.venv\Scripts\python IF23_Localisation\scripts\02_honest_evaluation.py bssid  # biaisé vs honnête
.venv\Scripts\python IF23_Localisation\scripts\03_honest_report_artifacts.py  # par zone, importance
.venv\Scripts\python IF23_Localisation\scripts\04_sanitized_model.py          # modèle assaini
.venv\Scripts\python IF23_Localisation\scripts\05_figures.py                  # figures principales
.venv\Scripts\python IF23_Localisation\scripts\07_alt1_reproduction.py        # jalon ALT1
.venv\Scripts\python IF23_Localisation\scripts\08_bayesian.py                 # bayésien (cours)
.venv\Scripts\python IF23_Localisation\scripts\09_hyperparam_tuning.py        # optimisation
.venv\Scripts\python IF23_Localisation\scripts\10_hierarchical.py             # hybride étage->salle
.venv\Scripts\python IF23_Localisation\scripts\11_figures_alt3.py             # figures ALT3
.venv\Scripts\python IF23_Localisation\scripts\06_generate_report.py          # rapport Word
```

Modules : `src/robust_localization.py` (features), `src/eval_utils.py` (protocoles honnêtes),
`src/bayesian_localization.py` (modèle bayésien du cours).

## Données

Format brut : `BSSID, SSID, RSSI(dBm), Time`. Un **snapshot** = ensemble des AP
visibles à un instant donné, agrégé en un vecteur de features. 24 zones,
~1 498 snapshots, 529 features (campagne février 2026 : 10 et 13 fév).

## Avertissement méthodologique (ALT3)

Les scores « 99 % holdout / 100 % live » des livrables ALT2 reposaient sur une
**évaluation biaisée** : (1) split aléatoire de snapshots temporellement
autocorrélés et (2) « évaluation live » réalisée sur les données d'entraînement
elles-mêmes (in-sample). ALT3 fournit une estimation honnête via séparation
**temporelle** des données (holdout temporel + StratifiedGroupKFold sur blocs).
Voir `scripts/02_honest_evaluation.py` et le rapport.
