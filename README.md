# IF23 - Localisation indoor par scan WiFi

Classification de la salle dans laquelle se trouve un appareil à partir des SSID/BSSID/RSSI captés autour de lui.

## Structure

```
IF23_Localisation/
├── data/
│   ├── raw/            # Mesures historiques (Feb), une CSV par salle
│   ├── raw_alt2/       # Mesures alternatives (Avr), recapture pour P102/P103/P104/P202/P203/P204
│   ├── cleaned/        # Datasets nettoyés (issus de notebook 01)
│   └── exports/        # wifi_data.csv, wifi_measurements.json, RSSI0.xlsx
├── notebooks/
│   ├── 00_legacy_*.ipynb               # Anciens prototypes (collecte GUI)
│   ├── 01_data_cleaning.ipynb          # Nettoyage + dataset_unified.csv
│   ├── 02_model_training.ipynb         # Pipeline legacy (RF + NN)
│   ├── 03_live_prediction.ipynb        # Prédiction live legacy
│   ├── 04_robust_training.ipynb        # Pipeline robuste (vectorisation SSID)
│   ├── 05_robust_prediction.ipynb      # Évaluation robuste
│   ├── 06_live_gui_notebook.ipynb      # GUI Tkinter d'origine (basique)
│   ├── 07_advanced_classification.ipynb  # 10 algos, encoding SSID|BSSID, >99%
│   ├── 08_models_evolution.ipynb       # Évolution v1→v3, heatmap par salle
│   ├── 09_distance_regression.ipynb    # Estimation distance log-distance (bonus)
│   └── 10_intra_zone_regression.ipynb  # Régression (x,y) intra-zone (PDF page 5)
├── data/
│   └── regression/
│       └── dataset_regression.csv      # X, Y + RSSI par BSSID (46 points)
├── src/
│   ├── robust_localization.py          # Feature builder classification
│   ├── distance_estimator.py           # LogDistanceEstimator (bonus)
│   └── intra_zone_regression.py        # IntraZoneRegressor (PDF conforme)
├── models/
│   ├── artifacts_combined/             # **ALT1+ALT2, 21 zones, recommandé pour live, 100%**
│   ├── artifacts_alt1/                 # ALT1 zones fusionnées, 18 zones, 100%
│   ├── artifacts_alt2/                 # ALT2 récent, 10 zones, 100%
│   ├── artifacts_v2/                   # ALT1 sous-zones A/B, 26 classes, 100%
│   ├── artifacts_robust/               # legacy BSSID, ExtraTrees ~99%
│   ├── artifacts_robust_ssid/          # SSID-only, débogage uniquement
│   ├── artifacts_distance/             # LogDistanceEstimator (bonus)
│   ├── artifacts_regression/           # IntraZoneRegressor (x,y), MAE 1.19m
│   └── legacy/                         # Anciens .pkl
├── app/
│   ├── wifi_scan.py                    # Scanner partagé (pywifi)
│   ├── live_app.py                     # GUI classification de salle (live)
│   ├── distance_app.py                 # GUI estimation distance à un AP (bonus)
│   └── position_app.py                 # GUI position (x,y) intra-zone (PDF page 5)
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### Entraîner / réentraîner les modèles
```bash
cd notebooks/
jupyter lab 04_robust_training.ipynb
```

### Lancer la GUI live (classification de salle)
```bash
cd app/
python live_app.py
```
La GUI scanne le WiFi en continu et affiche en temps réel la salle prédite + top-5 + confiance.

### Lancer la GUI position (x, y) intra-zone — PDF page 5
```bash
cd app/
python position_app.py
```
Affiche la position prédite (x, y) dans le repère de la zone, sur une carte 2D, avec la liste des BSSID visibles (ceux connus du modèle marqués ✓).

### Lancer la GUI distance — bonus
```bash
cd app/
python distance_app.py
```
Estimation log-distance de la distance à un AP cible. Pas dans le PDF, à utiliser en complément.

## Performances des modèles

| Version | Source | Classes | Features | Test acc | Best model |
|---------|--------|---------|----------|----------|------------|
| v1 (legacy) | `models/legacy/` | 16 | 6 BSSID | 73.1% | RF custom |
| v2a (robust SSID) | `models/artifacts_robust_ssid/` | 24 | 74 | 98.3% | RandomForest |
| v2b (robust full) | `models/artifacts_robust/` | 24 | 529 | 99.0% | ExtraTrees |
| **v3 (advanced)** | `models/artifacts_v2/` | **26** | **704** | **100.0%** | **VotingClassifier (RF+ET+LR)** |

L'astuce v3 : remplacer le SSID brut (~13 valeurs UTTetudiants/eduroam/etc partagées entre tous les AP) par un identifiant composé `SSID|BSSID` qui distingue chaque AP physique → ~140 features discriminantes par scan.

## Pièces classées (24)
P101A/B, P102A/B, P103A, S101A/B, S102A/B, S103B, S104, S202A/B, S203A/B, S204A/B,
PALIER1ER/2E, couloirB, couloirB(2e), couloirPbasA/B, pallierPbas

## Notes
- Les CSV `raw_alt2/wifi_P204.csv` et `raw_alt2/wifi_P202.csv` proviennent du split d'un fichier mal étiqueté (un seul scan continu sur deux salles, coupé à 15:27:52 le 28/04/2026).
- Pour ajouter de nouvelles salles, déposer `wifi_<salle>.csv` dans `data/raw/` puis rejouer le notebook 04.
