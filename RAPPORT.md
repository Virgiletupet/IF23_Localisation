# IF23 P — Géolocalisation indoor par WiFi

**Auteur** : Virgile Tupet (binôme 6 — avec Maxime LE GAL et Molka GHADDAB)
**Encadrante** : Farah Chehade
**Période** : Février 2026 — Avril 2026
**Document généré** : 30 avril 2026

---

## Résumé exécutif

Ce projet vise à localiser un ordinateur portable en milieu indoor (bâtiments P et S de l'UTT) à partir des signaux WiFi captés. Deux problèmes complémentaires sont traités, conformément au cahier des charges :

1. **Classification multi-classe** : prédire la zone (salle ou couloir) où se trouve l'appareil.
2. **Régression intra-zone** : estimer les coordonnées (x, y) précises dans le repère d'une zone choisie.

Les modèles atteignent **100% de test accuracy** en classification (sur le hold-out des datasets ALT1 et ALT1+ALT2 combinés) et une **erreur médiane de 1.15 m** en régression sur la zone cartographiée. L'évolution temporelle ALT1 → ALT2 (3 mois d'écart) révèle une dégradation à **47.6%** sans ré-apprentissage, restaurée à **100%** avec seulement 50% des nouvelles données — démonstration claire du drift WiFi.

Le projet livre également deux interfaces graphiques temps réel (CustomTkinter) et un module Python réutilisable.

---

## 1. Contexte et objectifs

### 1.1 Contexte du projet

L'objectif pédagogique est de construire un système de localisation indoor **sans GPS**, en exploitant les signaux WiFi (RSSI : Received Signal Strength Indicator) captés par un ordinateur. À chaque position, l'ordinateur enregistre :

- les noms des réseaux (SSID),
- leurs adresses MAC (BSSID),
- la puissance des signaux (RSSI, en dBm).

Ces informations forment un vecteur caractéristique de la position. Ce vecteur sert d'entrée à des algorithmes d'IA (classification ou régression) qui apprennent à associer un signal à une localisation.

### 1.2 Spécifications du cahier des charges (PDF IF23 P)

| Exigence | Description |
|---|---|
| Classification | Prédire la zone parmi N (PDF : 10, nous : 18-26 selon découpage) |
| Régression | Estimer (x, y) dans une zone donnée |
| Vectorisation | `(RSSI BSSID₁, ..., RSSI BSSIDₙ)`, 0 ou -100 si BSSID non capté |
| Plusieurs algorithmes | Avec **toutes les features** puis **avec sélection** |
| Train/test split | Aléatoire |
| Évolution temporelle | Stratégies ALT1, ALT2, ALT3 obligatoires |
| Outils | Python, pywifi, pandas/openpyxl |

---

## 2. Timeline du projet

| Période | Phase | Livrables |
|---|---|---|
| Fév 2026 (10-13/02) | **ALT1 — Collecte initiale** | 27 fichiers CSV, ~1500 snapshots, 24 sous-zones |
| Fév 2026 (mi-fév) | Premiers prototypes | 2 notebooks legacy (collecte GUI, premier RF) |
| Fév 2026 | Pipeline robuste v1 | Module `robust_localization.py`, modèles SSID-only et BSSID legacy (98% / 99%) |
| Avr 2026 (28/04) | Restructuration | Arbo `data/raw/`, `notebooks/`, `src/`, `app/` |
| Avr 2026 (30/04) | **Modèle v2 (compound)** | Encoding SSID\|BSSID, 100% test accuracy, GUI live moderne |
| Avr 2026 (30/04) | **ALT2 — Re-mesures** | 10 fichiers, 3 nouvelles salles (P104, P204, S201) |
| Avr 2026 (30/04) | **Régression intra-zone** | Dataset 46 points × 52 BSSID, MAE 1.19 m |
| Avr 2026 (30/04) | **Stratégies ALT1→ALT2** | Notebook 11, drift quantifié |

---

## 3. Datasets

### 3.1 ALT1 — campagne initiale (février 2026)

**Localisation** : bâtiments P et S, 1er et 2nd étages.

| # zones (sous-découpage A/B) | Snapshots totaux | Format |
|---|---|---|
| 27 | ~1500 | CSV `wifi_<zone>.csv` avec colonnes `BSSID, SSID, RSSI(dBm), Time` |

**Sous-zones par fichier** :
- Salles : `P101A/B`, `P102A/B`, `P103A`, `S101A/B`, `S102A/B`, `S103B`, `S104`, `S202/A/B`, `S203A/B`, `S204A/B`, `P202`, `P203`
- Couloirs : `couloirB`, `couloirB(2e)`, `couloirPbasA/B`
- Paliers : `PALIER1ER`, `PALIER2E`, `pallierPbas`

Chaque salle a été divisée en 2 sous-zones (A à gauche, B à droite) pour augmenter la résolution spatiale.

### 3.2 ALT2 — re-mesures (avril 2026)

**Différence de protocole** : capture continue de la salle entière (~3 min, 50-100 snapshots/min) au lieu du découpage A/B.

| Salle ALT2 | Présent en ALT1 ? | Mapping ALT1 fusionné |
|---|---|---|
| P102 | oui (P102A + P102B) | P102 |
| P103 | oui (P103A) | P103 |
| **P104** | **non, nouvelle** | P104 |
| P202 | oui | P202 |
| P203 | oui | P203 |
| **P204** | **non, nouvelle** | P204 |
| S102 | oui (S102A + S102B) | S102 |
| S103 | oui (S103B) | S103 |
| **S201** | **non, nouvelle** | S201 |
| S203 | oui (S203A + S203B) | S203 |

Total : 10 fichiers, ~452 snapshots après agrégation, 7 zones communes avec ALT1, 3 nouvelles.

> **Anecdote** : le fichier `wifi_Alt2P204.csv` initial contenait en réalité deux salles (P204 puis P202) capturées à la suite sans changement de fichier. Détection d'un gap temporel de 82 s à 15:27:52, split automatique en deux fichiers `wifi_P204.csv` (411 lignes, 15:24-15:26) et `wifi_P202.csv` (367 lignes, 15:27-15:29).

### 3.3 Dataset régression intra-zone

**Format** : conforme au PDF page 5 : `X, Y, BSSID_1, ..., BSSID_N` avec une ligne par point de mesure.

| Caractéristique | Valeur |
|---|---|
| Points de mesure | 46 |
| BSSID référencés | 52 |
| Plage X | [-1.0, 4.0] m |
| Plage Y | [0.0, 7.0] m |
| Surface zone | ~5 × 7 m |

Chaque ligne = une position (x, y) avec le RSSI moyenné de chaque BSSID capté à cette position (0 si non capté), exactement comme spécifié page 5 du PDF.

### 3.4 Données de référence pour distance (RSSI0.xlsx)

Un scan unique de référence à proximité d'un AP, 73 BSSIDs avec leur RSSI à distance d₀ ≈ 1 m. Sert à la calibration du modèle log-distance (bonus, hors PDF).

---

## 4. Méthodologie

### 4.1 Vectorisation des features

Le défi central : transformer un scan WiFi (liste de tuples BSSID/SSID/RSSI) en un **vecteur de features de taille fixe**, identique pour tous les snapshots.

#### Pipeline `RobustFeatureBuilder`

Pour chaque snapshot (un timestamp), on agrège par identifiant réseau :
- `rssi_mean__<id>`, `rssi_max__<id>`, `rssi_std__<id>`, `rssi_count__<id>`, `presence__<id>` : 5 stats par identifiant
- 9 features de **résumé global** : `visible_ssid_count`, `rssi_max_overall`, `rssi_mean_overall`, `rssi_min_overall`, `rssi_std_overall`, `rssi_median_overall`, `rssi_p25_overall`, `rssi_p75_overall`, `strongest_gap`

Les valeurs RSSI manquantes sont remplies par **-100 dBm** (signal indétectable) — conforme au PDF page 4.

#### Choix de l'identifiant réseau

| Choix | Vocabulaire | Test accuracy | Pourquoi |
|---|---|---|---|
| **SSID seul** (`UTTetudiants`, `eduroam`...) | 13 | 98.3% | Trop peu d'identifiants — tous les AP UTT partagent les mêmes SSID |
| **BSSID seul** (`84:b2:61:1f:b5:37`) | 104 | 99.0% | Discriminant mais pas de contexte SSID |
| **Compound `SSID\|BSSID`** ⭐ | 139 | **100.0%** | Maximum d'information, BSSID multiplexé dans le pipeline SSID |

L'astuce du compound `SSID|BSSID` permet de réutiliser le pipeline SSID existant tout en bénéficiant de la richesse des BSSIDs. C'est la clé pour atteindre 100% en hold-out.

### 4.2 Algorithmes testés (classification)

Conformément au PDF (page 6 : "*plusieurs modèles d'IA, avec tous les réseaux puis une sélection*"), le notebook 07 évalue 10 algorithmes :

| Algo | CV accuracy | Test accuracy |
|---|---|---|
| **VotingClassifier** (RF + ET + LogReg, soft) | 0.9945 | **1.0000** |
| ExtraTrees (n=300, balanced) | 0.9929 | 0.9969 |
| RandomForest (n=200, balanced_subsample) | 0.9921 | 0.9969 |
| LogReg L2 (C=1, balanced) | 0.9921 | 0.9969 |
| HistGradientBoosting (lr=0.08) | 0.9882 | 0.9969 |
| MLP_64_32 + MLP_128_64 | ~0.99 | ~0.997 |
| KNN_3 + KNN_5 (distance weighting) | ~0.99 | ~0.997 |

Le **Voting** combine les forces de RF (capture les seuils RSSI), ExtraTrees (réduit la variance) et LogReg (frontières linéaires sur l'espace BSSID).

### 4.3 Algorithmes testés (régression intra-zone)

Notebook 10 : 8 régresseurs multi-output (entrée vecteur RSSI, sortie (x, y)).

| Algo | MAE_x (m) | MAE_y (m) | MAE euclid. (m) | Médiane (m) |
|---|---|---|---|---|
| **ExtraTrees** (n=400) | **0.92** | **0.65** | **1.19** | **1.15** |
| RandomForest (n=300) | 0.79 | 0.93 | 1.31 | 1.15 |
| KNN_3 distance-weighted | 1.30 | 0.60 | 1.48 | 1.39 |
| GradientBoosting (n=200) | 1.21 | 0.60 | 1.48 | 1.47 |
| Ridge | 1.10 | 1.16 | 1.79 | 1.74 |
| MLP_64_32 | 1.17 | 1.32 | 1.84 | 1.58 |

Avec le **best ExtraTrees** : P90 d'erreur = 2.18 m, max = 2.83 m sur une zone 5×7 m.

#### Sélection de BSSID (PDF "puis une sélection")

Test avec k ∈ {5, 10, 15, 20, 30, 52} BSSIDs sélectionnés par feature_importance d'un RF. Sur ce dataset (46 points), **toutes les 52 features** apportent de l'information : aucun gain à élaguer. À reconsidérer si le dataset s'agrandit.

---

## 5. Évolution temporelle ALT1 → ALT2 (PDF page 7)

### 5.1 Stratégies obligatoires

| Stratégie | Train | Test | Objectif |
|---|---|---|---|
| **1** | ALT1 seul | ALT2 (zones communes) | Mesurer le **drift pur** sans ré-apprentissage |
| **2** | ALT1 + 50% ALT2 | 50% restant ALT2 | Quantifier le **gain** d'un ré-entraînement partiel |
| 3 (futur) | ALT1 + ALT2 | ALT3 | Évaluer la stabilité à 6 mois |

### 5.2 Résultats

#### Stratégie 1 — train ALT1, test ALT2

| Zone | n test | Accuracy |
|---|---|---|
| **P202** | 26 | **100.0%** |
| S103 | 51 | 82.4% |
| S102 | 59 | 81.4% |
| P102 | 54 | 79.6% |
| **P103** | 57 | **0.0%** |
| **P203** | 47 | **0.0%** |
| **S203** | 40 | **0.0%** |

**Accuracy globale : 47.6%** (F1 macro = 0.33).

3 zones tombent à zéro → drift WiFi fort. Cause vraisemblable : déplacement d'AP, changement de canaux, nouvelles bornes invitées pour les TP, ou différence de protocole de capture (continu vs grille discrète).

#### Stratégie 2 — train ALT1 + 50% ALT2

**Accuracy globale : 100.0%** sur les 10 zones (y compris les 3 nouvelles P104, P204, S201).

Le contraste **47.6% → 100%** quantifie précisément le drift : il y a une dégradation réelle entre Feb et Avr, mais l'apport de seulement 50% des données récentes la compense intégralement.

### 5.3 Interprétation

Pour un déploiement industriel, ce résultat justifie un protocole de ré-apprentissage régulier (ex. mensuel). C'est le résultat-clef que le PDF cherche à mettre en évidence.

---

## 6. Architecture du code

```
IF23_Localisation/
├── data/
│   ├── raw/                  ALT1 (27 CSV, ~1500 snapshots)
│   ├── raw_alt2/             ALT2 (10 CSV, ~452 snapshots)
│   ├── regression/           dataset_regression.csv (46 points × 52 BSSID)
│   ├── cleaned/              dataset_unified pour pipeline legacy
│   └── exports/              wifi_data.csv, wifi_measurements.json, RSSI0.xlsx
├── notebooks/
│   ├── 00_legacy_*           prototypes initiaux (collecte GUI)
│   ├── 01_data_cleaning      nettoyage + dataset_unified
│   ├── 02_model_training     pipeline legacy (RF + NN)
│   ├── 04_robust_training    pipeline robuste, model zoo, CV+holdout
│   ├── 07_advanced_classif.  encoding compound SSID|BSSID, 10 algos, 100%
│   ├── 08_models_evolution   v1 → v3, heatmap par salle
│   ├── 09_distance_regression bonus log-distance
│   ├── 10_intra_zone_regression  (x, y), all-features vs sélection
│   └── 11_alt1_alt2_evolution   stratégies PDF page 7
├── src/
│   ├── robust_localization.py    feature builder + model zoo
│   ├── intra_zone_regression.py  IntraZoneRegressor (PDF page 5)
│   └── distance_estimator.py     LogDistanceEstimator (bonus)
├── models/
│   ├── artifacts_combined/   ⭐ ALT1+ALT2, 21 zones, 100%
│   ├── artifacts_alt1/       ALT1 fusionné, 18 zones, 100%
│   ├── artifacts_alt2/       ALT2 récent, 10 zones, 100%
│   ├── artifacts_v2/         ALT1 sous-zones A/B, 26 classes, 100%
│   ├── artifacts_robust*/    legacy
│   ├── artifacts_regression/ IntraZoneRegressor, MAE 1.19 m
│   └── artifacts_distance/   LogDistanceEstimator (bonus)
└── app/
    ├── live_app.py           classification de salle live
    ├── position_app.py       position (x, y) intra-zone live
    └── distance_app.py       estimation distance à un AP (bonus)
```

---

## 7. Interfaces graphiques

### 7.1 `live_app.py` — Classification de salle en direct

**Fonctionnalités** :
- Sélection du modèle (6 disponibles, recommandé : `Combined ALT1+ALT2`)
- Slider d'intervalle de scan (1-8 s)
- Détection automatique du format de features (SSID / BSSID / compound) selon le modèle chargé
- Affichage : salle prédite (police 52 pt, couleur conditionnelle), barre de confiance, top-5 prédictions
- **Vérité terrain** : sélectionner la salle réelle → indicateur OK/KO + accuracy live cumulée
- Liste scrollable des SSID visibles avec RSSI coloré
- Bouton **Snapshot CSV** pour sauvegarder un scan au format compatible training

### 7.2 `position_app.py` — Position (x, y) en direct

**Fonctionnalités** :
- Carte 2D matplotlib embarquée affichant la zone et le point prédit
- Coordonnées en gros (police 48 pt)
- Indicateur de couverture : nombre de BSSID connus du modèle effectivement captés
- Liste BSSID visibles avec ✓ pour ceux reconnus du modèle

### 7.3 `distance_app.py` — Distance à un AP (bonus)

Estimation log-distance avec slider `n` (path loss exponent), bouton de calibration sur distance vraie. Pas dans le PDF mais utile pour le diagnostic.

---

## 8. Visualisations produites

| Fichier | Contenu |
|---|---|
| `models/artifacts_v2/confusion_matrix_v2.png` | Matrice de confusion (26×26) du best Voting sur le hold-out — quasi-diagonale parfaite |
| `models/artifacts_robust/confusion_matrix_robust.png` | Confusion legacy ExtraTrees |
| `models/artifacts_robust_ssid/confusion_matrix_robust_ssid.png` | Confusion SSID-only |
| `models/artifacts_regression/prediction_scatter.png` | Scatter plot vrai vs prédit (x, y) sur le hold-out, lignes d'erreur tracées |
| Notebook 08 (heatmap inline) | Accuracy par salle × génération de modèle |
| Notebook 11 (heatmap inline) | Accuracy par zone × stratégie d'évolution (S1 vs S2) |
| Notebook 10 (graphique inline) | Effet du nombre de BSSID sélectionnés sur la MAE |
| Notebook 09 (graphique inline) | Distance vs RSSI pour différents `n` (modèle log-distance) |

---

## 9. Problèmes rencontrés et solutions

### 9.1 Mismatch d'encoding entre training et inference live

**Symptôme** : la GUI live donnait des prédictions aléatoires sur les modèles `artifacts_v2` et `artifacts_combined`.

**Cause** : `WiFiScanner.scan_once_by_ssid()` produisait un dict `{ssid: rssi}` (13 SSID nus) tandis que les modèles modernes attendent un dict `{ssid|bssid: rssi}` (139+ compounds). Tout le vecteur arrivait à -100 → prédiction aléatoire.

**Correction** : ajout de `detect_feature_format(feature_builder)` qui inspecte les clés de `selected_ssids` :
- contient `|` → format `compound`
- ressemble à une MAC → format `bssid`
- sinon → format `ssid`

Le scanner produit maintenant le bon format selon le modèle chargé. Validation : prédictions à >85% confiance sur tous les CSV de référence.

### 9.2 Drift de l'infrastructure WiFi

**Observation finale (30/04/2026, scan en salle inconnue)** :

```
Live : 71 BSSID captés
ALT1 : 104 BSSID dans le training
ALT2 : 117 BSSID dans le training
Live ∩ (ALT1 ∪ ALT2) : 0 BSSIDs
```

**0** des BSSIDs visibles actuellement n'est dans les datasets. Trois explications possibles :

1. **L'utilisateur teste depuis une zone non cartographiée** (autre bâtiment, étage non mesuré, ou hors UTT). C'est la cause la plus probable.
2. **L'infrastructure WiFi de l'UTT a été remplacée** entre fév-avr et maintenant. Les MAC visibles (`e4:aa:5d:9f:7d:e*`, `74:88:bb:03:cc:*`) sont toutes différentes de celles enregistrées (`e4:aa:5d:bf:25:*`, `e4:aa:5d:fd:3e:*`, `f8:6b:d9:01:*`). Les vendeurs (HPE) sont identiques mais les MAC complètes sont différentes → AP physiques distincts.
3. **L'antenne WiFi du PC capte différemment** (faux positif peu probable).

**Conséquence** : les modèles ne peuvent intrinsèquement rien prédire de fiable dans une zone où aucun BSSID de référence n'est visible. C'est une limitation fondamentale du fingerprinting WiFi : un modèle ne reconnaît que les "empreintes" qu'il a déjà vues.

**Recommandation** : pour tester la GUI live, se placer **physiquement dans une salle cartographiée** parmi {P101, P102, P103, P104, P202, P203, P204, S101, S102, S103, S104, S201, S202, S203, S204, PALIER1ER, PALIER2E, couloirB, couloirB(2e), couloirPbas, pallierPbas}. Si le problème persiste, c'est que l'infra a été modifiée et il faut **recapturer le dataset**.

### 9.3 Conflits de versions sklearn

Les artefacts legacy (`artifacts_robust`, `artifacts_robust_ssid`) ont été produits avec scikit-learn 1.6.1 mais le venv actuel utilise 1.5/1.8. Avertissement de version à l'unpickling, sans impact fonctionnel.

### 9.4 Tailles de classes inégales pour la CV stratifiée

`S202` (3 lignes) a été automatiquement filtrée car incompatible avec la CV 5-fold stratifiée. Les autres classes ont entre 32 et 97 snapshots, suffisant pour 5-fold.

---

## 10. Conclusions et perspectives

### 10.1 Acquis

- **Pipeline complet** vectorisation + 10 algos + sélection + évaluation conforme au PDF.
- **100% accuracy** en classification sur le hold-out (zones cartographiées et conditions de capture stables).
- **MAE 1.19 m** en régression intra-zone sur 5×7 m.
- **Stratégies d'évolution** quantifiant le drift WiFi (47.6% → 100% avec ré-apprentissage).
- **Interfaces temps réel** opérationnelles pour les deux problèmes.
- **Module Python réutilisable** pour intégrer le pipeline dans un autre projet.

### 10.2 Limites

- Le fingerprinting WiFi est intrinsèquement **non extrapolable** : on ne peut prédire que les zones cartographiées.
- Le drift WiFi (changement d'AP, infra) **invalide rapidement** un modèle entraîné. Nécessite un protocole de re-collecte régulier.
- Le dataset de régression couvre une seule zone — extension nécessaire pour généraliser.

### 10.3 Travaux futurs (Stratégie 3 — ALT3)

- Capturer ALT3 sur les mêmes zones pour mesurer la stabilité à 6 mois.
- Modèle hiérarchique (P/S → étage → zone) pour gestion d'erreur graceful (PDF page 8 — optionnelle).
- Exploitation de la mobilité (graphe de transitions) pour filtrer les sauts impossibles (PDF page 8 + A3).
- Étendre le dataset régression à plus de zones et plus de points.

---

## Annexe A — Bibliothèques utilisées

- **numpy, pandas** : manipulation de données
- **scikit-learn** : tous les modèles ML (RF, ET, LR, KNN, MLP, GBR, Voting, Stacking)
- **joblib** : sérialisation des modèles
- **matplotlib, seaborn** : visualisations
- **pywifi** : scan WiFi temps réel (Windows)
- **customtkinter** : interfaces graphiques modernes
- **openpyxl** : lecture Excel (RSSI0.xlsx)
- **jupyter, nbclient, nbformat** : exécution programmée des notebooks

## Annexe B — Comment lancer

```bash
# Installation
pip install -r requirements.txt

# Entraînement / re-entraînement
jupyter lab notebooks/07_advanced_classification.ipynb

# Évolution temporelle
jupyter lab notebooks/11_alt1_alt2_evolution.ipynb

# Régression intra-zone
jupyter lab notebooks/10_intra_zone_regression.ipynb

# GUI live de classification de salle
python app/live_app.py

# GUI live de position (x, y) intra-zone
python app/position_app.py
```
