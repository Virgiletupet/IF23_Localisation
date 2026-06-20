# Guide d'exécution — IF23 Localisation indoor WiFi

Comment lancer le projet sur ta machine (Windows). Deux usages :
**A.** l'analyse / les notebooks (sans WiFi), **B.** la version **live** (avec WiFi).

---

## 0. Prérequis
- Windows, **Python 3.13** (ou 3.10/3.11). Vérifier : `py --list`
- Pour la version live : un **PC avec carte WiFi activée** (pas en mode avion).

## 1. Installation (une seule fois)

Ouvre **PowerShell** dans le dossier du projet :

```powershell
cd "C:\Users\virgi\Documents\UTT\25-26\if23\projet\IF23_Localisation"
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install scikit-learn==1.8.0     # IMPORTANT : version des modèles
```

> Si `Activate.ps1` est bloqué (politique d'exécution) :
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` puis relance l'activation.
> (ou ignore l'activation et préfixe chaque commande par `.\.venv\Scripts\python.exe`)

## 2. Vérifier que tout fonctionne (sans WiFi)

```powershell
python verifier_setup.py
```

Cela charge le modèle et prédit sur un fichier de mesures réel : si une accuracy
élevée s'affiche, l'environnement et les modèles sont OK.

## 3. Version LIVE (avec WiFi) — les 3 interfaces

```powershell
.\.venv\Scripts\Activate.ps1        # si pas déjà actif
python app\live_app.py              # classification de salle en temps réel
python app\position_app.py          # position (x, y) intra-zone
python app\distance_app.py          # distance à une borne (bonus)
```

Dans **live_app** : choisir le modèle « **Combined ALT1+ALT2 (recommandé)** »,
régler l'intervalle de scan, et (optionnel) indiquer la vraie salle pour suivre
l'accuracy en direct.

## 4. Analyse / notebooks (sans WiFi)

```powershell
python -m jupyter lab
```
Ouvrir par exemple : `notebooks/07_advanced_classification.ipynb` (classification),
`notebooks/10_intra_zone_regression.ipynb` (position x,y),
`notebooks/11_alt1_alt2_evolution.ipynb` (évolution temporelle).

---

## Conseils & dépannage
- **Se placer dans une salle cartographiée** (bâtiments S/P, étages 1-2). Ailleurs,
  aucune borne connue n'est visible → la prédiction n'a pas de sens.
- **Scan lent** : Windows limite la fréquence des scans WiFi (≈ un toutes les
  quelques secondes), c'est normal.
- **pywifi : pas d'interface trouvée** → vérifier que le WiFi est activé ; au besoin
  lancer PowerShell **en administrateur**.
- **Erreur `multi_class` / dépicklage** → mauvaise version de scikit-learn :
  `python -m pip install scikit-learn==1.8.0`.
- **Drift** : si l'infrastructure WiFi de l'UTT a changé depuis les collectes, les
  empreintes peuvent ne plus correspondre → il faut recollecter (campagne ALT3).
```
