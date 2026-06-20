"""
bayesian_localization.py — Localisation bayésienne (modèle du cours IF23).

Implémente le classifieur bayésien de fingerprinting décrit en cours :
  P(zone | x) ∝ P(zone) · P(x | zone)
avec hypothèse d'indépendance conditionnelle des points d'accès (Naïve Bayes)
et vraisemblance gaussienne par AP :
  P(rssi_j | zone) = N(rssi_j ; µ_zj, σ_zj²)

Amélioration par rapport à un GaussianNB brut : la non-détection d'un AP est
modélisée explicitement par une loi de Bernoulli de présence (p_present),
plutôt que par une valeur de remplissage -100 qui crée des gaussiennes
dégénérées. La vraisemblance d'un AP absent vaut (1 - p_present), celle d'un
AP présent vaut p_present · N(rssi ; µ, σ²). Décision par maximum a posteriori.

Le modèle consomme la matrice de features de RobustFeatureBuilder en exploitant
les colonnes 'rssi_mean__<ap>' (RSSI) et 'presence__<ap>' (0/1).
"""
from __future__ import annotations

import re
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

_LOG2PI = np.log(2.0 * np.pi)


class GaussianBayesLocalizer(BaseEstimator, ClassifierMixin):
    def __init__(self, var_floor: float = 4.0, presence_smoothing: float = 1e-3,
                 fill_value: float = -100.0):
        self.var_floor = var_floor                # variance minimale (dBm²)
        self.presence_smoothing = presence_smoothing
        self.fill_value = fill_value

    # ------------------------------------------------------------------ utils
    def _split_columns(self, columns):
        rssi_cols, pres_cols, aps = {}, {}, []
        for c in columns:
            m = re.match(r"rssi_mean__(.+)", c)
            if m:
                rssi_cols[m.group(1)] = c; aps.append(m.group(1))
            m = re.match(r"presence__(.+)", c)
            if m:
                pres_cols[m.group(1)] = c
        aps = [a for a in aps if a in pres_cols]
        return aps, rssi_cols, pres_cols

    # ------------------------------------------------------------------ fit
    def fit(self, X, y):
        cols = list(X.columns)
        self.aps_, self.rssi_cols_, self.pres_cols_ = self._split_columns(cols)
        Xv = X
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n = len(y)

        rssi = np.column_stack([Xv[self.rssi_cols_[a]].to_numpy(float) for a in self.aps_])
        pres = np.column_stack([Xv[self.pres_cols_[a]].to_numpy(float) for a in self.aps_]) > 0.5

        self.priors_ = {}
        self.mu_ = {}; self.var_ = {}; self.ppres_ = {}
        for c in self.classes_:
            idx = (y == c)
            self.priors_[c] = idx.sum() / n
            p = pres[idx]                      # (n_c, n_ap) bool
            r = rssi[idx]
            ppres = p.mean(axis=0)
            ppres = np.clip(ppres, self.presence_smoothing, 1 - self.presence_smoothing)
            mu = np.empty(len(self.aps_)); var = np.empty(len(self.aps_))
            for j in range(len(self.aps_)):
                vals = r[p[:, j], j]
                if vals.size >= 2:
                    mu[j] = vals.mean(); var[j] = max(vals.var(), self.var_floor)
                elif vals.size == 1:
                    mu[j] = vals[0]; var[j] = self.var_floor * 4
                else:
                    mu[j] = self.fill_value; var[j] = self.var_floor * 25
            self.ppres_[c] = ppres; self.mu_[c] = mu; self.var_[c] = var
        return self

    # ------------------------------------------------------------------ predict
    def _log_posteriors(self, X):
        rssi = np.column_stack([X[self.rssi_cols_[a]].to_numpy(float) for a in self.aps_])
        pres = np.column_stack([X[self.pres_cols_[a]].to_numpy(float) for a in self.aps_]) > 0.5
        n = rssi.shape[0]
        out = np.empty((n, len(self.classes_)))
        for ci, c in enumerate(self.classes_):
            mu, var, pp = self.mu_[c], self.var_[c], self.ppres_[c]
            # AP présent : log p_present + log N(rssi; mu, var)
            log_gauss = -0.5 * (_LOG2PI + np.log(var) + (rssi - mu) ** 2 / var)
            log_present = np.log(pp) + log_gauss
            log_absent = np.log(1 - pp)
            contrib = np.where(pres, log_present, log_absent)
            out[:, ci] = np.log(self.priors_[c]) + contrib.sum(axis=1)
        return out

    def predict(self, X):
        lp = self._log_posteriors(X)
        return self.classes_[np.argmax(lp, axis=1)]

    def predict_proba(self, X):
        lp = self._log_posteriors(X)
        lp -= lp.max(axis=1, keepdims=True)
        p = np.exp(lp)
        return p / p.sum(axis=1, keepdims=True)
