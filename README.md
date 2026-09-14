# Initiez-vous au MLOps — Partie 1 · Home Credit Default Risk

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/package%20manager-uv-DE5FE9)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.7-9ACD32)
![XGBoost](https://img.shields.io/badge/XGBoost-3.4-EB0F00)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2-FFCC00)
![SHAP](https://img.shields.io/badge/SHAP-explainability-1F77B4)

Projet du parcours **OpenClassrooms — Initiez-vous au MLOps (Partie 1)**.
On y construit une première brique de modélisation *machine learning* sur les
données **[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)**,
avant d'y ajouter le suivi d'expériences avec **MLflow** dans la suite du parcours.

---

## 🎯 Problème

Prédire si un client remboursera son prêt ou aura des **difficultés de paiement**,
à partir de ses données de demande de crédit.

- **Type** : apprentissage supervisé, **classification binaire**
  (`TARGET = 0` remboursé · `TARGET = 1` défaut de paiement).
- **Métrique** : **ROC AUC** (adaptée au fort déséquilibre des classes ~92 % / 8 %).
- **Sortie** : une **probabilité** de défaut par client (et non une décision 0/1).

## 🧭 Démarche

Le notebook [`start-here-a-gentle-introduction.ipynb`](start-here-a-gentle-introduction.ipynb)
déroule un projet ML complet :

1. **Analyse exploratoire (EDA)** — distribution de la cible, valeurs manquantes,
   détection d'anomalies (ex. `DAYS_EMPLOYED = 365243`), corrélations.
2. **Préparation des données** — encodage des variables catégorielles
   (label / one-hot), imputation, mise à l'échelle, alignement train/test.
3. **Feature engineering** — variables polynomiales + features « métier »
   (ratios crédit/revenu, durée du prêt, ancienneté relative…).
4. **Modélisation & comparaison** — plusieurs modèles évalués selon le **même
   protocole** (validation croisée K-Fold, ROC AUC out-of-fold).
5. **Interprétabilité** — analyse **SHAP** (globale + locale).

## 🤖 Modèles comparés

Tous évalués avec le même protocole (K-Fold 5 plis, `random_state=50`, ROC AUC
out-of-fold, gestion du déséquilibre par pondération « balanced »).

| Modèle | Famille | ROC AUC (indicatif) |
|---|---|---|
| Régression logistique | Linéaire | ≈ 0.67 |
| Random Forest | Bagging | ≈ 0.68 |
| **LightGBM** | Gradient boosting | ≈ 0.74 |
| **XGBoost** | Gradient boosting | ≈ 0.74 |
| **CatBoost** | Gradient boosting | ≈ 0.74 |

> Les scores de boosting montent encore avec les features « métier » (≈ 0.75).
> Fonction unifiée `run_cv_model('lgb' | 'xgb' | 'cat', ...)` pour une comparaison équitable.

## 🔍 Interprétabilité (SHAP)

`TreeExplainer` décompose chaque prédiction en contributions additives par feature :

- **Global** (bar / beeswarm) : les scores externes `EXT_SOURCE_1/2/3` dominent
  largement, suivis des montants et durées de crédit et des variables d'âge/emploi.
- **Local** (waterfall) : explique *pourquoi* un dossier précis est jugé risqué —
  utile pour la transparence et la conformité côté crédit.

## 🗂️ Structure

```
.
├── start-here-a-gentle-introduction.ipynb   # notebook principal (EDA → modèles → SHAP)
├── input/                                    # données Kaggle (non versionnées)
├── *.csv                                     # fichiers de soumission générés
├── pyproject.toml                            # dépendances (uv)
├── uv.lock
└── README.md
```

## ⚙️ Installation

Prérequis : **Python 3.13** et **[uv](https://docs.astral.sh/uv/)**.

```bash
# Cloner puis installer les dépendances
uv sync
```

**Données** : elles ne sont pas versionnées (volumineuses). Télécharger les fichiers
depuis la page Kaggle [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data)
et les placer dans `input/` (`application_train.csv`, `application_test.csv`, …).

## ▶️ Utilisation

```bash
# Lancer Jupyter et ouvrir le notebook
uv run jupyter lab
```

Exécuter les cellules de haut en bas. Les modèles produisent des fichiers de
soumission au format Kaggle (`baseline_lgb.csv`, `baseline_xgb.csv`,
`baseline_catboost.csv`, …).

## 🛣️ Roadmap

- [x] EDA, préparation des données et feature engineering
- [x] Baseline (régression logistique) et Random Forest
- [x] Comparaison LightGBM / XGBoost / CatBoost (protocole unifié)
- [x] Interprétabilité SHAP (globale + locale)
- [ ] **Suivi d'expériences avec MLflow** — journalisation des paramètres,
      métriques (ROC AUC) et artefacts ; comparaison des runs ; *model registry*
- [ ] Optimisation du **seuil de décision** (recall / coût métier)
- [ ] Industrialisation : packaging du pipeline, API de prédiction

## 👤 Auteur

**Valérian Barreau** — parcours *AI Engineer* / OpenClassrooms.
