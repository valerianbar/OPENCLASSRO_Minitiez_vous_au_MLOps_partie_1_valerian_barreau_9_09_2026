# Initiez-vous au MLOps — Partie 1 · Home Credit Default Risk

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/package%20manager-uv-DE5FE9)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.7-9ACD32)
![XGBoost](https://img.shields.io/badge/XGBoost-3.4-EB0F00)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2-FFCC00)
![SHAP](https://img.shields.io/badge/SHAP-explainability-1F77B4)
![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow&logoColor=white)

Projet du parcours **OpenClassrooms — Initiez-vous au MLOps (Partie 1)**.
On construit une chaîne de *machine learning* complète sur les données
**[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)** :
de l'analyse exploratoire jusqu'au **suivi d'expériences (MLflow)**, à
l'**optimisation des hyperparamètres** et au **choix d'un seuil de décision métier**.

---

## 🎯 Problème

Prédire si un client remboursera son prêt ou aura des **difficultés de paiement**,
à partir de ses données de demande de crédit.

- **Type** : apprentissage supervisé, **classification binaire**
  (`TARGET = 0` remboursé · `TARGET = 1` défaut de paiement).
- **Métrique de classement** : **ROC AUC** (adaptée au fort déséquilibre ~92 % / 8 %).
- **Sortie** : une **probabilité** de défaut par client, convertie en décision via un **seuil**.

## 🧭 Démarche

Tout est dans le notebook
[`start-here-a-gentle-introduction.ipynb`](start-here-a-gentle-introduction.ipynb) :

1. **Analyse exploratoire (EDA)** — cible, valeurs manquantes, anomalies, corrélations.
2. **Préparation des données** — encodage (label / one-hot), imputation, mise à l'échelle, alignement train/test.
3. **Feature engineering** — variables polynomiales + features « métier » (ratios crédit/revenu, durée, ancienneté…).
4. **Comparaison de modèles** — protocole unifié (K-Fold, ROC AUC out-of-fold).
5. **Interprétabilité** — analyse **SHAP** (globale + locale).
6. **Suivi d'expériences** — journalisation des runs avec **MLflow**.
7. **Optimisation des hyperparamètres** — **GridSearchCV** sur les 3 modèles de boosting.
8. **Décision métier** — choix du **seuil** optimal sur la classe minoritaire (recall / coût).

## 🤖 Comparaison de modèles

Tous évalués avec le **même protocole** (K-Fold 5 plis, `random_state=50`, ROC AUC
out-of-fold, déséquilibre géré par pondération « balanced ») via une fonction
unifiée `run_cv_model('lgb' | 'xgb' | 'cat', ...)`.

| Modèle | Famille | ROC AUC OOF (indicatif) |
|---|---|---|
| Régression logistique | Linéaire | ≈ 0.67 |
| Random Forest | Bagging | ≈ 0.68 |
| **LightGBM** | Gradient boosting | ≈ 0.76 |
| **XGBoost** | Gradient boosting | ≈ 0.76 |
| **CatBoost** | Gradient boosting | ≈ 0.76 |

> Les 3 boosters sont au coude à coude ; CatBoost est légèrement en tête et
> surapprend le moins.

## 🔍 Interprétabilité (SHAP)

`TreeExplainer` décompose chaque prédiction en contributions additives :

- **Global** (bar / beeswarm) : `EXT_SOURCE_1/2/3` dominent largement, suivis des
  montants/durées de crédit et des variables d'âge/emploi.
- **Local** (waterfall) : explique *pourquoi* un dossier précis est jugé risqué —
  utile pour la transparence côté crédit.

## 📈 Suivi d'expériences (MLflow)

Chaque entraînement est journalisé dans MLflow (backend **SQLite**) :

- **Paramètres** : modèle, hyperparamètres, configuration de la validation croisée.
- **Métriques** : ROC AUC out-of-fold, ROC AUC train, écart de surapprentissage, AUC par pli.
- **Artefacts** : fichier de soumission, importances des features (CSV + graphique).

Lancer l'interface pour comparer les runs :

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
# puis ouvrir http://localhost:5000  (choisir l'experience "home_credit_default_risk")
```

> Astuce macOS : le port 5000 est parfois pris par AirPlay — utiliser `--port 5001` au besoin.

## 🛠️ Optimisation des hyperparamètres (GridSearchCV)

`GridSearchCV` (scoring `roc_auc`, CV stratifiée) appliqué aux **3 boosters**
via `tune_model('lgb' | 'xgb' | 'cat', ...)`, chaque meilleur réglage étant
loggé dans MLflow (`*_gridsearch`). La recherche tourne sur un **sous-échantillon**
pour rester rapide ; élargir les grilles ou `SUBSAMPLE` selon le temps disponible.

| Modèle | Hyperparamètres réglés |
|---|---|
| LightGBM | `num_leaves`, `learning_rate`, `n_estimators` |
| XGBoost | `max_depth`, `learning_rate`, `n_estimators` |
| CatBoost | `depth`, `learning_rate`, `iterations` |

## 🎚️ Choix du seuil de décision (métier)

Le ROC AUC évalue le classement, pas la décision. Sur les probabilités
**out-of-fold**, on choisit le seuil sur la classe minoritaire (`TARGET = 1`) :

- **Courbe Precision-Recall** et **Precision / Recall / F2 selon le seuil**.
- **Seuil F2-optimal** (favorise le recall).
- **Courbe de coût métier** : `COST_FN` (défaut manqué) vs `COST_FP` (bon client refusé)
  → **seuil coût-optimal**.
- **Tableau comparatif** : défauts détectés / manqués et bons clients refusés selon le seuil.

## 🗂️ Structure

```
.
├── start-here-a-gentle-introduction.ipynb   # notebook principal (EDA → modèles → SHAP → MLflow → tuning → seuil)
├── input/                                    # données Kaggle (non versionnées)
├── *.csv                                     # fichiers de soumission générés
├── mlflow.db / mlruns/                       # tracking MLflow (non versionnés)
├── pyproject.toml                            # dépendances (uv)
├── uv.lock
└── README.md
```

## ⚙️ Installation

Prérequis : **Python 3.13** et **[uv](https://docs.astral.sh/uv/)**.

```bash
uv sync
```

**Données** (non versionnées, volumineuses) : télécharger les fichiers depuis
[Home Credit Default Risk — Data](https://www.kaggle.com/c/home-credit-default-risk/data)
et les placer dans `input/` (`application_train.csv`, `application_test.csv`, …).

## ▶️ Utilisation

```bash
uv run jupyter lab      # ouvrir et exécuter le notebook de haut en bas
```

Les modèles produisent des fichiers de soumission au format Kaggle
(`baseline_lgb.csv`, `baseline_xgb.csv`, `baseline_catboost.csv`, …) et les runs
sont enregistrés dans MLflow.

## 🛣️ Roadmap

- [x] EDA, préparation des données et feature engineering
- [x] Baseline (régression logistique) et Random Forest
- [x] Comparaison LightGBM / XGBoost / CatBoost (protocole unifié)
- [x] Interprétabilité SHAP (globale + locale)
- [x] Suivi d'expériences avec MLflow (paramètres, métriques, artefacts)
- [x] Optimisation des hyperparamètres (GridSearchCV sur les 3 boosters)
- [x] Choix du seuil de décision (F2 / coût métier)
- [ ] Réentraîner les meilleurs hyperparamètres sur données complètes (runs `*_tuned`)
- [ ] *Model registry* MLflow — versionner et promouvoir le meilleur modèle
- [ ] Industrialisation : packaging du pipeline, API de prédiction

## 👤 Auteur

**Valérian Barreau** — parcours *AI Engineer* / OpenClassrooms.
