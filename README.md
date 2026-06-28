# Pipeline de préparation des données — Finova Credit Scoring

## Installation

1. Créer un environnement conda :
```bash
conda create -n Task4 python=3.12
conda activate Task4
```

2. Installer les dépendances :
```bash
pip install pandas scikit-learn jupyterlab
```

## Utilisation

**Pipeline standard (sans validation) :**
```bash
python pipeline_corrige.py finova_credits.db
```

**Pipeline avec contrôles qualité automatisés (bonus) :** correspond à une version orientée production du pipeline. Contrairement au pipeline principal, il applique des contrôles qualité bloquants en entrée. Si des violations sont détectées, le lot est rejeté et un rapport rapport_validation.csv est généré pour permettre l’analyse et la correction des données sources.
```bash
python pipeline_corrige_bonus.py finova_credits.db
```

Les jeux de données nettoyés sont exportés dans `finova_credits_clean.db` 
avec deux tables : `train` (788 dossiers) et `test` (198 dossiers).

## Défauts identifiés dans pipeline_v1.py

| # | Fonction | Défaut | Impact |
|---|----------|--------|--------|
| 1 | `prepare_datasets()` | Data leakage — StandardScaler fitté avant le split | Scores artificiellement optimistes |
| 2 | `fix_amounts()` | Valeurs comme "2 442 EUR" perdues silencieusement | 25 valeurs récupérables perdues |
| 3 | `clean_data()` | `dropna()` supprime 128 dossiers pour 2 colonnes non obligatoires | Perte de 12.5% des données |
| 4 | `encode_features()` | `LabelEncoder` assigne un ordre alphabétique arbitraire | Biais dans l'encodage des variables ordinales |
| 5 | `encode_features()` | `travailleur_etranger` — 5 variantes pour 2 valeurs | Bruit dans le modèle |
| 6 | `prepare_datasets()` | Absence de `random_state` | Résultats non reproductibles |
| 7 | `main()` | Accuracy seule comme métrique de validation | Insuffisant pour un scoring crédit déséquilibré |

## Corrections apportées

| Fonction | Correction |
|----------|------------|
| `fix_amounts()` | Parsing du texte avant conversion — récupère 25 valeurs supplémentaires. Imputation par la médiane au lieu de la moyenne |
| `clean_data()` | Remplacement de `dropna()` par imputation contextuelle. Suppression des doublons exacts et masqués |
| `remove_outliers()` | Nouvelle fonction — suppression des valeurs aberrantes selon règles métier et légales |
| `encode_features()` | `OrdinalEncoder` avec ordre métier explicite pour 5 colonnes. One-Hot Encoding pour les colonnes nominales. Normalisation de `travailleur_etranger` |
| `prepare_datasets()` | Split avant normalisation — correction du data leakage. Ajout de `random_state=42` |
| `main()` | `classification_report` complet avec recall sur la classe défaut |

## Structure du projet

```
Task4D/
├── pipeline_corrige.py         # Pipeline corrigé
├── pipeline_corrige_bonus.py   # Pipeline corrigé + contrôles qualité automatisés
├── notebook.ipynb              # Audit complet et documentation des corrections
├── finova_credits.db           # Base de données source
├── finova_credits_clean.db     # Base de données nettoyée (train + test)
├── rapport_validation.csv      # Rapport des violations détectées par le pipeline bonus
└── README.md                   # Ce fichier
```

