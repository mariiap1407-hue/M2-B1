# Datasheet — Jeu de données Finova Credit Scoring

## Provenance
- **Source** : Base de données interne Finova (`finova_credits.db`)
- **Contenu** : Historique des dossiers de crédit à la consommation
- **Format source** : Base SQLite, table `dossiers_credit`
- **Période** : Non renseignée dans les données sources

## Volumétrie

| Étape | Dossiers | Colonnes |
|-------|----------|----------|
| Données brutes | 1 025 | 33 |
| Après suppression colonnes personnelles | 1 025 | 23 |
| Après nettoyage (doublons + aberrations) | 986 | 23 |
| Après encodage | 986 | 47 |
| Jeu d'entraînement | 788 | 47 |
| Jeu de test | 198 | 47 |

## Traitements appliqués

1. **Suppression des colonnes personnelles** : nom, prénom, email, téléphone, adresse, ville, IBAN, numéro de sécurité sociale, date de naissance
2. **Correction de `montant_credit`** : parsing des formats texte ("2 442 EUR", "1559,00") — 25 valeurs récupérées. Imputation des NaN restants par la médiane (2 320€)
3. **Imputation des valeurs manquantes** : `epargne` → "pas d'épargne connue" (81 cas), `anciennete_emploi` → "sans emploi" (51 cas)
4. **Suppression des doublons** : 18 doublons exacts + 7 doublons masqués supprimés
5. **Suppression des valeurs aberrantes** : 14 enregistrements supprimés (âges impossibles, montants négatifs, durées irréalistes)
6. **Normalisation** : `travailleur_etranger` — 5 variantes ramenées à 2 valeurs propres
7. **Encodage ordinal** : `epargne`, `statut_compte_courant`, `historique_credit`, `anciennete_emploi`, `emploi`
8. **One-Hot Encoding** : `sexe`, `objet_credit`, `statut_personnel_sexe`, `autres_debiteurs`, `biens`, `autres_credits`, `logement`, `telephone_declare`, `travailleur_etranger`
9. **Normalisation numérique** : StandardScaler fitté uniquement sur le jeu d'entraînement

## Limites connues

- Dans cette version, l'imputation médiane de montant_credit est réalisée avant le découpage train/test. Pour une version production entièrement conforme, cette étape devrait être intégrée dans un Pipeline sklearn afin que la médiane soit apprise uniquement sur le train et appliquée sur le test.
- `numero_client` conservé comme feature au motif qu'il encoderait l'ancienneté client — lien non vérifié dans les données
- Imputation de `anciennete_emploi` par "sans emploi" pour les NaN — hypothèse non confirmée
- Déséquilibre des classes : 70% non-défaut / 30% défaut — à traiter lors de la modélisation
- `statut_personnel_sexe` présente une asymétrie entre hommes et femmes héritée du dataset original
- Recall de 0.42 sur la classe défaut avec la régression logistique de contrôle — modèle à optimiser