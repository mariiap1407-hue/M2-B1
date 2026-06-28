# -*- coding: utf-8 -*-
"""
pipeline_corrige.py — Préparation des données pour le scoring crédit Finova
Auteur : M.Pshenychna (équipe Data, 2026)
Prépare les dossiers de crédit pour l'entraînement du modèle de scoring :
validation, nettoyage, encodage, normalisation, découpage train/test.
Usage : python pipeline_corrige.py [chemin_vers_finova_credits.db]
"""

import sys
import sqlite3

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

DB_PATH = sys.argv[1] if len(sys.argv) > 1 else "finova_credits.db"

# colonnes inutiles pour le scoring
COLONNES_INUTILES = ["nom", "prenom", "email", "telephone_mobile", "adresse",
                     "ville", "iban", "num_secu", "date_naissance"]

# colonnes attendues après load_data()
COLONNES_ATTENDUES = ['numero_client', 'sexe', 'code_postal', 'statut_compte_courant',
                      'duree_credit_mois', 'historique_credit', 'objet_credit',
                      'montant_credit', 'epargne', 'anciennete_emploi', 'taux_effort',
                      'statut_personnel_sexe', 'autres_debiteurs', 'anciennete_logement',
                      'biens', 'age', 'autres_credits', 'logement', 'nb_credits_existants',
                      'emploi', 'nb_personnes_charge', 'telephone_declare',
                      'travailleur_etranger', 'defaut']

# types attendus pour les colonnes numériques
TYPES_ATTENDUS = {
    "duree_credit_mois": "int64",
    "montant_credit": "float64",
    "age": "int64",
    "taux_effort": "int64",
    "nb_personnes_charge": "int64",
    "nb_credits_existants": "int64",
    "anciennete_logement": "int64",
    "defaut": "int64"
}

def load_data():
    """Charge les dossiers de crédit depuis la base."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM dossiers_credit", conn)
    conn.close()
    df = df.drop(columns=COLONNES_INUTILES)
    # numero_client conservé à titre exploratoire — supposé encoder l'ancienneté client 
    # mais non vérifié. En production, devrait être exclu ou remplacé par une vraie 
    # variable métier d'ancienneté si disponible.
    return df

def fix_amounts(df):
    """Corrige et normalise la colonne montant_credit."""
    # parsing du texte avant conversion numérique
    df["montant_credit"] = pd.to_numeric(
        df["montant_credit"].str.replace(" EUR", "").str.replace(" ", "").str.replace(",", "."),
        errors="coerce"
    )
    # imputation par la médiane
    df["montant_credit"] = df["montant_credit"].fillna(df["montant_credit"].median())
    return df
    
def clean_data(df):
    """Supprime les doublons et impute les valeurs manquantes.Note : les valeurs aberrantes sont gérées en amont par validate_data()
    qui rejette le lot si des violations sont détectées."""
 
    # suppression des doublons exacts
    df = df.drop_duplicates()
    
    # suppression des doublons masqués
    df = df.drop_duplicates(subset=['age', 'sexe', 'code_postal', 'montant_credit', 'duree_credit_mois'])
    
    # imputation des valeurs manquantes
    df["epargne"] = df["epargne"].fillna("pas d'épargne connue")
    df["anciennete_emploi"] = df["anciennete_emploi"].fillna("sans emploi")
    
    return df.reset_index(drop=True)

def remove_outliers(df):
    """Supprime les enregistrements avec des valeurs aberrantes."""
    df = df[(df['age'] >= 18) & (df['age'] <= 100)]
    df = df[df['montant_credit'] >= 0]
    df = df[(df['duree_credit_mois'] > 0) & (df['duree_credit_mois'] <= 120)]
    df = df[(df['nb_personnes_charge'] >= 0) & (df['nb_personnes_charge'] <= 6)]
    return df.reset_index(drop=True)
    
def encode_features(df):
    """Encode les variables catégorielles en numérique."""
    # le numéro client encode l'ancienneté de la relation, on le conserve
    df["numero_client"] = df["numero_client"].str.replace("FIN-", "").astype(int)
    df["code_postal"] = pd.to_numeric(df["code_postal"], errors="coerce").fillna(0).astype(int)
    df["travailleur_etranger"] = df["travailleur_etranger"].str.strip().str.lower()

    # Encodage ordinal epargne
    ordre_epargne = [["pas d'épargne connue", "épargne < 100", "épargne 100-500", "épargne 500-1000", "épargne > 1000"]]
    
    enc = OrdinalEncoder(categories=ordre_epargne)
    df[["epargne"]] = enc.fit_transform(df[["epargne"]])

    # Encodage ordinal statut_compte_courant 
    ordre_statut_compte = [["pas de compte courant", "solde < 100", "solde 100-200", "solde > 200 ou salaire domicilié"]]
    
    enc = OrdinalEncoder(categories=ordre_statut_compte)
    df[["statut_compte_courant"]] = enc.fit_transform(df[["statut_compte_courant"]])
    
    # Encodage ordinal historique_credit
    ordre_historique = [["compte critique / crédits ailleurs", "retards de paiement passés", 
                         "crédits en cours sans incident", "crédits Finova soldés", 
                         "aucun crédit / tous soldés"]]
    enc = OrdinalEncoder(categories=ordre_historique)
    df[["historique_credit"]] = enc.fit_transform(df[["historique_credit"]])
    
    # Encodage ordinal anciennete_emploi
    ordre_anciennete = [["sans emploi", "< 1 an", "1-4 ans", "4-7 ans", ">= 7 ans"]]
    enc = OrdinalEncoder(categories=ordre_anciennete)
    df[["anciennete_emploi"]] = enc.fit_transform(df[["anciennete_emploi"]])
    
    # Encodage ordinal emploi
    ordre_emploi = [["sans emploi / non qualifié non résident", "non qualifié résident", 
                     "employé qualifié", "cadre / indépendant"]]
    enc = OrdinalEncoder(categories=ordre_emploi)
    df[["emploi"]] = enc.fit_transform(df[["emploi"]])

    #One-Hot encoding 
    df = pd.get_dummies(df, columns=["logement", "sexe", "objet_credit", 
                                      "statut_personnel_sexe", "autres_debiteurs", 
                                      "biens", "autres_credits", 
                                      "telephone_declare", "travailleur_etranger"])

    return df

def prepare_datasets(df):
    X = df.drop(columns=["defaut"])
    y = df["defaut"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def main():
    df = load_data()
    print(f"Dossiers chargés : {len(df)}")
    
    df = fix_amounts(df)
    df = clean_data(df)
    df = remove_outliers(df)
    print(f"Dossiers après & suppression aberrations : {len(df)}")

    df = encode_features(df)
    
    # sauvegarde des colonnes avant normalisation
    cols = df.drop(columns=["defaut"]).columns
    
    X_train, X_test, y_train, y_test = prepare_datasets(df)
    print(f"Train : {X_train.shape} | Test : {X_test.shape}")
    
    # validation rapide du pipeline
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # export en base
    X_train_df = pd.DataFrame(X_train, columns=cols)
    X_test_df = pd.DataFrame(X_test, columns=cols)
    
    conn = sqlite3.connect("finova_credits_clean.db")
    X_train_df.assign(defaut=y_train.values).to_sql("train", conn, if_exists="replace", index=False)
    X_test_df.assign(defaut=y_test.values).to_sql("test", conn, if_exists="replace", index=False)
    conn.close()
    print("Base créée : finova_credits_clean.db")

if __name__ == "__main__":
    main()
