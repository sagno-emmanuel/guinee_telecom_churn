# analyse_churn.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib # Pour sauvegarder et charger les modèles
import os

# --- 1. Fonction de chargement et de prétraitement ---
def load_and_preprocess_data(filepath="guinee_telecom_churn_FR.csv"):
    """Charge le fichier CSV, encode les variables catégorielles et normalise."""
    df = pd.read_csv(filepath)

    # Encodage des colonnes catégorielles
    label_encoder = LabelEncoder()
    categorical_cols = ['region', 'sexe', 'type_abonnement', 'forfait_international',
                        'messagerie_vocale', 'moyen_paiement', 'retour_client', 'resiliation']
    for col in categorical_cols:
        df[col] = label_encoder.fit_transform(df[col])

    # Séparation des features et de la cible
    features = [
        "region", "sexe", "age", "revenu_estime_gnf", "anciennete_mois",
        "type_abonnement", "forfait_international", "messagerie_vocale",
        "recharge_mensuelle_moy_gnf", "minutes_jour", "minutes_nuit",
        "minutes_internationales", "donnees_mo", "nombre_sms",
        "appels_service_client", "pannes_signalees_30j"
    ]
    X = df[features]
    y = df["resiliation"]

    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Retourner le scaler entraîné pour l'utiliser plus tard
    return X_scaled, y, scaler

# --- 2. Fonction pour entraîner et sauvegarder les modèles ---
def train_and_save_models(X, y, model_dir="modeles"):
    """Entraîne les trois modèles et les sauvegarde dans le dossier spécifié."""
    # Créer le dossier si il n'existe pas
    os.makedirs(model_dir, exist_ok=True)

    # Séparation des données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    # Initialisation des modèles
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=0),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=0),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=0)
    }

    # Entraînement et sauvegarde
    for name, model in models.items():
        print(f"Entraînement de {name}...")
        model.fit(X_train, y_train)
        # Sauvegarder le modèle
        joblib.dump(model, os.path.join(model_dir, f"modele_{name}.pkl"))
        print(f"Modèle {name} sauvegardé.")

    # Retourner les données de test pour l'évaluation
    return X_test, y_test

# --- 3. Fonction pour charger les modèles et faire des prédictions ---
def load_models_and_predict(X_input, model_dir="modeles"):
    """Charge les modèles sauvegardés et fait des prédictions sur les nouvelles données."""
    predictions = {}
    model_files = {
        "LogisticRegression": "modele_LogisticRegression.pkl",
        "RandomForest": "modele_RandomForest.pkl",
        "GradientBoosting": "modele_GradientBoosting.pkl"
    }
    for name, filename in model_files.items():
        model_path = os.path.join(model_dir, filename)
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            predictions[name] = model.predict(X_input)
        else:
            predictions[name] = f"Erreur : Modèle {name} non trouvé."
    return predictions

# --- 4. Fonction d'évaluation (pour l'analyse dans l'app) ---
def evaluate_models(X_test, y_test, model_dir="modeles"):
    """Évalue les modèles chargés sur les données de test."""
    results = {}
    model_files = {
        "Regression Logistique": "modele_LogisticRegression.pkl",
        "Random Forest": "modele_RandomForest.pkl",
        "Gradient Boosting": "modele_GradientBoosting.pkl"
    }
    for name, filename in model_files.items():
        model_path = os.path.join(model_dir, filename)
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                "accuracy": accuracy,
                "y_pred": y_pred,
                "report": classification_report(y_test, y_pred, output_dict=True)
            }
    return results


# --- Cette partie ne s'exécute que si vous lancez le script directement ---
# --- et non via l'import dans l'app Streamlit ---
if __name__ == "__main__":
    print("--- Chargement et préparation des données ---")
    X, y, scaler = load_and_preprocess_data()

    print("--- Entraînement et sauvegarde des modèles ---")
    X_test, y_test = train_and_save_models(X, y)

    print("--- Évaluation des modèles ---")
    results = evaluate_models(X_test, y_test)
    for name, metrics in results.items():
        print(f"{name} : Accuracy = {metrics['accuracy']:.4f}")