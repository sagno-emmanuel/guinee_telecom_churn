# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import joblib
import os
from sklearn.model_selection import train_test_split
# Importer vos fonctions depuis le script d'analyse
from analyse_churn import load_and_preprocess_data, evaluate_models, load_models_and_predict

# --- Configuration de la page ---
st.set_page_config(
    page_title="Analyse du Churn - Guinée Télécom",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Analyse du Churn des Clients de Guinée Télécom")
st.markdown("Cette application présente l'analyse du taux d'attrition (churn) des clients, avec des modèles de Machine Learning pour prédire les résiliations.")

# --- Sidebar pour la navigation ---
st.sidebar.title("Navigation")
options = st.sidebar.radio(
    "Choisissez une section :",
    ["Vue d'ensemble des données", "Analyse exploratoire", "Performance des modèles", "Prédiction sur un nouveau client"]
)

# --- Charger les données et les modèles (une seule fois) ---
@st.cache_resource
def load_data_and_models():
    """Charge les données prétraitées, les modèles et les métriques."""
    # Charger et prétraiter les données
    X, y, scaler = load_and_preprocess_data("guinee_telecom_churn_FR.csv")

    # Vérifier si les modèles existent, sinon les entraîner
    model_dir = "modeles"
    if not os.path.exists(model_dir) or len(os.listdir(model_dir)) == 0:
        st.info("Entraînement des modèles en cours... Cela peut prendre un moment.")
        from analyse_churn import train_and_save_models
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
        train_and_save_models(X, y)
    else:
        # Re-split pour avoir les données de test pour l'évaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    # Évaluer les modèles
    results = evaluate_models(X_test, y_test)

    return X, y, scaler, X_test, y_test, results

# Charger une fois
X, y, scaler, X_test, y_test, results = load_data_and_models()

# --- Chargement du DataFrame original pour les graphiques ---
df_original = pd.read_csv("guinee_telecom_churn_FR.csv")

# --- 1. Vue d'ensemble ---
if options == "Vue d'ensemble des données":
    st.header("Vue d'ensemble du jeu de données")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nombre total de clients", f"{len(df_original):,}")
    with col2:
        nb_churn = df_original[df_original['resiliation'] == 'Oui'].shape[0]
        st.metric("Clients résiliés", f"{nb_churn:,}", delta=f"{nb_churn/len(df_original)*100:.1f}% du total")
    with col3:
        st.metric("Nombre de features", df_original.shape[1] - 1)

    with st.expander("Aperçu des 5 premières lignes du jeu de données"):
        st.dataframe(df_original.head())

    with st.expander("Information sur les colonnes"):
        st.dataframe(pd.DataFrame({
            "Colonne": df_original.columns,
            "Type": df_original.dtypes.values,
            "Valeurs non nulles": df_original.count().values,
            "Valeurs uniques": df_original.nunique().values
        }))

# --- 2. Analyse exploratoire ---
elif options == "Analyse exploratoire":
    st.header("Analyse Exploratoire des Données (EDA)")

    # Graphique 1 : Répartition des résiliations
    st.subheader("Répartition des clients résiliés vs. restants")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='resiliation', data=df_original, palette='Set2', ax=ax1)
    ax1.set_title('Nombre de clients par statut de résiliation')
    ax1.set_xlabel('Résiliation (Non / Oui)')
    ax1.set_ylabel('Nombre de clients')
    st.pyplot(fig1)

    # Graphique 2 : Matrice de corrélation
    st.subheader("Matrice de corrélation des variables numériques")
    # Sélectionner uniquement les colonnes numériques pour la corrélation
    numeric_df = df_original.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr()

    fig2, ax2 = plt.subplots(figsize=(15, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax2)
    ax2.set_title('Matrice de corrélation')
    st.pyplot(fig2)

    # Graphique 3 : Distribution de l'âge par statut
    st.subheader("Distribution de l'âge par statut de résiliation")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x='resiliation', y='age', data=df_original, palette='Set2', ax=ax3)
    ax3.set_title('Distribution de l\'âge')
    st.pyplot(fig3)


# --- 3. Performance des modèles ---
elif options == "Performance des modèles":
    st.header("Performance des modèles de prédiction du Churn")

    # Créer un DataFrame pour les métriques de performance
    perf_df = pd.DataFrame({
        'Modèle': list(results.keys()),
        'Accuracy': [metrics['accuracy'] for metrics in results.values()]
    }).sort_values('Accuracy', ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Comparaison des Accuracy")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x='Accuracy', y='Modèle', data=perf_df, palette='viridis', ax=ax)
        ax.set_xlim(0, 1)
        ax.set_title('Comparaison de la précision des modèles')
        for i, v in enumerate(perf_df['Accuracy']):
            ax.text(v + 0.01, i, f"{v:.3f}", va='center')
        st.pyplot(fig)

    with col2:
        st.subheader("Meilleur modèle")
        best_model = perf_df.iloc[0]['Modèle']
        best_accuracy = perf_df.iloc[0]['Accuracy']
        st.metric("Modèle avec la meilleure performance", best_model, f"Accuracy = {best_accuracy:.3f}")

        st.subheader("Précision par modèle")
        st.dataframe(perf_df.style.format({'Accuracy': '{:.3f}'}))

    # Matrices de confusion et rapport pour chaque modèle
    st.subheader("Détail des performances par modèle")
    model_choice = st.selectbox("Sélectionnez un modèle à visualiser", list(results.keys()))

    if model_choice:
        metrics = results[model_choice]
        cm = confusion_matrix(y_test, metrics['y_pred'])
        report = metrics['report']

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'Matrice de confusion - {model_choice}')
            ax.set_xlabel('Prédit')
            ax.set_ylabel('Réel')
            st.pyplot(fig)

        with col2:
            st.dataframe(pd.DataFrame(report).transpose().style.format(precision=3))

# --- 4. Prédiction sur un nouveau client ---
else:
    st.header("Prédiction du churn pour un nouveau client")

    st.markdown("""
    Remplissez les informations du client ci-dessous pour obtenir une prédiction de sa probabilité de résiliation.
    """)

    # Création d'un formulaire
    with st.form(key='prediction_form'):
        col1, col2 = st.columns(2)
        with col1:
            region = st.selectbox("Région", sorted(df_original['region'].unique()))
            sexe = st.selectbox("Sexe", ['Homme', 'Femme'])
            age = st.number_input("Âge", min_value=18, max_value=100, value=30)
            revenu = st.number_input("Revenu estimé (GNF)", min_value=300000, max_value=4700000, value=700000)
            anciennete = st.number_input("Ancienneté (mois)", min_value=1, max_value=100, value=12)

        with col2:
            abonnement = st.selectbox("Type d'abonnement", df_original['type_abonnement'].unique())
            forfait_international = st.selectbox("Forfait International", ['Non', 'Oui'])
            messagerie_vocale = st.selectbox("Messagerie Vocale", ['Non', 'Oui'])
            recharge_moyenne = st.number_input("Recharge mensuelle moyenne (GNF)", min_value=10000, max_value=310000, value=50000)

        col3, col4 = st.columns(2)
        with col3:
            minutes_jour = st.number_input("Minutes d'appel (jour)", min_value=0, max_value=500, value=180)
            minutes_nuit = st.number_input("Minutes d'appel (nuit)", min_value=0, max_value=300, value=90)
            minutes_international = st.number_input("Minutes internationales", min_value=0, max_value=120, value=5)

        with col4:
            donnees_mo = st.number_input("Données utilisées (Mo)", min_value=50, max_value=6500, value=2500)
            nb_sms = st.number_input("Nombre de SMS", min_value=0, max_value=250, value=25)
            appels_service_client = st.number_input("Appels au service client", min_value=0, max_value=6, value=1)
            pannes = st.number_input("Pannes signalées (30 jours)", min_value=0, max_value=5, value=0)

        submit_button = st.form_submit_button(label="Prédire le churn")

    if submit_button:
        # Créer un dictionnaire avec les données du client
        # Attention : il faut encoder les données exactement comme lors de l'entraînement
        # Utiliser le même ordre de colonnes
        input_data = {
            "region": region,
            "sexe": sexe,
            "age": age,
            "revenu_estime_gnf": revenu,
            "anciennete_mois": anciennete,
            "type_abonnement": abonnement,
            "forfait_international": forfait_international,
            "messagerie_vocale": messagerie_vocale,
            "recharge_mensuelle_moy_gnf": recharge_moyenne,
            "minutes_jour": minutes_jour,
            "minutes_nuit": minutes_nuit,
            "minutes_internationales": minutes_international,
            "donnees_mo": donnees_mo,
            "nombre_sms": nb_sms,
            "appels_service_client": appels_service_client,
            "pannes_signalees_30j": pannes
        }

        # Transformer les données catégorielles en numériques
        # Il faut utiliser les mêmes encodeurs que lors de l'entraînement
        # Pour simplifier, on va utiliser un mapping direct (car les valeurs sont les mêmes)
        # Une meilleure approche serait de sauvegarder les encodeurs avec joblib
        mapping = {
            'region': {v: i for i, v in enumerate(sorted(df_original['region'].unique()))},
            'sexe': {'Homme': 1, 'Femme': 0},
            'type_abonnement': {v: i for i, v in enumerate(sorted(df_original['type_abonnement'].unique()))},
            'forfait_international': {'Non': 0, 'Oui': 1},
            'messagerie_vocale': {'Non': 0, 'Oui': 1}
        }

        # Encoder les variables
        input_data['region'] = mapping['region'][input_data['region']]
        input_data['sexe'] = mapping['sexe'][input_data['sexe']]
        input_data['type_abonnement'] = mapping['type_abonnement'][input_data['type_abonnement']]
        input_data['forfait_international'] = mapping['forfait_international'][input_data['forfait_international']]
        input_data['messagerie_vocale'] = mapping['messagerie_vocale'][input_data['messagerie_vocale']]

        # Créer le tableau numpy pour la prédiction
        features_order = [
            "region", "sexe", "age", "revenu_estime_gnf", "anciennete_mois",
            "type_abonnement", "forfait_international", "messagerie_vocale",
            "recharge_mensuelle_moy_gnf", "minutes_jour", "minutes_nuit",
            "minutes_internationales", "donnees_mo", "nombre_sms",
            "appels_service_client", "pannes_signalees_30j"
        ]
        X_new = np.array([[input_data[f] for f in features_order]])

        # Normaliser les données
        X_new_scaled = scaler.transform(X_new)

        # Faire la prédiction avec les modèles
        predictions = load_models_and_predict(X_new_scaled)

        # Afficher les résultats
        st.subheader("Résultats de la prédiction")

        # Sélectionner le meilleur modèle pour l'affichage principal
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        best_model_path = os.path.join("modeles", f"modele_{best_model_name.replace(' ', '')}.pkl")
        if os.path.exists(best_model_path):
            best_model = joblib.load(best_model_path)
            proba = best_model.predict_proba(X_new_scaled)[0]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Probabilité de rester", f"{proba[0]*100:.1f}%")
            with col2:
                st.metric("Probabilité de résilier", f"{proba[1]*100:.1f}%")
            with col3:
                prediction = "Résiliation probable" if proba[1] > 0.5 else "Client fidèle"
                st.metric("Prédiction", prediction, delta=None)

        # Afficher les prédictions de tous les modèles
        st.subheader("Prédictions par modèle")
        pred_df = pd.DataFrame({
            'Modèle': list(predictions.keys()),
            'Prédiction (0 = reste, 1 = résilie)': [p[0] if isinstance(p, np.ndarray) else p for p in predictions.values()]
        })
        st.dataframe(pred_df)