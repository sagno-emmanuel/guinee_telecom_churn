# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.metrics import confusion_matrix
import joblib
import os
from sklearn.model_selection import train_test_split
# Importer vos fonctions depuis le script d'analyse
from analyse_churn import load_and_preprocess_data, evaluate_models, load_models_and_predict

# --- Configuration de la page ---
st.set_page_config(
    page_title="Analyse du churn - Guinée Télécom",
    page_icon=":material/analytics:",
    layout="wide",
)

# Nom de fichier associé à chaque modèle (les clés de `evaluate_models` sont en français,
# les fichiers .pkl sont nommés en anglais par `analyse_churn.py`)
MODEL_FILES = {
    "Regression Logistique": "modele_LogisticRegression.pkl",
    "Random Forest": "modele_RandomForest.pkl",
    "Gradient Boosting": "modele_GradientBoosting.pkl",
}

# Ordre des variables attendu par les modèles
FEATURES_ORDER = [
    "region", "sexe", "age", "revenu_estime_gnf", "anciennete_mois",
    "type_abonnement", "forfait_international", "messagerie_vocale",
    "recharge_mensuelle_moy_gnf", "minutes_jour", "minutes_nuit",
    "minutes_internationales", "donnees_mo", "nombre_sms",
    "appels_service_client", "pannes_signalees_30j",
]
DECISION_THRESHOLD = 0.50

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
        train_and_save_models(X, y)

    # Re-split pour avoir les données de test pour l'évaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    # Évaluer les modèles
    results = evaluate_models(X_test, y_test)

    return X, y, scaler, X_test, y_test, results

# Charger une fois
X, y, scaler, X_test, y_test, results = load_data_and_models()

# --- Chargement du DataFrame original pour les graphiques ---
@st.cache_data
def load_raw_data():
    return pd.read_csv("guinee_telecom_churn_FR.csv")

df_original = load_raw_data()

# --- Indicateurs utilisés dans la bannière et les cartes de statistiques ---
best_model_name = max(results, key=lambda name: results[name]['accuracy'])
best_accuracy = results[best_model_name]['accuracy']
nb_regions = df_original['region'].nunique()
nb_payment_methods = df_original['moyen_paiement'].nunique()

# --- Bandeau d'en-tête ---
with st.container(border=True):
    header_left, header_right = st.columns([3, 1], vertical_alignment="center")
    with header_left:
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown("# :material/cell_tower:")
            with st.container():
                st.markdown("### Churn télécom — Guinée")
                st.caption(
                    "Outil de prédiction du risque de résiliation client basé sur des modèles de "
                    "machine learning, entraînés sur des données clients télécom en Guinée."
                )
        st.markdown(
            f":blue-badge[:material/model_training: {best_model_name}] "
            ":orange-badge[:material/query_stats: Data mining] "
            ":green-badge[:material/public: Guinée] "
            ":gray-badge[:material/signal_cellular_alt: Télécom]"
        )
    with header_right:
        st.metric("Régions couvertes", nb_regions, icon=":material/map:", border=True)
        st.metric("Algorithme", best_model_name, icon=":material/model_training:", border=True)
        st.metric("Seuil de décision", f"{DECISION_THRESHOLD:.2f}", icon=":material/target:", border=True)

# --- Onglets ---
tab_predict, tab_about = st.tabs(
    [":material/person_search: Prédiction client", ":material/info: À propos du modèle"],
    on_change="rerun",
)

# =========================================================================
# Onglet 1 : Prédiction client
# =========================================================================
if tab_predict.open:
    with tab_predict:
        with st.container(horizontal=True):
            st.metric("Régions de Guinée", nb_regions, icon=":material/map:", border=True)
            st.metric("Variables d'entrée", len(FEATURES_ORDER), icon=":material/table_chart:", border=True)
            st.metric("Meilleur modèle", best_model_name, icon=":material/model_training:", border=True)
            st.metric("Seuil de décision", f"{DECISION_THRESHOLD:.2f}", icon=":material/target:", border=True)
            st.metric("Moyens de paiement", nb_payment_methods, icon=":material/payments:", border=True)

        # Création d'un formulaire
        with st.form(key='prediction_form'):
            st.subheader(":material/person: Informations client")
            col1, col2, col3 = st.columns(3)
            with col1:
                region = st.selectbox("Région", sorted(df_original['region'].unique()))
                forfait_international = st.segmented_control(
                    "Forfait international", ['Non', 'Oui'], default='Non', required=True
                )
                minutes_international = st.number_input("Minutes internationales", min_value=0, max_value=120, value=5)
            with col2:
                sexe = st.segmented_control("Sexe", ['Homme', 'Femme'], default='Homme', required=True)
                age = st.slider("Âge", min_value=18, max_value=100, value=30)
                messagerie_vocale = st.segmented_control(
                    "Messagerie vocale", ['Non', 'Oui'], default='Oui', required=True
                )
            with col3:
                revenu = st.number_input("Revenu estimé (GNF)", min_value=300000, max_value=4700000, value=700000)
                anciennete = st.slider("Ancienneté (mois)", min_value=1, max_value=100, value=12)
                abonnement = st.segmented_control(
                    "Type d'abonnement", sorted(df_original['type_abonnement'].unique()),
                    default=sorted(df_original['type_abonnement'].unique())[0], required=True,
                )

            st.subheader(":material/bar_chart: Usage & comportement")
            col4, col5, col6 = st.columns(3)
            with col4:
                recharge_moyenne = st.number_input("Recharge mensuelle moyenne (GNF)", min_value=10000, max_value=310000, value=50000)
                donnees_mo = st.number_input("Données utilisées (Mo)", min_value=50, max_value=6500, value=2500)
            with col5:
                minutes_jour = st.number_input("Minutes d'appel (jour)", min_value=0, max_value=500, value=180)
                nb_sms = st.number_input("Nombre de SMS", min_value=0, max_value=250, value=25)
            with col6:
                minutes_nuit = st.number_input("Minutes d'appel (nuit)", min_value=0, max_value=300, value=90)
                col6a, col6b = st.columns(2)
                with col6a:
                    appels_service_client = st.number_input("Appels service client", min_value=0, max_value=6, value=1)
                with col6b:
                    pannes = st.number_input("Pannes signalées (30j)", min_value=0, max_value=5, value=0)

            submit_button = st.form_submit_button(label="Prédire le churn", icon=":material/insights:")

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
                "pannes_signalees_30j": pannes,
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
                'messagerie_vocale': {'Non': 0, 'Oui': 1},
            }

            # Encoder les variables
            input_data['region'] = mapping['region'][input_data['region']]
            input_data['sexe'] = mapping['sexe'][input_data['sexe']]
            input_data['type_abonnement'] = mapping['type_abonnement'][input_data['type_abonnement']]
            input_data['forfait_international'] = mapping['forfait_international'][input_data['forfait_international']]
            input_data['messagerie_vocale'] = mapping['messagerie_vocale'][input_data['messagerie_vocale']]

            # Créer le tableau numpy pour la prédiction
            X_new = np.array([[input_data[f] for f in FEATURES_ORDER]])

            # Normaliser les données
            X_new_scaled = scaler.transform(X_new)

            # Faire la prédiction avec les modèles
            predictions = load_models_and_predict(X_new_scaled)

            # Afficher les résultats
            st.subheader("Résultat de la prédiction")

            # Sélectionner le meilleur modèle pour l'affichage principal
            best_model_path = os.path.join("modeles", MODEL_FILES[best_model_name])
            if os.path.exists(best_model_path):
                best_model = joblib.load(best_model_path)
                proba = best_model.predict_proba(X_new_scaled)[0]
                churn_risk = proba[1] > DECISION_THRESHOLD

                with st.container(horizontal=True):
                    st.metric(
                        "Probabilité de rester", f"{proba[0] * 100:.1f}%",
                        icon=":material/favorite:", border=True,
                    )
                    st.metric(
                        "Probabilité de résilier", f"{proba[1] * 100:.1f}%",
                        icon=":material/person_off:", border=True,
                    )
                    with st.container(border=True):
                        if churn_risk:
                            st.badge("Résiliation probable", icon=":material/warning:", color="red")
                        else:
                            st.badge("Client fidèle", icon=":material/check_circle:", color="green")
                        st.caption(f"Modèle utilisé : {best_model_name}")

            # Afficher les prédictions de tous les modèles
            with st.container(border=True):
                st.markdown("**Prédictions par modèle**")
                pred_df = pd.DataFrame({
                    'Modèle': list(predictions.keys()),
                    'Prédiction': [
                        ("Résiliation" if (isinstance(p, np.ndarray) and p[0] == 1) else "Reste")
                        if isinstance(p, np.ndarray) else p
                        for p in predictions.values()
                    ],
                })
                st.dataframe(pred_df, hide_index=True, width="stretch")

# =========================================================================
# Onglet 2 : À propos du modèle (vue d'ensemble + EDA + performance)
# =========================================================================
if tab_about.open:
    with tab_about:
        # --- Vue d'ensemble du jeu de données ---
        st.subheader(":material/dashboard: Vue d'ensemble du jeu de données")

        nb_churn = int((df_original['resiliation'] == 'Oui').sum())
        churn_rate = nb_churn / len(df_original) * 100

        with st.container(horizontal=True):
            st.metric(
                "Clients au total", f"{len(df_original):,}",
                icon=":material/group:", border=True,
            )
            st.metric(
                "Clients résiliés", f"{nb_churn:,}", f"{churn_rate:.1f}% du total",
                icon=":material/person_off:", border=True, delta_color="off",
            )
            st.metric(
                "Variables explicatives", df_original.shape[1] - 1,
                icon=":material/table_chart:", border=True,
            )

        with st.container(border=True):
            st.markdown("**Aperçu des données**")
            st.dataframe(df_original.head(), hide_index=True, width="stretch")

        with st.expander("Informations sur les colonnes", icon=":material/info:"):
            st.dataframe(
                pd.DataFrame({
                    "Colonne": df_original.columns,
                    "Type": df_original.dtypes.astype(str).values,
                    "Valeurs non nulles": df_original.count().values,
                    "Valeurs uniques": df_original.nunique().values,
                }),
                hide_index=True,
                width="stretch",
            )

        # --- Analyse exploratoire ---
        st.subheader(":material/query_stats: Analyse exploratoire des données")

        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("**Répartition des résiliations**")
                counts = df_original['resiliation'].value_counts().reset_index()
                counts.columns = ['resiliation', 'nombre']
                chart = alt.Chart(counts).mark_bar().encode(
                    x=alt.X('resiliation:N', title='Résiliation'),
                    y=alt.Y('nombre:Q', title='Nombre de clients'),
                    color=alt.Color('resiliation:N', legend=None),
                    tooltip=['resiliation', 'nombre'],
                ).properties(height=320)
                st.altair_chart(chart, width="stretch")

        with col2:
            with st.container(border=True):
                st.markdown("**Distribution de l'âge par statut**")
                box = alt.Chart(df_original).mark_boxplot(extent='min-max').encode(
                    x=alt.X('resiliation:N', title='Résiliation'),
                    y=alt.Y('age:Q', title='Âge'),
                    color=alt.Color('resiliation:N', legend=None),
                ).properties(height=320)
                st.altair_chart(box, width="stretch")

        with st.container(border=True):
            st.markdown("**Matrice de corrélation des variables numériques**")
            numeric_df = df_original.select_dtypes(include=['int64', 'float64'])
            corr = numeric_df.corr().reset_index().melt(id_vars='index')
            corr.columns = ['var1', 'var2', 'correlation']

            heat = alt.Chart(corr).mark_rect().encode(
                x=alt.X('var2:N', title=None),
                y=alt.Y('var1:N', title=None),
                color=alt.Color(
                    'correlation:Q', title='Corrélation',
                    scale=alt.Scale(scheme='redblue', domain=[-1, 1]),
                ),
                tooltip=['var1', 'var2', alt.Tooltip('correlation:Q', format='.2f')],
            )
            text = alt.Chart(corr).mark_text(fontSize=9).encode(
                x=alt.X('var2:N'),
                y=alt.Y('var1:N'),
                text=alt.Text('correlation:Q', format='.2f'),
                color=alt.condition(
                    'abs(datum.correlation) > 0.5', alt.value('white'), alt.value('black')
                ),
            )
            st.altair_chart((heat + text).properties(height=440), width="stretch")

        # --- Performance des modèles ---
        st.subheader(":material/model_training: Performance des modèles")

        perf_df = pd.DataFrame({
            'Modèle': list(results.keys()),
            'Accuracy': [metrics['accuracy'] for metrics in results.values()],
        }).sort_values('Accuracy', ascending=False).reset_index(drop=True)

        col1, col2 = st.columns([3, 2])
        with col1:
            with st.container(border=True):
                st.markdown("**Comparaison des accuracy**")
                bars = alt.Chart(perf_df).mark_bar().encode(
                    x=alt.X('Accuracy:Q', scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y('Modèle:N', sort='-x', title=None),
                    color=alt.Color('Modèle:N', legend=None),
                    tooltip=['Modèle', alt.Tooltip('Accuracy:Q', format='.3f')],
                )
                labels = bars.mark_text(align='left', dx=3).encode(
                    text=alt.Text('Accuracy:Q', format='.3f')
                )
                st.altair_chart((bars + labels).properties(height=260), width="stretch")

        with col2:
            st.metric(
                "Meilleur modèle", best_model_name, f"Accuracy = {best_accuracy:.3f}",
                icon=":material/military_tech:", border=True, delta_color="off",
            )
            with st.container(border=True):
                st.markdown("**Précision par modèle**")
                st.dataframe(
                    perf_df.style.format({'Accuracy': '{:.3f}'}),
                    hide_index=True, width="stretch",
                )

        st.markdown("**Détail par modèle**")
        model_choice = st.segmented_control(
            "Modèle à examiner", list(results.keys()),
            default=list(results.keys())[0], required=True,
        )

        metrics = results[model_choice]
        cm = confusion_matrix(y_test, metrics['y_pred'])
        report = metrics['report']

        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown(f"**Matrice de confusion — {model_choice}**")
                cm_df = pd.DataFrame(
                    cm,
                    index=['Réel : reste', 'Réel : résilie'],
                    columns=['Prédit : reste', 'Prédit : résilie'],
                ).reset_index(names='Réel').melt(id_vars='Réel', var_name='Prédit', value_name='Nombre')

                threshold = cm.max() / 2
                cm_heat = alt.Chart(cm_df).mark_rect().encode(
                    x=alt.X('Prédit:N', title=None),
                    y=alt.Y('Réel:N', title=None),
                    color=alt.Color('Nombre:Q', scale=alt.Scale(scheme='blues'), legend=None),
                    tooltip=['Réel', 'Prédit', 'Nombre'],
                )
                cm_text = alt.Chart(cm_df).mark_text(fontSize=16).encode(
                    x=alt.X('Prédit:N'),
                    y=alt.Y('Réel:N'),
                    text='Nombre:Q',
                    color=alt.condition(
                        f'datum.Nombre > {threshold}', alt.value('white'), alt.value('black')
                    ),
                )
                st.altair_chart((cm_heat + cm_text).properties(height=260), width="stretch")

        with col2:
            with st.container(border=True):
                st.markdown("**Rapport de classification**")
                st.dataframe(
                    pd.DataFrame(report).transpose().style.format(precision=3),
                    width="stretch",
                )

        st.caption("Source des données : guinee_telecom_churn_FR.csv")
