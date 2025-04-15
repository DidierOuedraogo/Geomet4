import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import base64
from io import StringIO
from datetime import datetime
import time
import pickle
import joblib
import os
import shap
import folium
from streamlit_folium import folium_static
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
import warnings
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings('ignore')

# Tentative d'importation de TensorFlow - utilisée pour les réseaux de neurones
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Application de prédiction géométallurgique",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS pour le style de l'application
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
        color: #31333F;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2563EB;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #F9FAFB;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    .footer {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #E5E7EB;
        text-align: center;
        font-size: 0.8rem;
        color: #6B7280;
    }
    .sidebar .sidebar-content {
        background-color: #F1F5F9;
    }
    .stProgress > div > div {
        background-color: #1E3A8A;
    }
    .plot-container {
        background-color: white;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .highlight {
        background-color: #FEF3C7;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .feature-importance-container {
        height: 400px;
        overflow-y: auto;
    }
    .data-info {
        font-size: 0.9rem;
        color: #4B5563;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E0F2FE;
        border-left: 4px solid #0EA5E9;
        padding: 0.8rem;
        margin-bottom: 1rem;
        border-radius: 0 5px 5px 0;
    }
    .warning-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 0.8rem;
        margin-bottom: 1rem;
        border-radius: 0 5px 5px 0;
    }
    .error-box {
        background-color: #FEE2E2;
        border-left: 4px solid #EF4444;
        padding: 0.8rem;
        margin-bottom: 1rem;
        border-radius: 0 5px 5px 0;
    }
    .success-box {
        background-color: #DCFCE7;
        border-left: 4px solid #22C55E;
        padding: 0.8rem;
        margin-bottom: 1rem;
        border-radius: 0 5px 5px 0;
    }
    /* Styliser les métriques */
    .metric-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        flex: 1;
        margin: 0 0.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6B7280;
    }
    /* Styliser les sélecteurs */
    .stSelectbox > div > div {
        background-color: white;
    }
    /* Styliser la barre de progression */
    .stProgress > div {
        border-radius: 10px;
    }
    /* Styliser les onglets */
    .stTabs > div > div > div {
        border-radius: 5px 5px 0 0;
    }
    .stTabs > div > div > div[aria-selected="true"] {
        background-color: #1E3A8A;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour l'authentification
def check_password():
    """Returns `True` if the user had the correct password."""
    
    # Configuration des utilisateurs et mots de passe avec des mots de passe plus complexes
    users = {
        "didier": "Geo_Metal2025!",
        "admin": "S3cur3P@ssw0rd!"
    }
    
    # Initialiser l'état d'authentification s'il n'existe pas
    if "authentication_status" not in st.session_state:
        st.session_state["authentication_status"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = ""
        
    # Si l'utilisateur est déjà authentifié, retourner True
    if st.session_state["authentication_status"]:
        return True
    
    # Sinon, afficher le formulaire de connexion
    st.title("Application de prédiction géométallurgique")
    st.subheader("Connexion")
    
    # Formulaire de connexion
    with st.form("login_form"):
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Se connecter")
    
    if submitted:
        # Vérifier les identifiants
        if username in users and users[username] == password:
            st.session_state["authentication_status"] = True
            st.session_state["username"] = username
            st.success(f"Bienvenue {username}!")
            st.rerun()
        else:
            st.error("Nom d'utilisateur ou mot de passe incorrect")
            return False
    
    return False

# Solution temporaire pour contourner l'authentification (à commenter en production)
if st.sidebar.checkbox("Mode développement (ignorer l'authentification)", False):
    st.session_state["authentication_status"] = True
    st.session_state["username"] = "didier"

# Vérifier l'authentification
if not check_password():
    st.stop()

# Initier les variables de session
if 'data' not in st.session_state:
    st.session_state.data = None
if 'csv_data' not in st.session_state:
    st.session_state.csv_data = None
if 'page' not in st.session_state:
    st.session_state.page = "Importation des données"
if 'features' not in st.session_state:
    st.session_state.features = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'model_pipeline' not in st.session_state:
    st.session_state.model_pipeline = None
if 'model_is_fitted' not in st.session_state:
    st.session_state.model_is_fitted = False
if 'neural_network' not in st.session_state:
    st.session_state.neural_network = {}
    
# Sidebar
with st.sidebar:
    st.title("Navigation")
    
    # Logo ou image de profil pour l'utilisateur connecté
    st.markdown(f"""
    <div style='text-align: center; padding: 1rem 0;'>
        <div style='width: 80px; height: 80px; border-radius: 50%; background-color: #1E3A8A; display: inline-flex; justify-content: center; align-items: center; color: white; font-size: 28px; font-weight: bold;'>
            {st.session_state["username"][0].upper()}
        </div>
        <p style='margin-top: 10px; font-weight: bold;'>{st.session_state["username"]}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Menu de navigation
    page = st.radio(
        "Choisir une page",
        ["Importation des données", "Exploration des données", "Modélisation", "Prédiction"]
    )
    
    st.session_state.page = page
    
    # Infos sur l'application
    st.markdown("---")
    st.markdown("### À propos")
    st.markdown("""
    Cette application permet de prédire la récupération métallurgique en utilisant des techniques d'apprentissage automatique avancées.
    
    Développé par Didier Ouedraogo, Géologue et Data Scientist.
    """)
    
    # Bouton de déconnexion
    st.markdown("---")
    if st.button("Se déconnecter"):
        st.query_params["logout"] = "true"
        st.rerun()

# Titre principal
st.title("Application de prédiction géométallurgique")
st.markdown(f"<p class='data-info'>Connecté en tant que: <strong>{st.session_state['username']}</strong> | Date: {datetime.now().strftime('%d/%m/%Y')}</p>", unsafe_allow_html=True)

# Page d'importation des données
if page == "Importation des données":
    st.markdown("<h2 class='sub-header'>Importer vos données</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Choisir la source des données")
    
    data_source = st.radio(
        "Source des données",
        ["Télécharger un fichier", "Utiliser un exemple"]
    )
    
    if data_source == "Télécharger un fichier":
        uploaded_file = st.file_uploader("Choisir un fichier CSV ou Excel", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            try:
                # Déterminer le type de fichier et charger les données
                if uploaded_file.name.endswith('.csv'):
                    # Options pour les fichiers CSV
                    encoding_option = st.selectbox(
                        "Encodage du fichier",
                        ["utf-8", "latin-1", "iso-8859-1", "cp1252"]
                    )
                    separator_option = st.selectbox(
                        "Séparateur de colonnes",
                        [",", ";", "\t"]
                    )
                    
                    data = pd.read_csv(uploaded_file, encoding=encoding_option, sep=separator_option)
                    raw_data = data.copy()
                else:
                    data = pd.read_excel(uploaded_file)
                    raw_data = data.copy()
                
                # Stocker les données dans la session state
                st.session_state.data = data
                
                # Créer une version CSV des données pour le téléchargement
                csv_data = data.to_csv(index=False).encode('utf-8')
                st.session_state.csv_data = csv_data
                
                # Afficher un aperçu des données
                st.markdown("### Aperçu des données")
                st.dataframe(data.head())
                
                # Afficher des informations sur les données
                st.markdown("### Informations sur les données")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Nombre de lignes", data.shape[0])
                with col2:
                    st.metric("Nombre de colonnes", data.shape[1])
                with col3:
                    st.metric("Mémoire utilisée", f"{data.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
                
                # Types de données
                st.markdown("##### Types de données")
                type_counts = data.dtypes.value_counts().reset_index()
                type_counts.columns = ["Type de données", "Nombre de colonnes"]
                
                # Convertir les types de données pour l'affichage
                type_counts["Type de données"] = type_counts["Type de données"].astype(str)
                
                fig = px.bar(
                    type_counts, 
                    x="Type de données", 
                    y="Nombre de colonnes",
                    color="Type de données",
                    text="Nombre de colonnes"
                )
                fig.update_layout(xaxis_title="Type de données", yaxis_title="Nombre de colonnes")
                st.plotly_chart(fig, use_container_width=True)
                
                # Valeurs manquantes
                st.markdown("##### Valeurs manquantes")
                missing_values = data.isnull().sum().reset_index()
                missing_values.columns = ["Colonne", "Nombre de valeurs manquantes"]
                missing_values = missing_values[missing_values["Nombre de valeurs manquantes"] > 0]
                
                if not missing_values.empty:
                    missing_values["Pourcentage"] = missing_values["Nombre de valeurs manquantes"] / len(data) * 100
                    missing_values = missing_values.sort_values("Nombre de valeurs manquantes", ascending=False)
                    
                    fig = px.bar(
                        missing_values,
                        x="Colonne",
                        y="Pourcentage",
                        text=missing_values["Nombre de valeurs manquantes"].apply(lambda x: f"{x} valeurs"),
                        title="Pourcentage de valeurs manquantes par colonne"
                    )
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("Aucune valeur manquante détectée dans les données!")
                
                # Offrir des options pour traiter les valeurs manquantes
                if not missing_values.empty:
                    st.markdown("##### Traitement des valeurs manquantes")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fill_method = st.selectbox(
                            "Méthode de remplissage",
                            ["Moyenne", "Médiane", "Mode", "Valeur spécifique", "Supprimer les lignes"]
                        )
                    
                    with col2:
                        if fill_method == "Valeur spécifique":
                            fill_value = st.number_input("Valeur de remplacement", value=0.0)
                        else:
                            fill_value = None
                    
                    if st.button("Traiter les valeurs manquantes"):
                        if fill_method == "Supprimer les lignes":
                            data = data.dropna()
                            st.success(f"Valeurs manquantes traitées: {len(raw_data) - len(data)} lignes supprimées")
                        else:
                            for col in missing_values["Colonne"]:
                                if pd.api.types.is_numeric_dtype(data[col]):
                                    if fill_method == "Moyenne":
                                        data[col] = data[col].fillna(data[col].mean())
                                    elif fill_method == "Médiane":
                                        data[col] = data[col].fillna(data[col].median())
                                    elif fill_method == "Mode":
                                        data[col] = data[col].fillna(data[col].mode()[0])
                                    elif fill_method == "Valeur spécifique":
                                        data[col] = data[col].fillna(fill_value)
                                else:
                                    if fill_method == "Mode":
                                        data[col] = data[col].fillna(data[col].mode()[0])
                                    elif fill_method == "Valeur spécifique":
                                        data[col] = data[col].fillna(str(fill_value))
                                    else:
                                        data[col] = data[col].fillna("Unknown")
                            
                            st.success("Valeurs manquantes traitées avec succès!")
                        
                        # Mettre à jour les données dans la session state
                        st.session_state.data = data
                        
                        # Mettre à jour la version CSV des données
                        csv_data = data.to_csv(index=False).encode('utf-8')
                        st.session_state.csv_data = csv_data
                        
                        # Réafficher l'aperçu
                        st.markdown("### Aperçu des données mises à jour")
                        st.dataframe(data.head())
                
                # Option pour continuer vers l'exploration des données
                if st.button("Passer à l'exploration des données"):
                    st.session_state.page = "Exploration des données"
                    st.rerun()
                
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier: {e}")
                st.info("Conseil: Vérifiez le format de votre fichier et assurez-vous qu'il est bien structuré.")
    
    else:  # Utiliser un exemple
        dataset_option = st.selectbox(
            "Choisir un jeu de données d'exemple",
            ["Données de récupération Cu-Au", "Données de récupération Zn-Pb", "Données de flottation"]
        )
        
        if st.button("Charger l'exemple"):
            # Générer des données factices selon l'option choisie
            np.random.seed(42)  # Pour la reproductibilité
            
            if dataset_option == "Données de récupération Cu-Au":
                # Générer des données pour la récupération Cu-Au
                n_samples = 500
                
                # Paramètres d'entrée
                tenor_cu = np.random.normal(1.5, 0.5, n_samples)  # Teneur en Cu (%)
                tenor_au = np.random.normal(2.0, 0.8, n_samples)  # Teneur en Au (g/t)
                grind_size = np.random.normal(75, 15, n_samples)  # Taille de broyage (microns)
                ph_value = np.random.normal(10.5, 1.0, n_samples)  # pH
                residence_time = np.random.normal(15, 5, n_samples)  # Temps de résidence (min)
                reagent_dosage = np.random.normal(50, 10, n_samples)  # Dosage de réactif (g/t)
                pulp_density = np.random.normal(35, 5, n_samples)  # Densité de pulpe (%)
                temperature = np.random.normal(25, 3, n_samples)  # Température (°C)
                
                # Types de minerai
                ore_types = np.random.choice(['Oxyde', 'Sulfure', 'Mixte'], n_samples, p=[0.3, 0.5, 0.2])
                
                # Création d'une relation entre les entrées et la récupération
                recovery_base = (
                    0.4 * tenor_cu + 
                    0.2 * tenor_au - 
                    0.01 * grind_size + 
                    0.05 * ph_value + 
                    0.03 * residence_time +
                    0.02 * reagent_dosage +
                    0.01 * pulp_density -
                    0.02 * temperature
                )
                
                # Ajout d'un effet pour le type de minerai
                ore_type_effect = np.zeros(n_samples)
                ore_type_effect[ore_types == 'Oxyde'] = -5
                ore_type_effect[ore_types == 'Sulfure'] = 10
                ore_type_effect[ore_types == 'Mixte'] = 0
                
                # Ajout d'interactions non linéaires
                interactions = (
                    0.05 * tenor_cu * tenor_au +
                    0.02 * ph_value * reagent_dosage -
                    0.01 * grind_size * pulp_density
                )
                
                # Calcul de la récupération finale avec bruit
                recovery = 50 + recovery_base + ore_type_effect + interactions + np.random.normal(0, 5, n_samples)
                
                # Limiter la récupération entre 0 et 100%
                recovery = np.clip(recovery, 5, 99)
                
                # Créer le DataFrame
                data = pd.DataFrame({
                    'Teneur_Cu': tenor_cu,
                    'Teneur_Au': tenor_au,
                    'Taille_Broyage': grind_size,
                    'pH': ph_value,
                    'Temps_Residence': residence_time,
                    'Dosage_Reactif': reagent_dosage,
                    'Densite_Pulpe': pulp_density,
                    'Temperature': temperature,
                    'Type_Minerai': ore_types,
                    'Recuperation': recovery
                })
                
                # Ajouter quelques coordonnées factices pour l'échantillonnage
                data['Longitude'] = np.random.uniform(-5.0, 5.0, n_samples)
                data['Latitude'] = np.random.uniform(-5.0, 5.0, n_samples)
                data['Profondeur'] = np.random.uniform(0, 500, n_samples)
                
            elif dataset_option == "Données de récupération Zn-Pb":
                # Générer des données pour la récupération Zn-Pb
                n_samples = 500
                
                # Paramètres d'entrée
                tenor_zn = np.random.normal(6.0, 1.5, n_samples)  # Teneur en Zn (%)
                tenor_pb = np.random.normal(3.0, 1.0, n_samples)  # Teneur en Pb (%)
                grind_size = np.random.normal(65, 10, n_samples)  # Taille de broyage (microns)
                ph_value = np.random.normal(9.0, 0.8, n_samples)  # pH
                residence_time = np.random.normal(12, 3, n_samples)  # Temps de résidence (min)
                collector_dosage = np.random.normal(40, 8, n_samples)  # Dosage de collecteur (g/t)
                frother_dosage = np.random.normal(20, 5, n_samples)  # Dosage de moussant (g/t)
                pulp_density = np.random.normal(32, 4, n_samples)  # Densité de pulpe (%)
                
                # Types de minerai
                ore_types = np.random.choice(['Primaire', 'Secondaire', 'Transitionnel'], n_samples, p=[0.4, 0.4, 0.2])
                
                # Création d'une relation entre les entrées et la récupération
                recovery_base = (
                    0.35 * tenor_zn + 
                    0.25 * tenor_pb - 
                    0.02 * grind_size + 
                    0.06 * ph_value + 
                    0.04 * residence_time +
                    0.03 * collector_dosage +
                    0.02 * frother_dosage +
                    0.01 * pulp_density
                )
                
                # Ajout d'un effet pour le type de minerai
                ore_type_effect = np.zeros(n_samples)
                ore_type_effect[ore_types == 'Primaire'] = 8
                ore_type_effect[ore_types == 'Secondaire'] = -3
                ore_type_effect[ore_types == 'Transitionnel'] = 0
                
                # Ajout d'interactions non linéaires
                interactions = (
                    0.04 * tenor_zn * tenor_pb +
                    0.03 * ph_value * collector_dosage -
                    0.02 * grind_size * pulp_density
                )
                
                # Calcul de la récupération finale avec bruit
                recovery = 55 + recovery_base + ore_type_effect + interactions + np.random.normal(0, 6, n_samples)
                
                # Limiter la récupération entre 0 et 100%
                recovery = np.clip(recovery, 10, 98)
                
                # Créer le DataFrame
                data = pd.DataFrame({
                    'Teneur_Zn': tenor_zn,
                    'Teneur_Pb': tenor_pb,
                    'Taille_Broyage': grind_size,
                    'pH': ph_value,
                    'Temps_Residence': residence_time,
                    'Dosage_Collecteur': collector_dosage,
                    'Dosage_Moussant': frother_dosage,
                    'Densite_Pulpe': pulp_density,
                    'Type_Minerai': ore_types,
                    'Recuperation': recovery
                })
                
                # Ajouter quelques coordonnées factices pour l'échantillonnage
                data['Longitude'] = np.random.uniform(-5.0, 5.0, n_samples)
                data['Latitude'] = np.random.uniform(-5.0, 5.0, n_samples)
                data['Profondeur'] = np.random.uniform(0, 300, n_samples)
                
            else:  # Données de flottation
                # Générer des données pour la flottation
                n_samples = 500
                
                # Paramètres d'entrée
                tenor_metal = np.random.normal(2.5, 0.8, n_samples)  # Teneur en métal (%)
                particle_size = np.random.normal(50, 10, n_samples)  # Taille des particules (microns)
                collector_conc = np.random.normal(100, 20, n_samples)  # Concentration du collecteur (g/t)
                frother_conc = np.random.normal(30, 8, n_samples)  # Concentration du moussant (g/t)
                air_flow_rate = np.random.normal(10, 2, n_samples)  # Débit d'air (L/min)
                impeller_speed = np.random.normal(1200, 150, n_samples)  # Vitesse de l'agitateur (rpm)
                slurry_density = np.random.normal(30, 5, n_samples)  # Densité de la pulpe (%)
                ph_value = np.random.normal(8.5, 0.5, n_samples)  # pH
                
                # Types de circuit
                circuit_types = np.random.choice(['Rougher', 'Cleaner', 'Scavenger'], n_samples, p=[0.6, 0.3, 0.1])
                
                # Création d'une relation entre les entrées et le taux de récupération
                recovery_base = (
                    0.3 * tenor_metal - 
                    0.02 * particle_size + 
                    0.01 * collector_conc + 
                    0.005 * frother_conc +
                    0.02 * air_flow_rate +
                    0.01 * (impeller_speed / 100) +
                    0.005 * slurry_density +
                    0.03 * ph_value
                )
                
                # Ajout d'un effet pour le type de circuit
                circuit_effect = np.zeros(n_samples)
                circuit_effect[circuit_types == 'Rougher'] = 0
                circuit_effect[circuit_types == 'Cleaner'] = 10
                circuit_effect[circuit_types == 'Scavenger'] = -15
                
                # Ajout d'interactions non linéaires
                interactions = (
                    0.03 * tenor_metal * (collector_conc / 100) +
                    0.02 * ph_value * (collector_conc / 100) -
                    0.01 * particle_size * (air_flow_rate / 10)
                )
                
                # Calcul du taux de récupération final avec bruit
                recovery = 60 + recovery_base + circuit_effect + interactions + np.random.normal(0, 5, n_samples)
                
                # Limiter la récupération entre 0 et 100%
                recovery = np.clip(recovery, 15, 97)
                
                # Créer le DataFrame
                data = pd.DataFrame({
                    'Teneur_Metal': tenor_metal,
                    'Taille_Particule': particle_size,
                    'Concentration_Collecteur': collector_conc,
                    'Concentration_Moussant': frother_conc,
                    'Debit_Air': air_flow_rate,
                    'Vitesse_Agitateur': impeller_speed,
                    'Densite_Pulpe': slurry_density,
                    'pH': ph_value,
                    'Type_Circuit': circuit_types,
                    'Recuperation': recovery
                })
                
                # Ajouter quelques coordonnées factices pour l'échantillonnage
                data['X_Position'] = np.random.uniform(-10.0, 10.0, n_samples)
                data['Y_Position'] = np.random.uniform(-10.0, 10.0, n_samples)
                data['Z_Position'] = np.random.uniform(0, 20, n_samples)
            
            # Stocker les données dans la session state
            st.session_state.data = data
            
            # Créer une version CSV des données pour le téléchargement
            csv_data = data.to_csv(index=False).encode('utf-8')
            st.session_state.csv_data = csv_data
            
            # Afficher un aperçu des données
            st.markdown("### Aperçu des données")
            st.dataframe(data.head())
            
            # Afficher des informations sur les données
            st.markdown("### Informations sur les données")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Nombre de lignes", data.shape[0])
            with col2:
                st.metric("Nombre de colonnes", data.shape[1])
            
            # Option pour continuer vers l'exploration des données
            if st.button("Passer à l'exploration des données"):
                st.session_state.page = "Exploration des données"
                st.rerun()
    
    # Bouton de téléchargement des données si elles sont disponibles
    if st.session_state.csv_data is not None:
        st.download_button(
            label="Télécharger les données (CSV)",
            data=st.session_state.csv_data,
            file_name="donnees_metallurgiques.csv",
            mime="text/csv"
        )
    st.markdown("</div>", unsafe_allow_html=True)

# Page d'exploration des données
elif page == "Exploration des données":
    st.markdown("<h2 class='sub-header'>Exploration et Analyse des Données</h2>", unsafe_allow_html=True)
    
    # Vérifier si des données ont été importées
    if st.session_state.data is None:
        st.warning("Aucune donnée n'a été importée. Veuillez aller à la page 'Importation des données'.")
        if st.button("Aller à la page d'importation"):
            st.session_state.page = "Importation des données"
            st.rerun()
    else:
        data = st.session_state.data
        
        # Options d'analyse
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Statistiques descriptives")
        
        # Afficher les statistiques descriptives
        with st.expander("Afficher les statistiques descriptives"):
            st.write(data.describe().T)
            
            # Option pour visualiser la distribution de chaque variable numérique
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if numeric_cols:
                selected_var = st.selectbox("Choisir une variable pour visualiser sa distribution", numeric_cols)
                
                fig = px.histogram(
                    data, 
                    x=selected_var, 
                    marginal="box", 
                    title=f"Distribution de {selected_var}"
                )
                st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Analyse de corrélation
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Analyse de corrélation")
        
        # Sélection des colonnes numériques
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        
        if numeric_data.shape[1] > 1:  # Au moins 2 colonnes numériques sont nécessaires pour la corrélation
            # Options de visualisation
            corr_method = st.radio(
                "Méthode de corrélation",
                ["Pearson", "Spearman"],
                horizontal=True
            )
            
            # Calculer la matrice de corrélation
            if corr_method == "Pearson":
                corr_matrix = numeric_data.corr(method='pearson')
            else:
                corr_matrix = numeric_data.corr(method='spearman')
            
            # Afficher la matrice de corrélation
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale='RdBu_r',
                aspect="auto",
                title="Matrice de corrélation"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Sélectionner les variables pour le nuage de points
            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox("Variable X", numeric_data.columns)
            with col2:
                y_var = st.selectbox("Variable Y", numeric_data.columns, index=min(1, len(numeric_data.columns)-1))
            
            # Ajouter une option de coloration
            color_var = st.selectbox("Colorer par", ["Aucune"] + data.columns.tolist())
            
            # Créer le nuage de points
            if color_var == "Aucune":
                fig = px.scatter(
                    data,
                    x=x_var,
                    y=y_var,
                    title=f"{y_var} vs {x_var}",
                    trendline="ols"
                )
            else:
                fig = px.scatter(
                    data,
                    x=x_var,
                    y=y_var,
                    color=color_var,
                    title=f"{y_var} vs {x_var} (coloré par {color_var})"
                )
                
                # Ajouter une ligne de tendance si la variable de coloration est numérique
                if data[color_var].dtype in ['float64', 'int64']:
                    fig.update_traces(marker=dict(size=8))
                    
                    # Ajouter une ligne de tendance globale
                    try:
                        import statsmodels.api as sm
                        from statsmodels.formula.api import ols
                        
                        # Formulation du modèle
                        model = ols(f'"{y_var}" ~ "{x_var}"', data=data).fit()
                        
                        # Générer des points pour la ligne de tendance
                        x_range = np.linspace(data[x_var].min(), data[x_var].max(), 100)
                        y_pred = model.params[0] + model.params[1] * x_range
                        
                        # Ajouter la ligne de tendance
                        fig.add_traces(
                            go.Scatter(
                                x=x_range, 
                                y=y_pred, 
                                mode='lines',
                                name='Tendance générale',
                                line=dict(color='black', dash='dash')
                            )
                        )
                    except:
                        pass
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Afficher les variables les plus corrélées à une variable spécifique
            st.markdown("#### Variables les plus corrélées")
            target_var = st.selectbox("Choisir une variable cible", numeric_data.columns)
            
            # Obtenir et trier les corrélations pour la variable cible
            correlations = corr_matrix[target_var].sort_values(ascending=False)
            correlations = correlations.drop(target_var)  # Supprimer l'auto-corrélation
            
            # Afficher un graphique des corrélations
            fig = px.bar(
                x=correlations.index,
                y=correlations.values,
                title=f"Corrélations avec {target_var}",
                labels={"x": "Variables", "y": f"Corrélation avec {target_var}"},
                color=correlations.values,
                color_continuous_scale='RdBu_r'
            )
            
            # Mettre à jour la mise en page pour améliorer la lisibilité
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Option pour le diagnostic de multicolinéarité (VIF)
            if st.checkbox("Effectuer un diagnostic de multicolinéarité (VIF)"):
                try:
                    # Sélectionner uniquement les colonnes numériques
                    X = numeric_data.dropna()
                    
                    # Ajouter une constante pour l'intercepte
                    X = sm.add_constant(X)
                    
                    # Calculer les VIF pour chaque variable
                    vif_data = pd.DataFrame()
                    vif_data["Variable"] = X.columns
                    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                    
                    # Supprimer la constante
                    vif_data = vif_data[vif_data["Variable"] != "const"]
                    
                    # Trier par VIF
                    vif_data = vif_data.sort_values("VIF", ascending=False)
                    
                    # Afficher les VIF
                    st.write("Facteurs d'Inflation de la Variance (VIF):")
                    st.write("VIF > 10 indique une forte multicolinéarité")
                    
                    fig = px.bar(
                        vif_data,
                        x="Variable",
                        y="VIF",
                        title="Facteurs d'Inflation de la Variance (VIF)",
                        color="VIF",
                        color_continuous_scale="Viridis"
                    )
                    fig.add_hline(y=10, line_dash="dash", line_color="red", annotation_text="Seuil de forte multicolinéarité")
                    fig.add_hline(y=5, line_dash="dash", line_color="orange", annotation_text="Seuil de multicolinéarité modérée")
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommandations basées sur les VIF
                    high_vif = vif_data[vif_data["VIF"] > 10]
                    if not high_vif.empty:
                        st.warning(f"Variables avec forte multicolinéarité (VIF > 10): {', '.join(high_vif['Variable'].tolist())}")
                        st.write("Recommandation: Envisagez de retirer certaines de ces variables ou d'utiliser l'ACP.")
                except Exception as e:
                    st.error(f"Erreur lors du calcul des VIF: {e}")
        else:
            st.warning("Au moins 2 colonnes numériques sont nécessaires pour l'analyse de corrélation.")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Visualisations avancées
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Visualisations avancées")
        
        # Types de visualisations disponibles
        viz_type = st.selectbox(
            "Type de visualisation",
            ["Boîtes à moustaches", "Violin Plots", "Pairplot", "3D Scatter Plot", "Heatmap (grille)", "Mappage des données"]
        )
        
        if viz_type == "Boîtes à moustaches":
            # Options pour les boîtes à moustaches
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if numeric_cols and categorical_cols:
                y_var = st.selectbox("Variable numérique (y)", numeric_cols)
                x_var = st.selectbox("Variable catégorielle (x)", categorical_cols)
                
                fig = px.box(
                    data,
                    x=x_var,
                    y=y_var,
                    color=x_var,
                    title=f"Boîtes à moustaches de {y_var} par {x_var}",
                    points="all"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Vous avez besoin d'au moins une variable numérique et une variable catégorielle pour cette visualisation.")
        
        elif viz_type == "Violin Plots":
            # Options pour les violin plots
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if numeric_cols and categorical_cols:
                y_var = st.selectbox("Variable numérique (y)", numeric_cols)
                x_var = st.selectbox("Variable catégorielle (x)", categorical_cols)
                
                fig = px.violin(
                    data,
                    x=x_var,
                    y=y_var,
                    color=x_var,
                    box=True,
                    points="all",
                    title=f"Violin Plots de {y_var} par {x_var}"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Vous avez besoin d'au moins une variable numérique et une variable catégorielle pour cette visualisation.")
        
        elif viz_type == "Pairplot":
            # Options pour le pairplot
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            if len(numeric_cols) >= 2:
                # Sélection des variables à inclure (max 5 pour éviter les problèmes de performance)
                selected_vars = st.multiselect(
                    "Sélectionner les variables (max 5 recommandé)",
                    numeric_cols,
                    default=numeric_cols[:min(4, len(numeric_cols))]
                )
                
                if len(selected_vars) >= 2:
                    # Option pour la coloration
                    color_var = st.selectbox("Colorer par", ["Aucune"] + data.columns.tolist())
                    
                    if color_var == "Aucune":
                        fig = px.scatter_matrix(
                            data,
                            dimensions=selected_vars,
                            title="Matrice de nuages de points"
                        )
                    else:
                        fig = px.scatter_matrix(
                            data,
                            dimensions=selected_vars,
                            color=color_var,
                            title=f"Matrice de nuages de points (coloré par {color_var})"
                        )
                    
                    fig.update_traces(diagonal_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Veuillez sélectionner au moins 2 variables.")
            else:
                st.warning("Vous avez besoin d'au moins 2 variables numériques pour cette visualisation.")
        
        elif viz_type == "3D Scatter Plot":
            # Options pour le scatter plot 3D
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            if len(numeric_cols) >= 3:
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_var = st.selectbox("Variable X", numeric_cols)
                with col2:
                    y_var = st.selectbox("Variable Y", numeric_cols, index=min(1, len(numeric_cols)-1))
                with col3:
                    z_var = st.selectbox("Variable Z", numeric_cols, index=min(2, len(numeric_cols)-1))
                
                # Option pour la coloration
                color_var = st.selectbox("Colorer par", ["Aucune"] + data.columns.tolist())
                
                if color_var == "Aucune":
                    fig = px.scatter_3d(
                        data,
                        x=x_var,
                        y=y_var,
                        z=z_var,
                        title=f"Nuage de points 3D: {x_var} vs {y_var} vs {z_var}"
                    )
                else:
                    fig = px.scatter_3d(
                        data,
                        x=x_var,
                        y=y_var,
                        z=z_var,
                        color=color_var,
                        title=f"Nuage de points 3D: {x_var} vs {y_var} vs {z_var} (coloré par {color_var})"
                    )
                
                fig.update_layout(height=700)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Vous avez besoin d'au moins 3 variables numériques pour cette visualisation.")
        
        elif viz_type == "Heatmap (grille)":
            # Options pour la heatmap
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if numeric_cols and len(categorical_cols) >= 2:
                z_var = st.selectbox("Variable numérique (valeur)", numeric_cols)
                x_var = st.selectbox("Variable catégorielle (x)", categorical_cols)
                y_var = st.selectbox("Variable catégorielle (y)", categorical_cols, index=min(1, len(categorical_cols)-1))
                
                # Agréger les données
                agg_func = st.selectbox("Fonction d'agrégation", ["Moyenne", "Médiane", "Somme", "Nombre"])
                
                if agg_func == "Moyenne":
                    df_pivot = data.pivot_table(values=z_var, index=y_var, columns=x_var, aggfunc='mean')
                elif agg_func == "Médiane":
                    df_pivot = data.pivot_table(values=z_var, index=y_var, columns=x_var, aggfunc='median')
                elif agg_func == "Somme":
                    df_pivot = data.pivot_table(values=z_var, index=y_var, columns=x_var, aggfunc='sum')
                else:  # Nombre
                    df_pivot = data.pivot_table(values=z_var, index=y_var, columns=x_var, aggfunc='count')
                
                # Afficher la heatmap
                fig = px.imshow(
                    df_pivot,
                    color_continuous_scale=px.colors.sequential.Viridis,
                    text_auto=True,
                    aspect="auto",
                    title=f"{agg_func} de {z_var} par {x_var} et {y_var}"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Vous avez besoin d'au moins une variable numérique et deux variables catégorielles pour cette visualisation.")
        
        elif viz_type == "Mappage des données":
            # Vérifier si les coordonnées spatiales sont disponibles
            spatial_columns = []
            for pair in [['Longitude', 'Latitude'], ['X', 'Y'], ['X_Position', 'Y_Position'], ['E', 'N']]:
                if all(col in data.columns for col in pair):
                    spatial_columns = pair
                    break
            
            if spatial_columns:
                st.markdown("#### Visualisation spatiale des données")
                
                # Sélectionner la variable pour la coloration
                color_var = st.selectbox(
                    "Colorer par",
                    data.columns.tolist(),
                    index=0
                )
                
                # Créer une carte avec folium
                if spatial_columns[0] in ['Longitude', 'E'] and spatial_columns[1] in ['Latitude', 'N']:
                    # Calculer le centre de la carte
                    center_lat = data[spatial_columns[1]].mean()
                    center_lon = data[spatial_columns[0]].mean()
                    
                    # Créer la carte
                    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
                    
                    # Déterminer la palette de couleurs en fonction du type de variable
                    if data[color_var].dtype in ['float64', 'int64']:
                        # Palette numérique
                        color_scale = cm.linear.Viridis_09.scale(
                            data[color_var].min(),
                            data[color_var].max()
                        )
                        
                        # Ajouter les points à la carte
                        for idx, row in data.iterrows():
                            folium.CircleMarker(
                                location=[row[spatial_columns[1]], row[spatial_columns[0]]],
                                radius=5,
                                color=color_scale(row[color_var]),
                                fill=True,
                                fill_color=color_scale(row[color_var]),
                                tooltip=f"{color_var}: {row[color_var]}"
                            ).add_to(m)
                        
                        # Ajouter la légende
                        color_scale.add_to(m)
                    else:
                        # Palette catégorielle
                        categories = data[color_var].unique()
                        color_map = {}
                        
                        # Générer des couleurs pour chaque catégorie
                        import random
                        for category in categories:
                            color_map[category] = '#%02X%02X%02X' % (
                                random.randint(0, 255),
                                random.randint(0, 255),
                                random.randint(0, 255)
                            )
                        
                        # Ajouter les points à la carte
                        for idx, row in data.iterrows():
                            folium.CircleMarker(
                                location=[row[spatial_columns[1]], row[spatial_columns[0]]],
                                radius=5,
                                color=color_map[row[color_var]],
                                fill=True,
                                fill_color=color_map[row[color_var]],
                                tooltip=f"{color_var}: {row[color_var]}"
                            ).add_to(m)
                        
                        # Ajouter la légende
                        from branca.element import Template, MacroElement
                        
                        template = """
                        {% macro html(this, kwargs) %}
                        <div style="position: fixed; 
                            bottom: 50px; right: 50px; 
                            border:2px solid grey; z-index:9999; 
                            background-color:white;
                            padding: 10px;
                            border-radius: 5px;
                            ">
                            <h4>Légende</h4>
                            {% for category, color in color_map.items() %}
                            <div>
                                <span style="background-color: {{ color }}; 
                                    display: inline-block; 
                                    width: 12px; height: 12px;
                                    border-radius: 50%;
                                    margin-right: 5px;"></span>
                                {{ category }}
                            </div>
                            {% endfor %}
                        </div>
                        {% endmacro %}
                        """
                        
                        macro = MacroElement()
                        macro._template = Template(template)
                        macro.color_map = color_map
                        m.add_child(macro)
                    
                    # Afficher la carte
                    folium_static(m)
                else:
                    # Pour les systèmes de coordonnées locaux, utiliser un scatter plot
                    fig = px.scatter(
                        data,
                        x=spatial_columns[0],
                        y=spatial_columns[1],
                        color=color_var,
                        title=f"Distribution spatiale colorée par {color_var}",
                        opacity=0.7
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Aucune coordonnée spatiale détectée dans les données. Recherche des colonnes comme: Longitude/Latitude, X/Y, E/N, X_Position/Y_Position.")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Sélection des features et de la cible pour la modélisation
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Sélection des variables pour la modélisation")
        
        # Suggestion automatique pour la cible (variable de récupération)
        target_suggestions = [col for col in data.columns if 'recup' in col.lower() or 'recovery' in col.lower()]
        if target_suggestions:
            default_target = target_suggestions[0]
        else:
            default_target = data.select_dtypes(include=['float64', 'int64']).columns[-1]
        
        # Sélection de la variable cible
        target = st.selectbox(
            "Sélectionner la variable cible (récupération métallurgique)",
            data.columns.tolist(),
            index=data.columns.tolist().index(default_target) if default_target in data.columns else 0
        )
        
        # Exclure la variable cible et d'autres variables non pertinentes des features potentielles
        exclude_cols = [target]
        exclude_cols.extend([col for col in data.columns if 'id' in col.lower() or 'date' in col.lower()])
        potential_features = [col for col in data.columns if col not in exclude_cols]
        
        # Sélection des features
        features = st.multiselect(
            "Sélectionner les variables prédictives",
            potential_features,
            default=potential_features[:min(5, len(potential_features))]
        )
        
        if features and target:
            # Stocker les features et la cible dans la session state
            st.session_state.features = features
            st.session_state.target = target
            
            # Aperçu du jeu de données sélectionné
            st.markdown("#### Aperçu des variables sélectionnées")
            st.dataframe(data[features + [target]].head())
            
            # Visualiser la distribution de la variable cible
            fig = px.histogram(
                data,
                x=target,
                marginal="box",
                title=f"Distribution de la variable cible: {target}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Option pour l'augmentation de données
            if st.checkbox("Voulez-vous effectuer une augmentation de données?"):
                st.markdown("#### Augmentation de données")
                
                # Sélectionner le nombre d'échantillons à générer
                n_samples = st.slider("Nombre d'échantillons à générer", 100, 1000, 500)
                
                if st.button("Générer des données supplémentaires"):
                    with st.spinner("Génération des données en cours..."):
                        try:
                            # Extraire les données originales
                            X_orig = data[features]
                            y_orig = data[target]
                            
                            # Vérifier s'il y a des variables catégorielles
                            cat_features = X_orig.select_dtypes(include=['object', 'category']).columns.tolist()
                            
                            if cat_features:
                                # Traiter les variables catégorielles
                                cat_encoder = OneHotEncoder(sparse=False, drop='first')
                                X_cat = cat_encoder.fit_transform(X_orig[cat_features])
                                
                                # Extraire les variables numériques
                                num_features = X_orig.select_dtypes(include=['float64', 'int64']).columns.tolist()
                                X_num = X_orig[num_features].values if num_features else np.array([]).reshape(X_orig.shape[0], 0)
                                
                                # Combiner les variables numériques et catégorielles encodées
                                X_combined = np.hstack([X_num, X_cat])
                                
                                # Appliquer SMOTE pour générer de nouvelles données
                                sm = SMOTE(random_state=42, k_neighbors=min(5, len(X_orig)-1))
                                X_resampled, y_resampled = sm.fit_resample(X_combined, y_orig)
                                
                                # Sélectionner uniquement les nouveaux échantillons générés
                                X_new = X_resampled[len(X_orig):]
                                y_new = y_resampled[len(y_orig):]
                                
                                # Reconvertir les variables catégorielles
                                if num_features:
                                    X_num_new = X_new[:, :len(num_features)]
                                    X_cat_new = X_new[:, len(num_features):]
                                else:
                                    X_num_new = np.array([]).reshape(X_new.shape[0], 0)
                                    X_cat_new = X_new
                                
                                # Créer un nouveau DataFrame pour les variables numériques
                                augmented_data_num = pd.DataFrame(X_num_new, columns=num_features)
                                
                                # Inverser la transformation pour les variables catégorielles
                                cat_names = []
                                for i, cat in enumerate(cat_features):
                                    cat_names.extend([f"{cat}_{cls}" for cls in cat_encoder.categories_[i][1:]])
                                
                                # Convertir les valeurs one-hot encodées en catégories
                                augmented_data_cat = pd.DataFrame()
                                start_idx = 0
                                
                                for i, cat in enumerate(cat_features):
                                    n_categories = len(cat_encoder.categories_[i]) - 1  # -1 car une catégorie est supprimée (drop='first')
                                    cat_values = []
                                    
                                    # Valeurs encodées pour cette variable
                                    if n_categories > 0:
                                        cat_encoded = X_cat_new[:, start_idx:start_idx+n_categories]
                                        start_idx += n_categories
                                        
                                        # Convertir en indices de catégories
                                        for row in cat_encoded:
                                            # Si tous les bits sont 0, c'est la première catégorie (celle qui a été "dropped")
                                            if np.all(row == 0):
                                                cat_idx = 0
                                            else:
                                                # Sinon, trouver l'index du bit 1 et ajouter 1 (car la première catégorie est "dropped")
                                                cat_idx = np.argmax(row) + 1
                                            
                                            cat_values.append(cat_encoder.categories_[i][cat_idx])
                                    else:
                                        # Si une seule catégorie, utiliser cette catégorie pour tous les échantillons
                                        cat_values = [cat_encoder.categories_[i][0]] * len(X_cat_new)
                                    
                                    augmented_data_cat[cat] = cat_values
                                
                                # Combiner les données numériques et catégorielles
                                augmented_data = pd.concat([augmented_data_num, augmented_data_cat], axis=1)
                                
                                # Ajouter la variable cible
                                augmented_data[target] = y_new
                                
                                # Limiter au nombre d'échantillons demandé
                                augmented_data = augmented_data.iloc[:min(n_samples, len(augmented_data))]
                                
                            else:
                                # Cas plus simple: uniquement des variables numériques
                                X_orig_values = X_orig.values
                                
                                # Appliquer SMOTE
                                sm = SMOTE(random_state=42, k_neighbors=min(5, len(X_orig)-1))
                                X_resampled, y_resampled = sm.fit_resample(X_orig_values, y_orig)
                                
                                # Sélectionner uniquement les nouveaux échantillons générés
                                X_new = X_resampled[len(X_orig):]
                                y_new = y_resampled[len(y_orig):]
                                
                                # Créer un DataFrame pour les données augmentées
                                augmented_data = pd.DataFrame(X_new, columns=features)
                                augmented_data[target] = y_new
                                
                                # Limiter au nombre d'échantillons demandé
                                augmented_data = augmented_data.iloc[:min(n_samples, len(augmented_data))]
                            
                            # Afficher un aperçu des données augmentées
                            st.markdown("#### Aperçu des données augmentées")
                            st.dataframe(augmented_data.head())
                            
                            # Option pour combiner les données originales et augmentées
                            combine_option = st.radio(
                                "Comment utiliser les données augmentées?",
                                ["Combine avec les données originales", "Utiliser uniquement les données augmentées"]
                            )
                            
                            if combine_option == "Combine avec les données originales":
                                # Combiner les données
                                combined_data = pd.concat([data[features + [target]], augmented_data], ignore_index=True)
                                st.session_state.data = combined_data
                                st.success(f"Données augmentées! Nombre total d'échantillons: {len(combined_data)}")
                            else:
                                # Utiliser uniquement les données augmentées
                                st.session_state.data = augmented_data
                                st.success(f"Données remplacées par {len(augmented_data)} échantillons augmentés")
                            
                            # Mettre à jour les CSV data pour le téléchargement
                            csv_data = st.session_state.data.to_csv(index=False).encode('utf-8')
                            st.session_state.csv_data = csv_data
                            
                            # Bouton de téléchargement des données augmentées
                            st.download_button(
                                label="Télécharger les données augmentées (CSV)",
                                data=csv_data,
                                file_name="donnees_augmentees.csv",
                                mime="text/csv"
                            )
                        except Exception as e:
                            st.error(f"Erreur lors de l'augmentation des données: {e}")
                            st.info("Conseil: Assurez-vous que vos données sont appropriées pour SMOTE. Les variables catégorielles avec trop de catégories peuvent causer des problèmes.")
            
            # Bouton pour passer à la modélisation
            if st.button("Passer à la modélisation"):
                # Préparer les données pour la modélisation
                X = data[features]
                y = data[target]
                
                # Division des données en ensembles d'entraînement et de test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Stocker dans la session state
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                
                # Passer à la page de modélisation
                st.session_state.page = "Modélisation"
                st.rerun()
        else:
            st.warning("Veuillez sélectionner au moins une variable prédictive et une variable cible.")
        st.markdown("</div>", unsafe_allow_html=True)

# Page de modélisation
elif page == "Modélisation":
    st.markdown("<h2 class='sub-header'>Modélisation Prédictive</h2>", unsafe_allow_html=True)
    
    # Vérifier si les données et les variables ont été définies
    if st.session_state.data is None:
        st.warning("Aucune donnée n'a été importée. Veuillez aller à la page 'Importation des données'.")
        if st.button("Aller à la page d'importation"):
            st.session_state.page = "Importation des données"
            st.rerun()
    elif st.session_state.features is None or st.session_state.target is None:
        st.warning("Variables prédictives et cible non définies. Veuillez aller à la page 'Exploration des données'.")
        if st.button("Aller à la page d'exploration"):
            st.session_state.page = "Exploration des données"
            st.rerun()
    else:
        # Récupérer les données pertinentes
        data = st.session_state.data
        features = st.session_state.features
        target = st.session_state.target
        
        # Si les données d'entraînement/test ne sont pas encore divisées, le faire maintenant
        if st.session_state.X_train is None:
            X = data[features]
            y = data[target]
            
            # Division des données en ensembles d'entraînement et de test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Stocker dans la session state
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
        else:
            X_train = st.session_state.X_train
            X_test = st.session_state.X_test
            y_train = st.session_state.y_train
            y_test = st.session_state.y_test
        
        # Afficher des informations sur les données
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Données pour la modélisation")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Échantillons d'entraînement", X_train.shape[0])
        with col2:
            st.metric("Échantillons de test", X_test.shape[0])
        with col3:
            st.metric("Variables prédictives", len(features))
        
        # Afficher les features sélectionnées
        st.markdown("#### Variables prédictives sélectionnées")
        st.write(", ".join(features))
        
        # Afficher la cible
        st.markdown("#### Variable cible")
        st.write(target)
        
        # Option pour modifier la division train/test
        if st.checkbox("Modifier la division entraînement/test"):
            test_size = st.slider("Proportion de données de test (%)", 10, 40, 20)
            
            if st.button("Appliquer la nouvelle division"):
                X = data[features]
                y = data[target]
                
                # Nouvelle division
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
                
                # Mettre à jour la session state
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                
                st.success(f"Division mise à jour: {100-test_size}% entraînement, {test_size}% test")
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Configuration du modèle
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Configuration du modèle prédictif")
        
        # Type de modèle
        model_type = st.selectbox(
            "Sélectionner le type de modèle",
            [
                "Régression Linéaire",
                "Ridge",
                "Lasso",
                "ElasticNet",
                "SVR",
                "Arbre de Décision",
                "Random Forest",
                "Gradient Boosting",
                "XGBoost",
                "Réseau de Neurones"
            ]
        )
        
        # Vérifier si TensorFlow est disponible pour les réseaux de neurones
        if model_type == "Réseau de Neurones" and not TENSORFLOW_AVAILABLE:
            st.warning("TensorFlow n'est pas installé. Veuillez installer TensorFlow pour utiliser les réseaux de neurones ou choisir un autre modèle.")
            model_type = "Random Forest"  # Fallback à Random Forest
        
        # Configuration du prétraitement
        st.markdown("#### Prétraitement des données")
        
        # Identifier les variables catégorielles et numériques
        categorical_features = []
        numeric_features = []
        
        for feature in features:
            if data[feature].dtype == 'object' or data[feature].dtype.name == 'category':
                categorical_features.append(feature)
            else:
                numeric_features.append(feature)
        
        # Options de scaling pour les variables numériques
        scaler_option = st.selectbox(
            "Méthode de mise à l'échelle pour les variables numériques",
            ["StandardScaler", "MinMaxScaler", "RobustScaler", "Aucune"]
        )
        
        # Options d'encodage pour les variables catégorielles
        if categorical_features:
            encoding_method = st.selectbox(
                "Méthode d'encodage pour les variables catégorielles",
                ["One-Hot Encoding", "Ordinal Encoding"]
            )
        else:
            encoding_method = None
        
        # Configuration avancée selon le type de modèle
        st.markdown("#### Paramètres du modèle")
        
        if model_type in ["Ridge", "Lasso", "ElasticNet"]:
            alpha = st.slider("Alpha (régularisation)", 0.01, 10.0, 1.0)
        
        if model_type == "ElasticNet":
            l1_ratio = st.slider("Ratio L1 (Lasso vs Ridge)", 0.0, 1.0, 0.5)
        
        if model_type == "SVR":
            kernel = st.selectbox("Noyau", ["linear", "poly", "rbf", "sigmoid"])
            C = st.slider("C (régularisation)", 0.1, 10.0, 1.0)
        
        if model_type in ["Random Forest", "Gradient Boosting"]:
            n_estimators = st.slider("Nombre d'estimateurs", 10, 500, 100)
            max_depth = st.slider("Profondeur maximale", 1, 30, 10)
        
        if model_type == "XGBoost":
            try:
                import xgboost as xgb
                n_estimators_xgb = st.slider("Nombre d'estimateurs", 10, 500, 100)
                max_depth_xgb = st.slider("Profondeur maximale", 1, 15, 6)
                learning_rate_xgb = st.slider("Taux d'apprentissage", 0.01, 0.3, 0.1)
                subsample_xgb = st.slider("Sous-échantillonnage", 0.5, 1.0, 0.8)
                XGBOOST_AVAILABLE = True
            except ImportError:
                st.warning("XGBoost n'est pas installé. Utilisation de Gradient Boosting à la place.")
                model_type = "Gradient Boosting"
                XGBOOST_AVAILABLE = False
        
        if model_type == "Réseau de Neurones" and TENSORFLOW_AVAILABLE:
            # Configuration du réseau de neurones
            n_layers = st.slider("Nombre de couches cachées", 1, 5, 2)
            
            layer_sizes = []
            for i in range(n_layers):
                layer_sizes.append(st.slider(f"Neurones dans la couche {i+1}", 4, 128, 32))
            
            dropout_rate = st.slider("Taux de dropout", 0.0, 0.5, 0.2)
            learning_rate = st.slider("Taux d'apprentissage", 0.0001, 0.01, 0.001, step=0.0001)
            batch_size = st.slider("Taille du batch", 8, 128, 32)
            epochs = st.slider("Nombre d'époques", 10, 500, 100)
        
        # Validation croisée
        st.markdown("#### Validation")
        use_cv = st.checkbox("Utiliser la validation croisée", value=True)
        
        if use_cv:
            n_folds = st.slider("Nombre de plis (folds)", 2, 10, 5)
        
        # Bouton pour entraîner le modèle
        if st.button("Entraîner le modèle"):
            with st.spinner("Entraînement du modèle en cours..."):
                try:
                    # Préparation des transformateurs pour le prétraitement
                    preprocessor_parts = []
                    
                    # Preprocessing pour les variables numériques
                    if numeric_features:
                        if scaler_option == "StandardScaler":
                            numeric_transformer = Pipeline(steps=[
                                ('scaler', StandardScaler())
                            ])
                        elif scaler_option == "MinMaxScaler":
                            numeric_transformer = Pipeline(steps=[
                                ('scaler', MinMaxScaler())
                            ])
                        elif scaler_option == "RobustScaler":
                            numeric_transformer = Pipeline(steps=[
                                ('scaler', RobustScaler())
                            ])
                        else:  # Aucune
                            numeric_transformer = Pipeline(steps=[
                                ('passthrough', 'passthrough')
                            ])
                        
                        preprocessor_parts.append(('num', numeric_transformer, numeric_features))
                    
                    # Preprocessing pour les variables catégorielles
                    if categorical_features:
                        if encoding_method == "One-Hot Encoding":
                            categorical_transformer = Pipeline(steps=[
                                ('encoder', OneHotEncoder(drop='first', sparse=False))
                            ])
                        else:  # Ordinal Encoding
                            categorical_transformer = Pipeline(steps=[
                                ('encoder', OrdinalEncoder())
                            ])
                        
                        preprocessor_parts.append(('cat', categorical_transformer, categorical_features))
                    
                    # Préprocesseur complet
                    preprocessor = ColumnTransformer(
                        transformers=preprocessor_parts,
                        sparse_threshold=0
                    )
                    
                    # Sélection et configuration du modèle
                    if model_type == "Régression Linéaire":
                        model = LinearRegression()
                    elif model_type == "Ridge":
                        model = Ridge(alpha=alpha)
                    elif model_type == "Lasso":
                        model = Lasso(alpha=alpha)
                    elif model_type == "ElasticNet":
                        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
                    elif model_type == "SVR":
                        model = SVR(kernel=kernel, C=C)
                    elif model_type == "Arbre de Décision":
                        model = DecisionTreeRegressor(random_state=42)
                    elif model_type == "Random Forest":
                        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                    elif model_type == "Gradient Boosting":
                        model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                    elif model_type == "XGBoost" and XGBOOST_AVAILABLE:
                        model = xgb.XGBRegressor(
                            n_estimators=n_estimators_xgb,
                            max_depth=max_depth_xgb,
                            learning_rate=learning_rate_xgb,
                            subsample=subsample_xgb,
                            random_state=42
                        )
                    elif model_type == "Réseau de Neurones" and TENSORFLOW_AVAILABLE:
                        # Pour les réseaux de neurones, nous allons créer un modèle Keras
                        # mais ne pas l'inclure dans le pipeline scikit-learn
                        
                        # Préprocesser les données
                        X_train_processed = preprocessor.fit_transform(X_train)
                        X_test_processed = preprocessor.transform(X_test)
                        
                        # Définition de l'architecture du réseau
                        nn_model = Sequential()
                        
                        # Couche d'entrée
                        nn_model.add(Dense(layer_sizes[0], activation='relu', input_shape=(X_train_processed.shape[1],)))
                        nn_model.add(Dropout(dropout_rate))
                        
                        # Couches cachées
                        for i in range(1, n_layers):
                            nn_model.add(Dense(layer_sizes[i], activation='relu'))
                            nn_model.add(Dropout(dropout_rate))
                        
                        # Couche de sortie
                        nn_model.add(Dense(1))  # Une seule sortie pour la régression
                        
                        # Compilation du modèle
                        optimizer = Adam(learning_rate=learning_rate)
                        nn_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
                        
                        # Early stopping pour éviter le surapprentissage
                        early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
                        
                        # Entraînement du modèle
                        history = nn_model.fit(
                            X_train_processed, y_train,
                            validation_split=0.2,
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[early_stop],
                            verbose=0
                        )
                        
                        # Prédictions et évaluation
                        y_pred = nn_model.predict(X_test_processed).flatten()
                        
                        # Calculer les métriques
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        # Stocker le modèle, le préprocesseur et les métriques dans la session state
                        st.session_state.neural_network = {
                            'model': nn_model,
                            'preprocessor': preprocessor,
                            'history': history.history,
                            'metrics': {
                                'mse': mse,
                                'rmse': rmse,
                                'mae': mae,
                                'r2': r2
                            },
                            'predictions': y_pred
                        }
                        
                        # Afficher les résultats
                        st.success("Modèle de réseau de neurones entraîné avec succès!")
                        
                        # Graphique de la perte (loss) pendant l'entraînement
                        fig = px.line(
                            x=range(1, len(history.history['loss'])+1),
                            y=[history.history['loss'], history.history['val_loss']],
                            labels={'x': 'Époques', 'y': 'Perte (MSE)'},
                            title="Évolution de la perte pendant l'entraînement",
                            color_discrete_sequence=['blue', 'red']
                        )
                        
                        # Mise à jour des noms de série
                        fig.data[0].name = 'Entraînement'
                        fig.data[1].name = 'Validation'
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Afficher les métriques
                        st.markdown("#### Métriques de performance")
                        
                        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                        
                        with metrics_col1:
                            st.metric("MSE", f"{mse:.4f}")
                        with metrics_col2:
                            st.metric("RMSE", f"{rmse:.4f}")
                        with metrics_col3:
                            st.metric("MAE", f"{mae:.4f}")
                        with metrics_col4:
                            st.metric("R²", f"{r2:.4f}")
                        
                        # Graphique de prédictions vs valeurs réelles
                        fig = px.scatter(
                            x=y_test,
                            y=y_pred,
                            labels={'x': 'Valeurs réelles', 'y': 'Prédictions'},
                            title="Prédictions vs Valeurs réelles"
                        )
                        
                        # Ajouter une ligne diagonale parfaite
                        min_val = min(min(y_test), min(y_pred))
                        max_val = max(max(y_test), max(y_pred))
                        
                        fig.add_trace(
                            go.Scatter(
                                x=[min_val, max_val],
                                y=[min_val, max_val],
                                mode='lines',
                                line=dict(color='red', dash='dash'),
                                name='y = x (prédiction parfaite)'
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculer les résidus
                        residuals = y_test - y_pred
                        
                        # Graphique des résidus
                        fig = px.scatter(
                            x=y_pred,
                            y=residuals,
                            labels={'x': 'Prédictions', 'y': 'Résidus'},
                            title="Résidus vs Prédictions"
                        )
                        
                        # Ajouter une ligne horizontale à y=0
                        fig.add_hline(y=0, line_dash="dash", line_color="red")
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        # Pour les autres modèles, utiliser un pipeline scikit-learn
                        pipeline = Pipeline(steps=[
                            ('preprocessor', preprocessor),
                            ('model', model)
                        ])
                        
                        # Entraînement du modèle
                        pipeline.fit(X_train, y_train)
                        
                        # Évaluation du modèle sur les données de test
                        y_pred = pipeline.predict(X_test)
                        
                        # Validation croisée
                        if use_cv:
                            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=n_folds, scoring='neg_mean_squared_error')
                            cv_rmse = np.sqrt(-cv_scores)
                        
                        # Calculer les métriques
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        # Stocker le pipeline dans la session state
                        st.session_state.model_pipeline = pipeline
                        st.session_state.model_is_fitted = True
                        
                        # Afficher les résultats
                        st.success(f"Modèle {model_type} entraîné avec succès!")
                        
                        # Afficher les métriques
                        st.markdown("#### Métriques de performance")
                        
                        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                        
                        with metrics_col1:
                            st.metric("MSE", f"{mse:.4f}")
                        with metrics_col2:
                            st.metric("RMSE", f"{rmse:.4f}")
                        with metrics_col3:
                            st.metric("MAE", f"{mae:.4f}")
                        with metrics_col4:
                            st.metric("R²", f"{r2:.4f}")
                        
                        # Afficher les résultats de validation croisée si utilisée
                        if use_cv:
                            st.markdown("#### Résultats de validation croisée")
                            st.write(f"RMSE moyen (CV-{n_folds}): {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
                            
                            # Graphique des scores de CV
                            fig = px.bar(
                                x=[f"Fold {i+1}" for i in range(n_folds)],
                                y=cv_rmse,
                                labels={'x': 'Fold', 'y': 'RMSE'},
                                title=f"RMSE par fold (validation croisée {n_folds}-fold)"
                            )
                            
                            # Ajouter une ligne pour la moyenne
                            fig.add_hline(y=cv_rmse.mean(), line_dash="dash", line_color="red", annotation_text="Moyenne")
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Graphique de prédictions vs valeurs réelles
                        fig = px.scatter(
                            x=y_test,
                            y=y_pred,
                            labels={'x': 'Valeurs réelles', 'y': 'Prédictions'},
                            title="Prédictions vs Valeurs réelles"
                        )
                        
                        # Ajouter une ligne diagonale parfaite
                        min_val = min(min(y_test), min(y_pred))
                        max_val = max(max(y_test), max(y_pred))
                        
                        fig.add_trace(
                            go.Scatter(
                                x=[min_val, max_val],
                                y=[min_val, max_val],
                                mode='lines',
                                line=dict(color='red', dash='dash'),
                                name='y = x (prédiction parfaite)'
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculer les résidus
                        residuals = y_test - y_pred
                        
                        # Graphique des résidus
                        fig = px.scatter(
                            x=y_pred,
                            y=residuals,
                            labels={'x': 'Prédictions', 'y': 'Résidus'},
                            title="Résidus vs Prédictions"
                        )
                        
                        # Ajouter une ligne horizontale à y=0
                        fig.add_hline(y=0, line_dash="dash", line_color="red")
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Importance des variables (si disponible)
                        try:
                            if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
                                importances = pipeline.named_steps['model'].feature_importances_
                                
                                # Pour récupérer les noms des features après transformation
                                if categorical_features and encoding_method == "One-Hot Encoding":
                                    # C'est plus complexe avec le one-hot encoding
                                    # Nous devons reconstruire les noms des features
                                    feature_names = []
                                    feature_names.extend(numeric_features)
                                    
                                    # Récupérer les noms des features après one-hot encoding
                                    cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
                                    cat_features_encoded = cat_encoder.get_feature_names_out(categorical_features)
                                    feature_names.extend(cat_features_encoded)
                                else:
                                    feature_names = features
                                
                                # Créer un DataFrame pour l'importance des features
                                importance_df = pd.DataFrame({
                                    'Variable': feature_names,
                                    'Importance': importances
                                }).sort_values('Importance', ascending=False)
                                
                                fig = px.bar(
                                    importance_df, 
                                    x='Variable', 
                                    y='Importance',
                                    title="Importance des Variables"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Impossible d'afficher l'importance des variables: {e}")
                        
                        # SHAP values pour l'interprétabilité
                        if st.checkbox("Afficher l'analyse SHAP (interprétabilité avancée)"):
                            try:
                                with st.spinner("Calcul des valeurs SHAP en cours..."):
                                    # Création de l'explainer SHAP
                                    X_test_processed = pipeline.named_steps['preprocessor'].transform(X_test)
                                    explainer = shap.TreeExplainer(pipeline.named_steps['model'])
                                    shap_values = explainer.shap_values(X_test_processed)
                                    
                                    # Conversion du plot SHAP en figure matplotlib
                                    st.subheader("Graphique de résumé SHAP")
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    
                                    # Tenter de reconstruire les noms des features pour SHAP
                                    try:
                                        if categorical_features and encoding_method == "One-Hot Encoding":
                                            # Reconstruire les noms des features pour One-Hot Encoding
                                            feature_names = []
                                            feature_names.extend(numeric_features)
                                            cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
                                            cat_features_encoded = cat_encoder.get_feature_names_out(categorical_features)
                                            feature_names.extend(cat_features_encoded)
                                            
                                            shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names, show=False)
                                        else:
                                            shap.summary_plot(shap_values, X_test_processed, feature_names=features, show=False)
                                    except:
                                        # En cas d'échec, procéder sans noms de features
                                        shap.summary_plot(shap_values, X_test_processed, show=False)
                                    
                                    st.pyplot(fig)
                                    plt.clf()
                            except Exception as e:
                                st.error(f"Erreur lors du calcul des valeurs SHAP: {e}")
                    
                    # Pour les réseaux de neurones, option de sauvegarde du modèle
                    if model_type == "Réseau de Neurones" and TENSORFLOW_AVAILABLE:
                        if st.button("Sauvegarder le modèle"):
                            # Sauvegarder le modèle et le préprocesseur
                            model_path = "modele_reseau_neuronal.h5"
                            preprocessor_path = "preprocesseur.joblib"
                            
                            nn_model.save(model_path)
                            joblib.dump(preprocessor, preprocessor_path)
                            
                            # Créer un lien de téléchargement pour le modèle
                            with open(model_path, "rb") as f:
                                model_bytes = f.read()
                                b64_model = base64.b64encode(model_bytes).decode()
                                href_model = f'<a href="data:file/h5;base64,{b64_model}" download="{model_path}">Télécharger le modèle de réseau neuronal</a>'
                                st.markdown(href_model, unsafe_allow_html=True)
                            
                            # Créer un lien de téléchargement pour le préprocesseur
                            with open(preprocessor_path, "rb") as f:
                                preprocessor_bytes = f.read()
                                b64_preprocessor = base64.b64encode(preprocessor_bytes).decode()
                                href_preprocessor = f'<a href="data:file/joblib;base64,{b64_preprocessor}" download="{preprocessor_path}">Télécharger le préprocesseur</a>'
                                st.markdown(href_preprocessor, unsafe_allow_html=True)
                            
                            st.success("Modèle et préprocesseur sauvegardés avec succès!")
                except Exception as e:
                    st.error(f"Erreur lors de l'entraînement du modèle: {e}")
                    st.info("Conseil: Vérifiez vos données et assurez-vous qu'il n'y a pas de valeurs manquantes ou de problèmes de type de données.")
        st.markdown("</div>", unsafe_allow_html=True)

# Page de prédiction
elif page == "Prédiction":
    st.markdown("<h2 class='sub-header'>Prédiction de Récupération Métallurgique</h2>", unsafe_allow_html=True)
    
    # Vérifier si un modèle a été entraîné
    model_trained = (st.session_state.model_pipeline is not None and st.session_state.model_is_fitted) or \
                   (st.session_state.neural_network is not None and 'model' in st.session_state.neural_network)
    
    if not model_trained:
        st.warning("Aucun modèle n'a été entraîné. Veuillez aller à la page 'Modélisation' pour entraîner un modèle.")
        if st.button("Aller à la page Modélisation"):
            st.session_state.page = "Modélisation"
            st.rerun()
    elif st.session_state.features is None:
        st.warning("Variables prédictives non définies. Veuillez aller à la page 'Exploration des données'.")
    else:
        # Vérifier quel type de modèle est disponible
        if st.session_state.model_pipeline is not None and st.session_state.model_is_fitted:
            pipeline = st.session_state.model_pipeline
            model_type = "standard"
        else:
            nn_data = st.session_state.neural_network
            model_type = "neural_network"
        
        features = st.session_state.features
        target = st.session_state.target
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Prédiction pour un nouvel échantillon")
        
        # Méthode pour rentrer les valeurs
        input_method = st.radio("Méthode d'entrée des données", ["Saisie manuelle", "Importer un fichier"])
        
        if input_method == "Saisie manuelle":
            # Création d'un formulaire pour entrer les valeurs des features
            input_data = {}
            
            for feature in features:
                # Obtenir la plage des valeurs dans les données d'entraînement pour définir les min/max
                if st.session_state.X_train is not None:
                    if pd.api.types.is_numeric_dtype(st.session_state.X_train[feature]):
                        min_val = float(st.session_state.X_train[feature].min())
                        max_val = float(st.session_state.X_train[feature].max())
                        mean_val = float(st.session_state.X_train[feature].mean())
                        
                        # Ajuster légèrement les min/max pour éviter les problèmes de types
                        min_val = min_val * 0.9 if min_val > 0 else min_val * 1.1
                        max_val = max_val * 1.1 if max_val > 0 else max_val * 0.9
                        
                        input_data[feature] = st.slider(
                            f"{feature}", 
                            min_val, 
                            max_val, 
                            mean_val
                        )
                    else:
                        # Pour les variables catégorielles
                        categories = st.session_state.X_train[feature].dropna().unique()
                        input_data[feature] = st.selectbox(f"{feature}", options=categories)
                else:
                    # Fallback si X_train n'est pas disponible
                    input_data[feature] = st.number_input(f"{feature}")
            
            # Bouton pour faire la prédiction
            if st.button("Prédire"):
                try:
                    # Préparation des données d'entrée
                    input_df = pd.DataFrame([input_data])
                    
                    # Faire la prédiction
                    if model_type == "standard":
                        prediction = pipeline.predict(input_df)[0]
                    else:
                        # Pour le réseau de neurones
                        preprocessor = nn_data['preprocessor']
                        nn_model = nn_data['model']
                        
                        # Prétraitement
                        input_processed = preprocessor.transform(input_df)
                        
                        # Prédiction
                        prediction = nn_model.predict(input_processed)[0][0]
                    
                    # Afficher la prédiction
                    st.success(f"Prédiction de la récupération métallurgique: **{prediction:.2f}%**")
                    
                    # Jauge pour visualiser la prédiction
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prediction,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': f"Prédiction de {target}"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "#2ecc71"},
                            'steps': [
                                {'range': [0, 50], 'color': "#e74c3c"},
                                {'range': [50, 80], 'color': "#f39c12"},
                                {'range': [80, 100], 'color': "#2ecc71"}
                            ]
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur lors de la prédiction: {e}")
                    st.info("Il semble que le modèle n'a pas été correctement entraîné ou qu'il y a un problème avec les données d'entrée. Veuillez retourner à la page Modélisation.")
                
        else:  # Importation d'un fichier
            st.markdown("""
            Téléchargez un fichier CSV ou Excel contenant les données pour lesquelles vous souhaitez faire des prédictions.
            Le fichier doit contenir les colonnes suivantes:
            """)
            
            # Afficher les features requises
            for feature in features:
                st.markdown(f"- {feature}")
            
            # Upload du fichier
            uploaded_file = st.file_uploader("Télécharger votre fichier de données", type=["csv", "xlsx"])
            
            if uploaded_file is not None:
                try:
                    # Chargement des données
                    if uploaded_file.name.endswith('.csv'):
                        predict_data = pd.read_csv(uploaded_file)
                    else:
                        predict_data = pd.read_excel(uploaded_file)
                    
                    # Vérifier que toutes les colonnes requises sont présentes
                    missing_cols = [col for col in features if col not in predict_data.columns]
                    
                    if missing_cols:
                        st.error(f"Colonnes manquantes dans le fichier: {', '.join(missing_cols)}")
                    else:
                        # Extraire seulement les colonnes nécessaires
                        predict_data = predict_data[features]
                        
                        # Faire les prédictions
                        try:
                            if model_type == "standard":
                                predictions = pipeline.predict(predict_data)
                            else:
                                # Pour le réseau de neurones
                                preprocessor = nn_data['preprocessor']
                                nn_model = nn_data['model']
                                
                                # Prétraitement
                                input_processed = preprocessor.transform(predict_data)
                                
                                # Prédiction
                                predictions = nn_model.predict(input_processed).flatten()
                            
                            # Ajouter les prédictions au dataframe
                            results = predict_data.copy()
                            results[f"{target}_predit"] = predictions
                            
                            # Afficher les résultats
                            st.subheader("Résultats des prédictions")
                            st.dataframe(results)
                            
                            # Histogramme des prédictions
                            fig = px.histogram(
                                results, 
                                x=f"{target}_predit", 
                                nbins=20,
                                title=f"Distribution des prédictions de {target}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Option pour télécharger les résultats
                            csv = results.to_csv(index=False)
                            b64 = base64.b64encode(csv.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="resultats_predictions.csv">Télécharger les résultats des prédictions</a>'
                            st.markdown(href, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Erreur lors de la prédiction: {e}")
                            st.info("Il semble que le modèle n'a pas été correctement entraîné ou qu'il y a un problème avec les données d'entrée. Veuillez retourner à la page Modélisation.")
                except Exception as e:
                    st.error(f"Erreur lors du traitement du fichier: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Section pour la simulation et l'optimisation
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Optimisation des Paramètres")
        
        st.markdown("""
        Vous pouvez utiliser cette section pour explorer l'effet de différents paramètres sur la récupération métallurgique 
        et identifier les paramètres optimaux pour maximiser la récupération.
        """)
        
        # Sélection des paramètres à optimiser
        numeric_features = [f for f in features if st.session_state.X_train is not None and pd.api.types.is_numeric_dtype(st.session_state.X_train[f])]
        
        params_to_optimize = st.multiselect(
            "Sélectionnez les paramètres à optimiser",
            numeric_features,
            default=numeric_features[:min(2, len(numeric_features))]
        )
        
        if len(params_to_optimize) >= 1:
            # Valeurs par défaut pour les paramètres qui ne sont pas optimisés
            default_values = {}
            for feature in features:
                if feature not in params_to_optimize:
                    if st.session_state.X_train is not None:
                        if pd.api.types.is_numeric_dtype(st.session_state.X_train[feature]):
                            default_values[feature] = float(st.session_state.X_train[feature].mean())
                        else:
                            # Pour les variables catégorielles, prendre la valeur la plus fréquente
                            default_values[feature] = st.session_state.X_train[feature].mode()[0]
                    else:
                        default_values[feature] = 0.0
            
            # Créer des sliders pour les plages des paramètres à optimiser
            param_ranges = {}
            
            for param in params_to_optimize:
                if st.session_state.X_train is not None:
                    min_val = float(st.session_state.X_train[param].min())
                    max_val = float(st.session_state.X_train[param].max())
                    
                    # Ajuster légèrement les min/max pour éviter les problèmes de types
                    min_val = min_val * 0.9 if min_val > 0 else min_val * 1.1
                    max_val = max_val * 1.1 if max_val > 0 else max_val * 0.9
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        param_min = st.number_input(f"Min pour {param}", value=min_val)
                    with col2:
                        param_max = st.number_input(f"Max pour {param}", value=max_val)
                    
                    param_ranges[param] = (param_min, param_max)
                else:
                    param_ranges[param] = (0, 10)  # Valeurs par défaut
            
            # Bouton pour lancer la simulation
            if st.button("Lancer la simulation"):
                with st.spinner("Simulation en cours..."):
                    try:
                        if len(params_to_optimize) == 1:
                            # Simulation 1D
                            param = params_to_optimize[0]
                            param_range = np.linspace(param_ranges[param][0], param_ranges[param][1], 100)
                            
                            sim_results = []
                            for val in param_range:
                                input_data = default_values.copy()
                                input_data[param] = val
                                input_df = pd.DataFrame([input_data])
                                
                                if model_type == "standard":
                                    prediction = pipeline.predict(input_df)[0]
                                else:
                                    # Pour le réseau de neurones
                                    preprocessor = nn_data['preprocessor']
                                    nn_model = nn_data['model']
                                    
                                    # Prétraitement
                                    input_processed = preprocessor.transform(input_df)
                                    
                                    # Prédiction
                                    prediction = nn_model.predict(input_processed)[0][0]
                                
                                sim_results.append(prediction)
                            
                            # Visualisation
                            fig = px.line(
                                x=param_range, 
                                y=sim_results,
                                labels={"x": param, "y": f"Prédiction de {target}"},
                                title=f"Effet de {param} sur {target}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Trouver la valeur optimale
                            optimal_idx = np.argmax(sim_results)
                            optimal_val = param_range[optimal_idx]
                            optimal_result = sim_results[optimal_idx]
                            
                            st.success(f"Valeur optimale de {param}: {optimal_val:.2f} → {optimal_result:.2f}% de récupération")
                            
                        elif len(params_to_optimize) == 2:
                            # Simulation 2D
                            param1, param2 = params_to_optimize
                            param1_range = np.linspace(param_ranges[param1][0], param_ranges[param1][1], 30)
                            param2_range = np.linspace(param_ranges[param2][0], param_ranges[param2][1], 30)
                            
                            # Créer une grille de valeurs
                            param1_grid, param2_grid = np.meshgrid(param1_range, param2_range)
                            sim_results = np.zeros_like(param1_grid)
                            
                            # Calculer les prédictions pour chaque combinaison
                            for i in range(len(param1_range)):
                                for j in range(len(param2_range)):
                                    input_data = default_values.copy()
                                    input_data[param1] = param1_grid[j, i]
                                    input_data[param2] = param2_grid[j, i]
                                    input_df = pd.DataFrame([input_data])
                                    
                                    if model_type == "standard":
                                        sim_results[j, i] = pipeline.predict(input_df)[0]
                                    else:
                                        # Pour le réseau de neurones
                                        preprocessor = nn_data['preprocessor']
                                        nn_model = nn_data['model']
                                        
                                        # Prétraitement
                                        input_processed = preprocessor.transform(input_df)
                                        
                                        # Prédiction
                                        sim_results[j, i] = nn_model.predict(input_processed)[0][0]
                            
                            # Visualisation de la surface de réponse
                            fig = go.Figure(data=[go.Surface(z=sim_results, x=param1_range, y=param2_range)])
                            fig.update_layout(
                                title=f"Surface de réponse pour {target}",
                                scene=dict(
                                    xaxis_title=param1,
                                    yaxis_title=param2,
                                    zaxis_title=target
                                ),
                                width=700,
                                height=700
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Graphique de contour pour une visualisation plus claire
                            fig = px.contour(
                                x=param1_range, 
                                y=param2_range, 
                                z=sim_results,
                                labels=dict(x=param1, y=param2, z=target),
                                title=f"Contours de {target}",
                                height=500
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Trouver les valeurs optimales
                            optimal_idx = np.unravel_index(np.argmax(sim_results), sim_results.shape)
                            optimal_val1 = param1_range[optimal_idx[1]]
                            optimal_val2 = param2_range[optimal_idx[0]]
                            optimal_result = sim_results[optimal_idx]
                            
                            st.success(f"Valeurs optimales: {param1}={optimal_val1:.2f}, {param2}={optimal_val2:.2f} → {optimal_result:.2f}% de récupération")
                    except Exception as e:
                        st.error(f"Erreur lors de la simulation: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

# Gestion des paramètres d'URL pour la déconnexion
if st.query_params.get("logout", [""])[0] == "true":
    st.session_state["authentication_status"] = False
    st.session_state["username"] = ""
    st.query_params.clear()
    st.rerun()

# Footer
st.markdown("<div class='footer'>Développé par Didier Ouedraogo, P.Geo, Géologue et Data Scientist - Application de Prédiction de Récupération Métallurgique © 2025</div>", unsafe_allow_html=True)