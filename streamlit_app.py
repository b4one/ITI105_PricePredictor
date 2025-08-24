"""
File: streamlit_app.py
Purpose: Enhanced Streamlit web application for HDB price prediction with improved UI
Author: Team AlgoRiddler - ITI105 Project
Date: Aug 2025
Dependencies: streamlit, pandas, numpy, joblib, plotly
Input: User inputs via web interface
Output: Price predictions with confidence intervals and market insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# ---- Cross-environment path setup (script, Jupyter, Colab) ----
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

# If repo layout has data/ next to the script, use the script folder as root
PROJECT_ROOT = SCRIPT_DIR

# (Optional) auto-detect parent if data/ is one level up
expected = PROJECT_ROOT / "data" / "raw" / "base_hdb_resale_prices_2015Jan-2025Jun.csv"
if not expected.exists() and (SCRIPT_DIR.parent / "data" / "raw" / "base_hdb_resale_prices_2015Jan-2025Jun.csv").exists():
    PROJECT_ROOT = SCRIPT_DIR.parent

DATA_DIR           = PROJECT_ROOT / "data"
RAW_DATA_DIR       = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR         = PROJECT_ROOT / "models"
OUTPUT_DIR         = PROJECT_ROOT / "output"
RANDOM_STATE       = 42

# Ensure folders exist
for p in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="HDB Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        #text-align: left;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    # .info-box {
    #     background: #f8f9fa;
    #     padding: 1rem;
    #     border-radius: 8px;
    #     border-left: 4px solid #17a2b8;
    #     margin: 1rem 0;
    # }
    .info-box{
        background:#f8f9fa;
        padding:1rem 1.25rem;
        border-radius:10px;
        border-left:4px solid #17a2b8;
        margin:0.5rem auto 1.25rem;      /* auto centers horizontally */
        box-shadow:0 2px 4px rgba(0,0,0,.06);
    }
    .info-center{
        max-width:1500px;                  /* keep it narrow, centered */
        text-align:center;
        font-size:1.15rem;                /* more pronounced */
        font-weight:600;
    }
    .stSelectbox > div > div { background-color: #f8f9fa; }
    .stSlider > div > div { background-color: #f8f9fa; }
    .block-container { padding-top: 1rem; }
    .main-content {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .right-panel {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .insight-label { font-size: 0.85rem; color: #6c757d; margin-bottom: 0.25rem; }
    #.insight-value { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.75rem; }
    .insight-value { font-size: 1.2rem; font-weight: 500; margin-bottom: 0.75rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_reference_data():
    """Load reference data for market insights chart."""
    try:
        data_path = RAW_DATA_DIR / 'base_hdb_resale_prices_2015Jan-2025Jun.csv'
        if data_path.exists():
            df = pd.read_csv(data_path)
            return {'sample_data': df.sample(min(1000, len(df)), random_state=42)}
        else:
            return None
    except Exception as e:
        st.warning(f"Could not load reference data: {e}")
        return None

@st.cache_resource
def load_models():
    """Load all trained models and preprocessing objects."""
    models = {}
    try:
        # Load individual models
        models['linear'] = joblib.load(MODELS_DIR / 'best_linear_model.joblib')
        models['tree'] = joblib.load(MODELS_DIR / 'best_tree_model.joblib')
        models['boosting'] = joblib.load(MODELS_DIR / 'best_boosting_model.joblib')
        # Preprocessing objects
        models['linear_prep'] = joblib.load(MODELS_DIR / "linear_preprocessing.joblib")
        models['tree_prep'] = joblib.load(MODELS_DIR / "tree_preprocessing.joblib")
        models['boosting_prep'] = joblib.load(MODELS_DIR / "boosting_preprocessing.joblib")
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def get_flat_type_ordinal(flat_type):
    """Convert flat type to ordinal encoding."""
    mapping = {'1 ROOM': 1, '2 ROOM': 2, '3 ROOM': 3, '4 ROOM': 4, '5 ROOM': 5, 'EXECUTIVE': 6, 'MULTI-GENERATION': 7}
    return mapping.get(flat_type, 4)

def get_room_count(flat_type):
    """Get room count from flat type."""
    mapping = {'1 ROOM': 1, '2 ROOM': 2, '3 ROOM': 3, '4 ROOM': 4, '5 ROOM': 5, 'EXECUTIVE': 5, 'MULTI-GENERATION': 5}
    return mapping.get(flat_type, 4)

def get_market_segment(flat_type):
    """Determine market segment based on flat type."""
    if flat_type in ['1 ROOM', '2 ROOM']:
        return 'Budget'
    elif flat_type in ['3 ROOM', '4 ROOM']:
        return 'Mass Market'
    elif flat_type == '5 ROOM':
        return 'Premium'
    else:
        return 'Luxury'

def get_lease_health(remaining_years):
    """Determine lease health category."""
    if remaining_years >= 80:
        return 'Excellent'
    elif remaining_years >= 60:
        return 'Good'
    elif remaining_years >= 40:
        return 'Fair'
    elif remaining_years >= 20:
        return 'Poor'
    else:
        return 'Critical'

def get_storey_category(storey_level):
    """Categorize storey level."""
    if storey_level <= 3:
        return 'Low'
    elif storey_level <= 10:
        return 'Mid'
    else:
        return 'High'

def preprocess_input_linear(user_input, preprocessing_objects):
    """Preprocess user input for linear models."""
    features = {}
    # Numerical
    features['floor_area_sqm'] = user_input['floor_area_sqm']
    features['storey_level'] = user_input['storey_level']
    features['flat_age'] = user_input['flat_age']
    features['remaining_lease_years'] = 99 - user_input['flat_age']
    features['flat_type_ordinal'] = get_flat_type_ordinal(user_input['flat_type'])
    features['transaction_year'] = 2025
    features['lease_used_ratio'] = user_input['flat_age'] / 99.0
    features['lease_remaining_ratio'] = features['remaining_lease_years'] / 99.0
    features['area_per_room'] = user_input['floor_area_sqm'] / get_room_count(user_input['flat_type'])
    features['bank_loan_eligible'] = 1 if features['remaining_lease_years'] >= 60 else 0
    features['hdb_loan_eligible'] = 1 if features['remaining_lease_years'] >= 20 else 0
    features['cash_buyers_only'] = 1 if features['remaining_lease_years'] < 20 else 0
    # Categorical (OHE)
    categorical_features = ['town', 'flat_type', 'flat_model', 'market_segment', 'lease_health', 'storey_category']
    feature_names = preprocessing_objects.get('feature_names', [])
    one_hot_categories = preprocessing_objects.get('one_hot_categories')
    for cat_feature in categorical_features:
        if cat_feature == 'market_segment':
            value = get_market_segment(user_input['flat_type'])
        elif cat_feature == 'lease_health':
            value = get_lease_health(features['remaining_lease_years'])
        elif cat_feature == 'storey_category':
            value = get_storey_category(user_input['storey_level'])
        else:
            value = user_input[cat_feature]
        if one_hot_categories and (cat_feature in one_hot_categories):
            categories = one_hot_categories[cat_feature]
            for category in categories:
                features[f'{cat_feature}_{category}'] = 1 if str(value) == str(category) else 0
        else:
            prefix = f'{cat_feature}_'
            candidates = [c for c in feature_names if c.startswith(prefix)]
            if candidates:
                target = f'{prefix}{value}'
                for c in candidates:
                    features[c] = 1 if c == target else 0
    # DataFrame + scaling
    feature_df = pd.DataFrame([features])
    for col in feature_names:
        if col not in feature_df.columns:
            feature_df[col] = 0
    feature_df = feature_df.reindex(columns=feature_names, fill_value=0)
    scaler = preprocessing_objects.get('scaler')
    if scaler is not None:
        try:
            if hasattr(scaler, "feature_names_in_"):
                scaler_cols = list(scaler.feature_names_in_)
                missing = [c for c in scaler_cols if c not in feature_df.columns]
                for m in missing:
                    feature_df[m] = 0.0
                Xs = feature_df[scaler_cols].astype(float)
                feature_df[scaler_cols] = scaler.transform(Xs)
            else:
                Xs = feature_df.select_dtypes(include=[np.number])
                feature_df[Xs.columns] = scaler.transform(Xs)
        except Exception as e:
            st.warning(f"[linear] scaler.transform failed; continuing unscaled. Error: {e}")
    return feature_df.values

def preprocess_input_tree(user_input, preprocessing_objects):
    """Preprocess user input for tree models."""
    features = {}
    # Numerical
    features['floor_area_sqm'] = user_input['floor_area_sqm']
    features['storey_level'] = user_input['storey_level']
    features['flat_age'] = user_input['flat_age']
    features['remaining_lease_years'] = 99 - user_input['flat_age']
    features['flat_type_ordinal'] = get_flat_type_ordinal(user_input['flat_type'])
    features['transaction_year'] = 2025
    features['lease_used_ratio'] = user_input['flat_age'] / 99.0
    features['lease_remaining_ratio'] = features['remaining_lease_years'] / 99.0
    features['area_per_room'] = user_input['floor_area_sqm'] / get_room_count(user_input['flat_type'])
    features['bank_loan_eligible'] = 1 if features['remaining_lease_years'] >= 60 else 0
    features['hdb_loan_eligible'] = 1 if features['remaining_lease_years'] >= 20 else 0
    features['cash_buyers_only'] = 1 if features['remaining_lease_years'] < 20 else 0
    # Categorical (Label Encoders)
    categorical_features = ['town', 'flat_type', 'flat_model', 'market_segment', 'lease_health', 'storey_category']
    label_encoders = preprocessing_objects.get('label_encoders', {})
    feature_names = preprocessing_objects.get('feature_names', [])
    for cat_feature in categorical_features:
        if cat_feature == 'market_segment':
            value = get_market_segment(user_input['flat_type'])
        elif cat_feature == 'lease_health':
            value = get_lease_health(features['remaining_lease_years'])
        elif cat_feature == 'storey_category':
            value = get_storey_category(user_input['storey_level'])
        else:
            value = user_input[cat_feature]
        if cat_feature in label_encoders:
            le = label_encoders[cat_feature]
            try:
                features[f'{cat_feature}_encoded'] = le.transform([str(value)])[0]
            except ValueError:
                features[f'{cat_feature}_encoded'] = 0
    feature_df = pd.DataFrame([features]).reindex(columns=feature_names, fill_value=0)
    return feature_df.values

def preprocess_input_boosting(user_input, preprocessing_objects):
    """Preprocess user input for boosting models."""
    features = {}
    # Numerical
    features['floor_area_sqm'] = user_input['floor_area_sqm']
    features['storey_level'] = user_input['storey_level']
    features['flat_age'] = user_input['flat_age']
    features['remaining_lease_years'] = 99 - user_input['flat_age']
    features['flat_type_ordinal'] = get_flat_type_ordinal(user_input['flat_type'])
    features['transaction_year'] = 2025
    features['lease_used_ratio'] = user_input['flat_age'] / 99.0
    features['lease_remaining_ratio'] = features['remaining_lease_years'] / 99.0
    features['area_per_room'] = user_input['floor_area_sqm'] / get_room_count(user_input['flat_type'])
    features['bank_loan_eligible'] = 1 if features['remaining_lease_years'] >= 60 else 0
    features['hdb_loan_eligible'] = 1 if features['remaining_lease_years'] >= 20 else 0
    features['cash_buyers_only'] = 1 if features['remaining_lease_years'] < 20 else 0
    # Target-encode high-cardinality
    target_encode_features = ['town', 'flat_model']
    target_encoders = preprocessing_objects.get('target_encoders', {})
    for feature in target_encode_features:
        if feature in target_encoders:
            te = target_encoders[feature]
            temp_df = pd.DataFrame({feature: [user_input[feature]]})
            try:
                features[f'{feature}_target_encoded'] = te.transform(temp_df).ravel()[0]
            except Exception:
                features[f'{feature}_target_encoded'] = 500000
    # Label-encode low-cardinality
    label_encode_features = ['flat_type', 'market_segment', 'lease_health']
    label_encoders = preprocessing_objects.get('label_encoders', {})
    for feature in label_encode_features:
        if feature == 'market_segment':
            value = get_market_segment(user_input['flat_type'])
        elif feature == 'lease_health':
            value = get_lease_health(99 - user_input['flat_age'])
        else:
            value = user_input[feature]
        if feature in label_encoders:
            le = label_encoders[feature]
            try:
                features[f'{feature}_encoded'] = le.transform([str(value)])[0]
            except ValueError:
                features[f'{feature}_encoded'] = 0
    feature_names = preprocessing_objects.get('feature_names', [])
    feature_df = pd.DataFrame([features]).reindex(columns=feature_names, fill_value=0)
    return feature_df.values

def predict_price(user_input, models):
    """Generate predictions from all models."""
    predictions = {}
    try:
        X_linear = preprocess_input_linear(user_input, models['linear_prep'])
        predictions['linear'] = max(0, models['linear'].predict(X_linear)[0])
        X_tree = preprocess_input_tree(user_input, models['tree_prep'])
        predictions['tree'] = max(0, models['tree'].predict(X_tree)[0])
        X_boosting = preprocess_input_boosting(user_input, models['boosting_prep'])
        predictions['boosting'] = max(0, models['boosting'].predict(X_boosting)[0])
        
        # # Weighted ensemble
        # weights = {'linear': 0.2, 'tree': 0.4, 'boosting': 0.4}
        # predictions['ensemble'] = sum(predictions[m] * weights[m] for m in predictions)
        
        # Improved weighted ensemble
        weights = {'linear': 0.2, 'tree': 0.4, 'boosting': 0.4}
        members = ['linear', 'tree', 'boosting']  # explicit order
        w_sum = sum(weights[m] for m in members) or 1.0
        predictions['ensemble'] = sum(predictions[m] * weights[m] for m in members) / w_sum

        return predictions
    except Exception as e:
        st.error(f"Error making predictions: {e}")
        return None

def create_prediction_chart(predictions):
    """Create a chart showing predictions from different models."""
    model_names = ['Linear Model', 'Tree Model', 'Boosting Model', 'Ensemble']
    model_keys = ['linear', 'tree', 'boosting', 'ensemble']
    prices = [predictions[key] for key in model_keys]
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#1f77b4']
    fig = go.Figure(data=[go.Bar(x=model_names, y=prices, marker_color=colors,
                                 text=[f'${p:,.0f}' for p in prices], textposition='auto')])
    fig.update_layout(title='Price Predictions by Model', xaxis_title='Model Type',
                      yaxis_title='Predicted Price (SGD)', showlegend=False,
                      height=400, template='plotly_white')
    return fig

def create_market_insights_chart(reference_data, user_input, predicted_price):
    """Create market insights visualization with dynamic star positioning."""
    if not reference_data:
        return None
    sample_data = reference_data['sample_data']
    similar_properties = sample_data[
        (sample_data['flat_type'] == user_input['flat_type']) &
        (sample_data['town'] == user_input['town'])
    ]
    if len(similar_properties) < 5:
        similar_properties = sample_data[sample_data['flat_type'] == user_input['flat_type']]
    if len(similar_properties) > 0:
        fig = px.scatter(similar_properties, x='floor_area_sqm', y='resale_price',
                         color='flat_model',
                         title=f"Market Position: {user_input['flat_type']} in {user_input['town']}",
                         labels={'floor_area_sqm': 'Floor Area (sqm)', 'resale_price': 'Resale Price (SGD)'})
        fig.add_trace(go.Scatter(x=[user_input['floor_area_sqm']], y=[predicted_price],
                                 mode='markers',
                                 marker=dict(symbol='star', size=20, color='red',
                                             line=dict(width=2, color='darkred')),
                                 name='Your Property',
                                 hovertemplate=f"Your Property<br>Area: {user_input['floor_area_sqm']} sqm"
                                               f"<br>Predicted Price: ${predicted_price:,.0f}<extra></extra>"))
        fig.update_layout(template="plotly_white", height=400, showlegend=True,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                      xanchor="right", x=1))
        return fig
    return None

def main():
    """Main Streamlit application."""
    st.markdown('<h1 class="main-header">🏠 HDB Price Predictor</h1>', unsafe_allow_html=True)
    #st.markdown('<div class="info-box">Predict HDB resale prices using advanced machine learning ensemble models</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-center">Predict HDB resale prices using advanced machine learning ensemble models</div>', unsafe_allow_html=True)

    with st.spinner('Loading ML models...'):
        models = load_models()
        reference_data = load_reference_data()

    # -------------------- LEFT PANEL (sorted selectboxes) --------------------
    st.sidebar.header("🏠 Property Details")

    st.sidebar.subheader("📍 Location")
    towns_list = [
        'SENGKANG', 'WOODLANDS', 'TAMPINES', 'PUNGGOL', 'JURONG WEST',
        'YISHUN', 'BEDOK', 'HOUGANG', 'CHOA CHU KANG', 'ANG MO KIO',
        'BUKIT BATOK', 'BUKIT MERAH', 'CLEMENTI', 'GEYLANG', 'KALLANG/WHAMPOA',
        'PASIR RIS', 'QUEENSTOWN', 'SEMBAWANG', 'SERANGOON', 'TOA PAYOH',
        'BISHAN', 'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'MARINE PARADE'
    ]
    town = st.sidebar.selectbox("Town", sorted(towns_list, key=str.upper))

    st.sidebar.subheader("🏢 Property Type")
    flat_types = ['3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', '2 ROOM', '1 ROOM', 'MULTI-GENERATION']
    flat_type = st.sidebar.selectbox("Flat Type", sorted(flat_types, key=str.upper))

    flat_models = [
        'Model A', 'Improved', 'New Generation', 'Premium Apartment', 'Simplified',
        'Standard', 'Apartment', 'Maisonette', 'Model A-Maisonette', 'Adjoined flat',
        'Premium Maisonette', 'Multi Generation', 'DBSS', 'Type S1', 'Type S2',
        'Model A2', 'Terrace', 'Improved-Maisonette', 'Premium Apartment Loft', '2-room'
    ]
    flat_model = st.sidebar.selectbox("Flat Model", sorted(flat_models, key=str.upper))
    # ------------------------------------------------------------------------

    st.sidebar.subheader("📏 Specifications")
    floor_area_sqm = st.sidebar.slider("Floor Area (sqm)", min_value=30, max_value=200, value=95, step=5)
    storey_level = st.sidebar.slider("Storey Level", min_value=1, max_value=50, value=8, step=1)

    st.sidebar.subheader("⏰ Age & Lease")
    flat_age = st.sidebar.slider("Flat Age (years)", min_value=0, max_value=50, value=15, step=1,
                                 help="Remaining lease will be automatically calculated (99 - flat age)")
    remaining_lease_years = 99 - flat_age
    st.sidebar.info(f"📅 Remaining Lease: {remaining_lease_years} years")

    user_input = {
        'town': town,
        'flat_type': flat_type,
        'flat_model': flat_model,
        'floor_area_sqm': floor_area_sqm,
        'storey_level': storey_level,
        'flat_age': flat_age
    }

    #col1, col2 = st.columns([2.5, 1.5], gap="medium")
    col1, col2 = st.columns([3, 1], gap="medium")

    with col1:
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        if st.button("🔮 Predict Price", type="primary", use_container_width=True):
            with st.spinner('Generating predictions...'):
                predictions = predict_price(user_input, models)
                if predictions:
                    st.session_state.predictions = predictions
                    ensemble_price = predictions['ensemble']
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h2>💰 Predicted Price</h2>
                        <h1>${ensemble_price:,.0f}</h1>
                        <p>Ensemble Model Prediction</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.subheader("📊 Model Comparison")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.markdown(f"""<div class="metric-card"><h4>📈 Linear Model</h4>
                        <h3>${predictions['linear']:,.0f}</h3></div>""", unsafe_allow_html=True)
                    with col_b:
                        st.markdown(f"""<div class="metric-card"><h4>🌳 Tree Model</h4>
                        <h3>${predictions['tree']:,.0f}</h3></div>""", unsafe_allow_html=True)
                    with col_c:
                        st.markdown(f"""<div class="metric-card"><h4>🚀 Boosting Model</h4>
                        <h3>${predictions['boosting']:,.0f}</h3></div>""", unsafe_allow_html=True)

                    fig = create_prediction_chart(predictions)
                    st.plotly_chart(fig, use_container_width=True)

                    if reference_data:
                        st.subheader("📈 Market Insights")
                        market_chart = create_market_insights_chart(reference_data, user_input, ensemble_price)
                        if market_chart:
                            st.plotly_chart(market_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="right-panel">', unsafe_allow_html=True)
        st.subheader("📋 Property Summary")
        summary_data = {
            "Location": town,
            "Type": flat_type,
            "Model": flat_model,
            "Area": f"{floor_area_sqm} sqm",
            "Floor": f"Level {storey_level}",
            "Age": f"{flat_age} years",
            "Remaining Lease": f"{remaining_lease_years} years"
        }
        for key, value in summary_data.items():
            st.text(f"{key}: {value}")

        if st.session_state.get('predictions') is not None:
            preds = st.session_state.predictions
            ensemble_price = preds['ensemble']
            st.subheader("💡 Property Insights")
            price_per_sqm = (ensemble_price / max(floor_area_sqm, 1))
            market_segment = get_market_segment(flat_type)
            lease_health = get_lease_health(remaining_lease_years)
            bank_ok = remaining_lease_years >= 60
            hdb_ok = remaining_lease_years >= 20
            left, right = st.columns(2)
            with left:
                st.markdown('<div class="insight-label">Price per sqm</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="insight-value">${price_per_sqm:,.0f}</div>', unsafe_allow_html=True)
                st.markdown('<div class="insight-label">Market Segment</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="insight-value">{market_segment}</div>', unsafe_allow_html=True)
                st.markdown('<div class="insight-label">Lease Health</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="insight-value">{lease_health}</div>', unsafe_allow_html=True)
            with right:
                st.markdown('<div class="insight-label">Bank Loan Eligible</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="insight-value">{"✅ Yes" if bank_ok else "❌ No"}</div>', unsafe_allow_html=True)
                st.markdown('<div class="insight-label">HDB Loan Eligible</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="insight-value">{"✅ Yes" if hdb_ok else "❌ No"}</div>', unsafe_allow_html=True)
                st.markdown('<div class="insight-label">Remaining Lease</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="insight-value">{remaining_lease_years} years</div>', unsafe_allow_html=True)

        st.subheader("🤖 About the Models")
        st.info("""
        **Ensemble Approach:**
        - Linear Model: Ridge/Lasso regression
        - Tree Model: Random Forest
        - Boosting Model: XGBoost
        - Ensemble: Weighted combination

        **Data Source:**
        Singapore HDB resale transactions (2015-2025)
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    if 'predictions' not in st.session_state:
        st.session_state.predictions = None

    if st.button("🔮 Predict Price", type="primary", use_container_width=True, key="predict_hidden"):
        predictions = predict_price(user_input, models)
        if predictions:
            st.session_state.predictions = predictions

if __name__ == "__main__":
    main()
