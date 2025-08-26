"""
File: streamlit_app_mobile.py
Purpose: ITI105 Price Predictor – Responsive (Desktop + Mobile)
- Mobile-first layout, single header, stable query params, safe Plotly figures
- Preserves original models, preprocessing, and weighted ensemble
Author: Team AlgoRiddler - ITI105 Project
Date: Aug 2025
Dependencies: streamlit, pandas, numpy, joblib, plotly
Input: User inputs via web interface
Output: Price predictions with confidence intervals and market insights
"""

from __future__ import annotations
import warnings
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

# ---------- Paths ----------
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

PROJECT_ROOT = SCRIPT_DIR
expected = PROJECT_ROOT / "data" / "raw" / "base_hdb_resale_prices_2015Jan-2025Jun.csv"
if not expected.exists() and (SCRIPT_DIR.parent / "data" / "raw" / "base_hdb_resale_prices_2015Jan-2025Jun.csv").exists():
    PROJECT_ROOT = SCRIPT_DIR.parent

DATA_DIR           = PROJECT_ROOT / "data"
RAW_DATA_DIR       = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR         = PROJECT_ROOT / "models"
OUTPUT_DIR         = PROJECT_ROOT / "output"
for p in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ---------- Page config ----------
st.set_page_config(
    page_title="HDB Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Device width detection (no st.stop; no blank page) ----------
def detect_device_width(default: int = 1200) -> int:
    """Try to read ?w= from URL. If missing, inject a tiny JS to set it, then
    return a conservative default for this first run so the page still renders."""
    w = st.query_params.get("w")
    if w:
        try:
            return int(w)
        except Exception:
            pass

    # Inject once to set ?w=<innerWidth> on the next run
    components.html(
        """
        <script>
          const params = new URLSearchParams(window.location.search);
          if (!params.get('w')) {
            const w = window.innerWidth || document.documentElement.clientWidth || 1200;
            params.set('w', w);
            const newQuery = '?' + params.toString();
            window.history.replaceState(null, '', window.location.pathname + newQuery);
            // no hard reload -> Streamlit will rerun automatically on next interaction
          }
        </script>
        """,
        height=0, width=0,
    )
    return default  # first run fallback

SCREEN_WIDTH = detect_device_width()
IS_MOBILE = SCREEN_WIDTH < 768

# ---------- CSS ----------
st.markdown(
    """
<style>
.main-header {
  font-size: 3rem;
  font-weight: 800;
  text-align: center;
  background: linear-gradient(90deg, #1f77b4, #ff7f0e);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 1.0rem;
}
.info-center{
  max-width:1500px;
  text-align:center;
  font-size:1.05rem;
  font-weight:600;
  background:#f8f9fa;
  padding:0.7rem 1rem;
  border-radius:10px;
  border-left:4px solid #17a2b8;
  margin-bottom: 1rem;
}
.prediction-card {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 1.25rem;
  border-radius: 14px;
  color: white;
  text-align: center;
  margin: 0.75rem 0;
}
.metric-card {
  background: white;
  padding: 0.9rem;
  border-radius: 10px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.08);
  border-left: 4px solid #1f77b4;
  margin: 0.5rem 0;
}
.main-content, .right-panel {
  background: white;
  border-radius: 10px;
  padding: 1rem;
  margin: 0.5rem;
  box-shadow: 0 2px 4px rgba(0,0,0,0.06);
}
.insight-label { font-size: 0.85rem; color: #6c757d; margin-bottom: 0.25rem; }
.insight-value { font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; }
.block-container { padding-top: 0.75rem; }

/* Mobile tweaks */
@media (max-width: 768px) {
  [data-testid="stSidebar"] { display: none; }
  .main-header { font-size: 2rem; margin-bottom: 0.6rem; }
  .info-center { font-size: 0.95rem; padding: 0.55rem 0.8rem; }
  .prediction-card { padding: 0.9rem; }
  .metric-card { padding: 0.75rem; }
  .block-container { padding: 0.5rem; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Header (single source; call once from main) ----------
def render_header():
    st.markdown('<h1 class="main-header">🏠 HDB Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-center">Predict HDB resale prices using advanced machine learning ensemble models</div>',
        unsafe_allow_html=True
    )

# ---------- Data & models ----------
@st.cache_data
def load_reference_data(is_mobile: bool):
    try:
        p = RAW_DATA_DIR / "base_hdb_resale_prices_2015Jan-2025Jun.csv"
        if p.exists():
            df = pd.read_csv(p)
            n = 600 if is_mobile else 1000  # smaller sample on mobile
            return {"sample_data": df.sample(min(n, len(df)), random_state=42)}
        return None
    except Exception as e:
        st.warning(f"Could not load reference data: {e}")
        return None

@st.cache_resource
def load_models():
    m = {}
    try:
        m["linear"]   = joblib.load(MODELS_DIR / "best_linear_model.joblib")
        m["tree"]     = joblib.load(MODELS_DIR / "best_tree_model.joblib")
        m["boosting"] = joblib.load(MODELS_DIR / "best_boosting_model.joblib")
        m["linear_prep"]   = joblib.load(MODELS_DIR / "linear_preprocessing.joblib")
        m["tree_prep"]     = joblib.load(MODELS_DIR / "tree_preprocessing.joblib")
        m["boosting_prep"] = joblib.load(MODELS_DIR / "boosting_preprocessing.joblib")
        return m
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# ---------- Domain helpers ----------
def get_flat_type_ordinal(flat_type):
    mapping = {'1 ROOM':1,'2 ROOM':2,'3 ROOM':3,'4 ROOM':4,'5 ROOM':5,'EXECUTIVE':6,'MULTI-GENERATION':7}
    return mapping.get(flat_type, 4)

def get_room_count(flat_type):
    mapping = {'1 ROOM':1,'2 ROOM':2,'3 ROOM':3,'4 ROOM':4,'5 ROOM':5,'EXECUTIVE':5,'MULTI-GENERATION':5}
    return mapping.get(flat_type, 4)

def get_market_segment(flat_type):
    if flat_type in ['1 ROOM','2 ROOM']: return 'Budget'
    if flat_type in ['3 ROOM','4 ROOM']: return 'Mass Market'
    if flat_type == '5 ROOM': return 'Premium'
    return 'Luxury'

def get_lease_health(remaining_years):
    if remaining_years >= 80: return 'Excellent'
    if remaining_years >= 60: return 'Good'
    if remaining_years >= 40: return 'Fair'
    if remaining_years >= 20: return 'Poor'
    return 'Critical'

def get_storey_category(storey_level):
    if storey_level <= 3: return 'Low'
    if storey_level <= 10: return 'Mid'
    return 'High'

# ---------- Preprocessing ----------
def preprocess_input_linear(user_input, preprocessing_objects):
    features = {}
    features['floor_area_sqm'] = user_input['floor_area_sqm']
    features['storey_level'] = user_input['storey_level']
    features['flat_age'] = user_input['flat_age']
    features['remaining_lease_years'] = 99 - user_input['flat_age']
    features['flat_type_ordinal'] = get_flat_type_ordinal(user_input['flat_type'])
    features['transaction_year'] = 2025
    features['lease_used_ratio'] = user_input['flat_age']/99.0
    features['lease_remaining_ratio'] = features['remaining_lease_years']/99.0
    features['area_per_room'] = user_input['floor_area_sqm']/get_room_count(user_input['flat_type'])
    features['bank_loan_eligible'] = 1 if features['remaining_lease_years'] >= 60 else 0
    features['hdb_loan_eligible'] = 1 if features['remaining_lease_years'] >= 20 else 0
    features['cash_buyers_only'] = 1 if features['remaining_lease_years'] < 20 else 0

    categorical_features = ['town','flat_type','flat_model','market_segment','lease_health','storey_category']
    feature_names = preprocessing_objects.get('feature_names', [])
    one_hot_categories = preprocessing_objects.get('one_hot_categories')

    for cat in categorical_features:
        if cat == 'market_segment': val = get_market_segment(user_input['flat_type'])
        elif cat == 'lease_health': val = get_lease_health(features['remaining_lease_years'])
        elif cat == 'storey_category': val = get_storey_category(user_input['storey_level'])
        else: val = user_input[cat]

        if one_hot_categories and (cat in one_hot_categories):
            for c in one_hot_categories[cat]:
                features[f'{cat}_{c}'] = 1 if str(val) == str(c) else 0
        else:
            prefix = f'{cat}_'
            candidates = [c for c in feature_names if c.startswith(prefix)]
            if candidates:
                target = f'{prefix}{val}'
                for c in candidates:
                    features[c] = 1 if c == target else 0

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
                for m in [c for c in scaler_cols if c not in feature_df.columns]:
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
    features = {}
    features['floor_area_sqm'] = user_input['floor_area_sqm']
    features['storey_level'] = user_input['storey_level']
    features['flat_age'] = user_input['flat_age']
    features['remaining_lease_years'] = 99 - user_input['flat_age']
    features['flat_type_ordinal'] = get_flat_type_ordinal(user_input['flat_type'])
    features['transaction_year'] = 2025
    features['lease_used_ratio'] = user_input['flat_age']/99.0
    features['lease_remaining_ratio'] = features['remaining_lease_years']/99.0
    features['area_per_room'] = user_input['floor_area_sqm']/get_room_count(user_input['flat_type'])
    features['bank_loan_eligible'] = 1 if features['remaining_lease_years'] >= 60 else 0
    features['hdb_loan_eligible'] = 1 if features['remaining_lease_years'] >= 20 else 0
    features['cash_buyers_only'] = 1 if features['remaining_lease_years'] < 20 else 0

    categorical_features = ['town','flat_type','flat_model','market_segment','lease_health','storey_category']
    label_encoders = preprocessing_objects.get('label_encoders', {})
    feature_names = preprocessing_objects.get('feature_names', [])

    for cat in categorical_features:
        if cat == 'market_segment': val = get_market_segment(user_input['flat_type'])
        elif cat == 'lease_health': val = get_lease_health(features['remaining_lease_years'])
        elif cat == 'storey_category': val = get_storey_category(user_input['storey_level'])
        else: val = user_input[cat]
        if cat in label_encoders:
            le = label_encoders[cat]
            try:
                features[f'{cat}_encoded'] = le.transform([str(val)])[0]
            except ValueError:
                features[f'{cat}_encoded'] = 0

    feature_df = pd.DataFrame([features]).reindex(columns=feature_names, fill_value=0)
    return feature_df.values

def preprocess_input_boosting(user_input, preprocessing_objects):
    features = {}
    features['floor_area_sqm'] = user_input['floor_area_sqm']
    features['storey_level'] = user_input['storey_level']
    features['flat_age'] = user_input['flat_age']
    features['remaining_lease_years'] = 99 - user_input['flat_age']
    features['flat_type_ordinal'] = get_flat_type_ordinal(user_input['flat_type'])
    features['transaction_year'] = 2025
    features['lease_used_ratio'] = user_input['flat_age']/99.0
    features['lease_remaining_ratio'] = features['remaining_lease_years']/99.0
    features['area_per_room'] = user_input['floor_area_sqm']/get_room_count(user_input['flat_type'])
    features['bank_loan_eligible'] = 1 if features['remaining_lease_years'] >= 60 else 0
    features['hdb_loan_eligible'] = 1 if features['remaining_lease_years'] >= 20 else 0
    features['cash_buyers_only'] = 1 if features['remaining_lease_years'] < 20 else 0

    # Target-encode high-cardinality
    target_encode_features = ['town','flat_model']
    target_encoders = preprocessing_objects.get('target_encoders', {})
    for f in target_encode_features:
        if f in target_encoders:
            te = target_encoders[f]
            tmp = pd.DataFrame({f:[user_input[f]]})
            try:
                features[f'{f}_target_encoded'] = te.transform(tmp).ravel()[0]
            except Exception:
                features[f'{f}_target_encoded'] = 500000

    # Label-encode low-cardinality
    label_encode_features = ['flat_type','market_segment','lease_health']
    label_encoders = preprocessing_objects.get('label_encoders', {})
    for f in label_encode_features:
        if f == 'market_segment': val = get_market_segment(user_input['flat_type'])
        elif f == 'lease_health': val = get_lease_health(99 - user_input['flat_age'])
        else: val = user_input[f]
        if f in label_encoders:
            le = label_encoders[f]
            try:
                features[f'{f}_encoded'] = le.transform([str(val)])[0]
            except ValueError:
                features[f'{f}_encoded'] = 0

    feature_names = preprocessing_objects.get('feature_names', [])
    feature_df = pd.DataFrame([features]).reindex(columns=feature_names, fill_value=0)
    return feature_df.values

# ---------- Prediction ----------
def predict_price(user_input, models):
    predictions = {}
    try:
        X_lin = preprocess_input_linear(user_input, models['linear_prep'])
        predictions['linear'] = max(0, models['linear'].predict(X_lin)[0])

        X_tree = preprocess_input_tree(user_input, models['tree_prep'])
        predictions['tree'] = max(0, models['tree'].predict(X_tree)[0])

        X_boost = preprocess_input_boosting(user_input, models['boosting_prep'])
        predictions['boosting'] = max(0, models['boosting'].predict(X_boost)[0])

        # Weighted ensemble with explicit members + normalization
        weights = {'linear': 0.2, 'tree': 0.4, 'boosting': 0.4}
        members = ['linear', 'tree', 'boosting']
        w_sum = sum(weights[m] for m in members) or 1.0
        predictions['ensemble'] = sum(predictions[m] * weights[m] for m in members) / w_sum
        return predictions
    except Exception as e:
        st.error(f"Error making predictions: {e}")
        return None

# ---------- Charts ----------
def create_prediction_chart(predictions, height=400):
    names  = ['Linear Model','Tree Model','Boosting Model','Ensemble']
    keys   = ['linear','tree','boosting','ensemble']
    prices = [predictions[k] for k in keys]
    colors = ['#ff7f0e','#2ca02c','#d62728','#1f77b4']

    fig = go.Figure(data=[go.Bar(
        x=names, y=prices, marker_color=colors,
        text=[f'${p:,.0f}' for p in prices], textposition='auto'
    )])
    fig.update_layout(
        title='Price Predictions by Model',
        xaxis_title='Model Type', yaxis_title='Predicted Price (SGD)',
        showlegend=False, height=height, template='plotly_white'
    )
    return fig

def create_market_insights_chart(reference_data, user_input, predicted_price, height=400):
    # Always return a valid figure (never None)
    fig = go.Figure()
    fig.update_layout(template="plotly_white", height=height)

    if not reference_data or "sample_data" not in reference_data:
        fig.update_layout(
            title="Market Position (no reference data available)",
            annotations=[dict(text="No reference data to display", x=0.5, y=0.5,
                              xref="paper", yref="paper", showarrow=False)]
        )
        return fig

    df = reference_data["sample_data"]

    similar = df[(df.get('flat_type') == user_input['flat_type']) &
                 (df.get('town') == user_input['town'])]
    if similar is None or len(similar) < 5:
        similar = df[df.get('flat_type') == user_input['flat_type']]

    if similar is None or len(similar) == 0:
        fig.update_layout(
            title=f"Market Position: {user_input['flat_type']} in {user_input['town']}",
            annotations=[dict(text="No similar transactions found", x=0.5, y=0.5,
                              xref="paper", yref="paper", showarrow=False)]
        )
        return fig

    fig = px.scatter(
        similar,
        x='floor_area_sqm', y='resale_price', color='flat_model',
        title=f"Market Position: {user_input['flat_type']} in {user_input['town']}",
        labels={'floor_area_sqm':'Floor Area (sqm)','resale_price':'Resale Price (SGD)'}
    )
    fig.add_trace(go.Scatter(
        x=[user_input['floor_area_sqm']], y=[predicted_price], mode='markers',
        marker=dict(symbol='star', size=20, color='red', line=dict(width=2, color='darkred')),
        name='Your Property',
        hovertemplate=(
            "Your Property<br>"
            f"Area: {user_input['floor_area_sqm']} sqm<br>"
            f"Predicted: ${predicted_price:,.0f}<extra></extra>"
        )
    ))
    fig.update_layout(
        template="plotly_white", height=height, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# ---------- Shared inputs ----------
TOWNS = [
    'SENGKANG','WOODLANDS','TAMPINES','PUNGGOL','JURONG WEST',
    'YISHUN','BEDOK','HOUGANG','CHOA CHU KANG','ANG MO KIO',
    'BUKIT BATOK','BUKIT MERAH','CLEMENTI','GEYLANG','KALLANG/WHAMPOA',
    'PASIR RIS','QUEENSTOWN','SEMBAWANG','SERANGOON','TOA PAYOH',
    'BISHAN','BUKIT PANJANG','BUKIT TIMAH','CENTRAL AREA','MARINE PARADE'
]
FLAT_TYPES  = ['3 ROOM','4 ROOM','5 ROOM','EXECUTIVE','2 ROOM','1 ROOM','MULTI-GENERATION']
FLAT_MODELS = [
    'Model A','Improved','New Generation','Premium Apartment','Simplified',
    'Standard','Apartment','Maisonette','Model A-Maisonette','Adjoined flat',
    'Premium Maisonette','Multi Generation','DBSS','Type S1','Type S2',
    'Model A2','Terrace','Improved-Maisonette','Premium Apartment Loft','2-room'
]

def render_inputs(container):
    with container:
        st.subheader("📍 Location")
        town = st.selectbox("Town", sorted(TOWNS, key=str.upper))

        st.subheader("🏢 Property Type")
        flat_type = st.selectbox("Flat Type", sorted(FLAT_TYPES, key=str.upper))
        flat_model = st.selectbox("Flat Model", sorted(FLAT_MODELS, key=str.upper))

        st.subheader("📏 Specifications")
        floor_area_sqm = st.slider("Floor Area (sqm)", 30, 200, 95, 5)

        st.subheader("⏰ Age & Lease")
        storey_level = st.slider("Storey Level", 1, 50, 8, 1)
        flat_age = st.slider(
            "Flat Age (years)", 0, 50, 15, 1,
            help="Remaining lease will be automatically calculated (99 - flat age)"
        )
        remaining_lease = 99 - flat_age
        st.info(f"📅 Remaining Lease: {remaining_lease} years")

        user_input = {
            'town': town,
            'flat_type': flat_type,
            'flat_model': flat_model,
            'floor_area_sqm': floor_area_sqm,
            'storey_level': storey_level,
            'flat_age': flat_age
        }
        return user_input, remaining_lease

# ---------- Desktop layout ----------
def render_desktop(models, reference_data):
    # Sidebar inputs
    st.sidebar.header("🏠 Property Details")
    user_input, remaining_lease = render_inputs(st.sidebar)

    # Two columns
    col1, col2 = st.columns([3, 1], gap="medium")

    with col1:
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        if st.button("🔮 Predict Price", type="primary", use_container_width=True):
            with st.spinner('Generating predictions...'):
                preds = predict_price(user_input, models)
                if preds:
                    st.session_state.predictions = preds
                    ens = preds['ensemble']
                    st.markdown(
                        f"""
                        <div class="prediction-card">
                            <h2>💰 Predicted Price</h2>
                            <h1>${ens:,.0f}</h1>
                            <p>Ensemble Model Prediction</p>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                    st.subheader("📊 Model Comparison")
                    c1,c2,c3 = st.columns(3)
                    with c1: st.markdown(f"""<div class="metric-card"><h4>📈 Linear Model</h4><h3>${preds['linear']:,.0f}</h3></div>""", unsafe_allow_html=True)
                    with c2: st.markdown(f"""<div class="metric-card"><h4>🌳 Tree Model</h4><h3>${preds['tree']:,.0f}</h3></div>""", unsafe_allow_html=True)
                    with c3: st.markdown(f"""<div class="metric-card"><h4>🚀 Boosting Model</h4><h3>${preds['boosting']:,.0f}</h3></div>""", unsafe_allow_html=True)

                    st.plotly_chart(create_prediction_chart(preds, height=400), use_container_width=True)

                    if reference_data:
                        st.subheader("📈 Market Insights")
                        st.plotly_chart(
                            create_market_insights_chart(reference_data, user_input, ens, height=400),
                            use_container_width=True
                        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="right-panel">', unsafe_allow_html=True)
        st.subheader("📋 Property Summary")
        summary = {
            "Location": user_input['town'],
            "Type": user_input['flat_type'],
            "Model": user_input['flat_model'],
            "Area": f"{user_input['floor_area_sqm']} sqm",
            "Floor": f"Level {user_input['storey_level']}",
            "Age": f"{user_input['flat_age']} years",
            "Remaining Lease": f"{remaining_lease} years",
        }
        for k,v in summary.items():
            st.text(f"{k}: {v}")

        if st.session_state.get('predictions') is not None:
            preds = st.session_state.predictions
            ens = preds['ensemble']
            st.subheader("💡 Property Insights")
            price_psm = ens / max(user_input['floor_area_sqm'], 1)
            market_seg = get_market_segment(user_input['flat_type'])
            lease_health = get_lease_health(remaining_lease)
            bank_ok = remaining_lease >= 60
            hdb_ok  = remaining_lease >= 20
            l, r = st.columns(2)
            with l:
                st.markdown('<div class="insight-label">Price per sqm</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="insight-value">${price_psm:,.0f}</div>', unsafe_allow_html=True)
                st.markdown('<div class="insight-label">Market Segment</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="insight-value">{market_seg}</div>', unsafe_allow_html=True)
                st.markdown('<div class="insight-label">Lease Health</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="insight-value">{lease_health}</div>', unsafe_allow_html=True)
            with r:
                st.markdown('<div class="insight-label">Bank Loan Eligible</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="insight-value">{"✅ Yes" if bank_ok else "❌ No"}</div>', unsafe_allow_html=True)
                st.markdown('<div class="insight-label">HDB Loan Eligible</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="insight-value">{"✅ Yes" if hdb_ok else "❌ No"}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="insight-label">Remaining Lease</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="insight-value">{remaining_lease} years</div>', unsafe_allow_html=True)

        st.subheader("🤖 About the Models")
        st.info("""**Ensemble Approach:**
- Linear Model: Ridge/Lasso regression
- Tree Model: Random Forest
- Boosting Model: XGBoost
- Ensemble: Weighted combination

**Data Source:** Singapore HDB resale transactions (2015–2025)
""")
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- Mobile layout ----------
def render_mobile(models, reference_data):
    exp = st.expander("🔧 Property Details", expanded=True)
    user_input, remaining_lease = render_inputs(exp)

    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        with st.spinner('Generating predictions...'):
            preds = predict_price(user_input, models)
            if preds:
                st.session_state.predictions = preds

    preds = st.session_state.get("predictions")
    if preds:
        ens = preds['ensemble']
        st.markdown(
            f"""
            <div class="prediction-card">
                <h2>💰 Predicted Price</h2>
                <h1>${ens:,.0f}</h1>
                <p>Ensemble Model Prediction</p>
            </div>""",
            unsafe_allow_html=True,
        )

        st.subheader("📊 Model Comparison")
        st.markdown(f"""<div class="metric-card"><h4>📈 Linear Model</h4><h3>${preds['linear']:,.0f}</h3></div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="metric-card"><h4>🌳 Tree Model</h4><h3>${preds['tree']:,.0f}</h3></div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="metric-card"><h4>🚀 Boosting Model</h4><h3>${preds['boosting']:,.0f}</h3></div>""", unsafe_allow_html=True)

        st.plotly_chart(create_prediction_chart(preds, height=320), use_container_width=True)

        if reference_data:
            st.subheader("📈 Market Insights")
            fig = create_market_insights_chart(reference_data, user_input, ens, height=320)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 Property Summary")
        summary = {
            "Location": user_input['town'],
            "Type": user_input['flat_type'],
            "Model": user_input['flat_model'],
            "Area": f"{user_input['floor_area_sqm']} sqm",
            "Floor": f"Level {user_input['storey_level']}",
            "Age": f"{user_input['flat_age']} years",
            "Remaining Lease": f"{remaining_lease} years",
        }
        for k, v in summary.items():
            st.text(f"{k}: {v}")

        st.subheader("💡 Property Insights")
        price_psm = ens / max(user_input['floor_area_sqm'], 1)
        market_seg = get_market_segment(user_input['flat_type'])
        lease_health = get_lease_health(remaining_lease)
        bank_ok = remaining_lease >= 60
        hdb_ok  = remaining_lease >= 20

        st.markdown('<div class="insight-label">Price per sqm</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight-value">${price_psm:,.0f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="insight-label">Market Segment</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight-value">{market_seg}</div>', unsafe_allow_html=True)
        st.markdown('<div class="insight-label">Lease Health</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight-value">{lease_health}</div>', unsafe_allow_html=True)
        st.markdown('<div class="insight-label">Bank Loan Eligible</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight-value">{"✅ Yes" if bank_ok else "❌ No"}</div>', unsafe_allow_html=True)
        st.markdown('<div class="insight-label">HDB Loan Eligible</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight-value">{"✅ Yes" if hdb_ok else "❌ No"}</div>', unsafe_allow_html=True)

    # Re-detect layout (rotation etc.)
    with st.expander("⚙️ Layout options"):
        if st.button("Re-detect screen size", use_container_width=True):
            if "w" in st.query_params:
                del st.query_params["w"]
            components.html(
                """
                <script>
                  const params = new URLSearchParams(window.location.search);
                  params.delete('w');
                  const qs = params.toString();
                  window.history.replaceState(null, '', window.location.pathname + (qs ? '?' + qs : ''));
                </script>
                """,
                height=0, width=0
            )
            st.rerun()

# ---------- Main ----------
def main():
    # Single header render
    st.markdown('<h1 class="main-header">🏠 HDB Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-center">Predict HDB resale prices using advanced machine learning ensemble models</div>',
        unsafe_allow_html=True
    )

    with st.spinner("Loading ML models & data…"):
        models = load_models()
        reference_data = load_reference_data(IS_MOBILE)

    if 'predictions' not in st.session_state:
        st.session_state.predictions = None

    if IS_MOBILE:
        render_mobile(models, reference_data)
    else:
        render_desktop(models, reference_data)

if __name__ == "__main__":
    main()
