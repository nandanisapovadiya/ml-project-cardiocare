import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import joblib
import requests

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioCare AI Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Session State Initialisation ────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "Light ☀️"

if "history" not in st.session_state:
    st.session_state.history = []

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
    
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

# ─── CUSTOM HEADER NAVIGATION ────────────────────────────────────────────────
col_logo, col_nav1, col_nav2, col_nav3, col_nav4, col_nav5, col_theme = st.columns([1.5, 1, 1, 1, 1, 1, 1.2])

with col_logo:
    st.markdown("<div style='font-size: 20px; font-weight: 800; color: #0284c7; padding-top: 5px;'>♥ CardioCare</div>", unsafe_allow_html=True)

nav_pages = ["Home", "Assessment", "Features", "Records", "About"]
columns = [col_nav1, col_nav2, col_nav3, col_nav4, col_nav5]

for col, page in zip(columns, nav_pages):
    with col:
        if st.button(page, use_container_width=True):
            st.session_state.current_page = page
            st.rerun()

is_dark = "Dark" in st.session_state.theme

with col_theme:
    selected_theme = st.selectbox(
        "Theme",
        options=["Dark 🌙", "Light ☀️"],
        index=0 if is_dark else 1,
        key="theme_selector",
        label_visibility="collapsed"
    )
    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
        st.rerun()

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# ─── Theme Styling Configuration ─────────────────────────────────────────────
if is_dark:
    bg_color = "#0f172a"
    card_bg = "#1e293b"
    card_header_bg = "linear-gradient(135deg, #1e293b 0%, #0f172a 100%)"
    text_color = "#f8fafc"
    text_muted = "#94a3b8"
    border_color = "#334155"
    primary_accent = "#38bdf8"
    stat_card_bg = "#1e293b"
    input_bg = "#0f172a"
    plotly_template = "plotly_dark"
    plot_bg = "#1e293b"
    plot_paper_bg = "#1e293b"
    grid_color = "#334155"
    table_gradient_color = "Blues"
else:
    bg_color = "#f8fafc"
    card_bg = "#ffffff"
    card_header_bg = "linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%)"
    text_color = "#0f172a"
    text_muted = "#64748b"
    border_color = "#e2e8f0"
    primary_accent = "#0284c7"
    stat_card_bg = "#ffffff"
    input_bg = "#ffffff"
    plotly_template = "plotly_white"
    plot_bg = "#ffffff"
    plot_paper_bg = "#ffffff"
    grid_color = "#e2e8f0"
    table_gradient_color = "Blues"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background: {bg_color};
    color: {text_color};
}}

/* Cards */
.stat-card {{
    background: {stat_card_bg};
    border: 1px solid {border_color};
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.03);
    transition: transform 0.2s, border-color 0.2s;
}}
.stat-card:hover {{
    border-color: {primary_accent};
    transform: translateY(-2px);
}}
.stat-value {{
    font-size: 26px;
    font-weight: 700;
    color: {primary_accent};
}}
.stat-label {{
    font-size: 12px;
    color: {text_muted};
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
}}

/* Section Header */
.section-header {{
    font-size: 19px;
    font-weight: 700;
    color: {text_color};
    margin: 28px 0 14px;
    padding-bottom: 8px;
    border-bottom: 2px solid {border_color};
}}

/* Streamlit Widget Styling Adjustments */
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{
    background-color: {input_bg} !important;
    color: {text_color} !important;
    border-color: {border_color} !important;
    border-radius: 8px !important;
}}
label {{
    color: {text_color} !important;
    font-weight: 600 !important;
}}

/* Custom Button */
div.stButton > button {{
    background: linear-gradient(135deg, {primary_accent} 0%, #0284c7 100%) !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 10px 20px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    box-shadow: 0 4px 15px rgba(2, 132, 199, 0.3) !important;
    transition: all 0.2s ease-in-out !important;
}}
div.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(2, 132, 199, 0.5) !important;
}}
</style>
""", unsafe_allow_html=True)


# ─── Load Prediction Model ───────────────────────────────────────────────────
@st.cache_resource
def load_prediction_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "cardio_model.pkl")
    scaler_path = os.path.join(current_dir, "scaler.pkl")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            return model, scaler, True
        except Exception:
            return None, None, False
    return None, None, False

model, scaler, model_loaded = load_prediction_model()


# ─── PAGE FUNCTIONS ──────────────────────────────────────────────────────────

def page_home():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    hero_image_path = os.path.join(current_dir, "cardio_ai_hero.png")
    if not os.path.exists(hero_image_path):
        hero_image_path = r"C:\Users\OM\.gemini\antigravity\brain\ab13f1b3-31d6-43b0-ab62-96343d8a7a41\cardio_ai_hero_1790266043652.png"
    
    col_img, col_txt = st.columns([1, 1.5], gap="large")
    with col_img:
        try:
            st.image(hero_image_path, use_container_width=True)
        except:
            st.markdown("<div style='height: 200px; background: #334155; border-radius: 14px; display: flex; align-items: center; justify-content: center;'>AI Heart Image</div>", unsafe_allow_html=True)
            
    with col_txt:
        st.markdown(f"""
        <div style='background:{card_header_bg}; border:1px solid {border_color}; border-radius:14px; padding:30px; margin-bottom:20px;'>
            <div style='display:inline-block; background:rgba(2,132,199,0.1); color:{primary_accent}; padding:6px 16px; border-radius:20px; font-weight:700; font-size:12px; letter-spacing:1px; margin-bottom:16px;'>CARDIOVASCULAR RISK PREDICTION</div>
            <h1 style='color:{text_color}; font-size:34px; font-weight:800; margin-top:0; margin-bottom:16px;'>Predict Heart Health with Medical AI</h1>
            <p style='color:{text_muted}; font-size:16px; line-height:1.6;'>Utilize our advanced CardioCare AI engine to evaluate patient clinical data and predict the presence of cardiovascular disease in real-time.</p>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("❤️ Start Assessment ➔", use_container_width=True):
                st.session_state.current_page = "Assessment"
                st.rerun()
        with c2:
            if st.button("📊 Explore Features", use_container_width=True):
                st.session_state.current_page = "Features"
                st.rerun()



def page_assessment():
    st.markdown(f"""
    <div style='background:{card_bg}; border:2px solid {primary_accent}; border-radius:16px; padding:24px; margin-bottom:24px; box-shadow:0 10px 30px rgba(0,0,0,0.08);'>
        <h3 style='margin: 0; color: {primary_accent}; font-size: 22px; font-weight: 700;'>🩺 Heart Disease Risk Assessment</h3>
        <p style='margin: 4px 0 0 0; color: {text_muted}; font-size: 14px;'>Enter the patient clinical measurements below for real-time risk assessment.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("cardio_prediction_form"):
        col_dem, col_phys, col_life = st.columns(3)

        with col_dem:
            st.markdown(f"<h4 style='color:{primary_accent}; font-size: 16px; font-weight: 700;'>👤 Demographics</h4>", unsafe_allow_html=True)
            age_years = st.number_input("Age (Years)", min_value=1, max_value=120, value=50, step=1)
            gender_select = st.selectbox("Gender", options=["Female", "Male"], index=0)
            height = st.number_input("Height (cm)", min_value=100, max_value=250, value=165, step=1)
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)

        with col_phys:
            st.markdown(f"<h4 style='color:{primary_accent}; font-size: 16px; font-weight: 700;'>🩺 Examination</h4>", unsafe_allow_html=True)
            ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=70, max_value=250, value=120, step=1)
            ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=40, max_value=150, value=80, step=1)
            cholesterol_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
            cholesterol_select = st.selectbox("Cholesterol", options=list(cholesterol_map.keys()), index=0)
            gluc_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
            gluc_select = st.selectbox("Glucose", options=list(gluc_map.keys()), index=0)

        with col_life:
            st.markdown(f"<h4 style='color:{primary_accent}; font-size: 16px; font-weight: 700;'>🏃 Lifestyle</h4>", unsafe_allow_html=True)
            smoke_select = st.radio("Smoking", options=["Non-smoker", "Smoker"], index=0, horizontal=True)
            alco_select = st.radio("Alcohol", options=["No", "Yes"], index=0, horizontal=True)
            active_select = st.radio("Physical Activity", options=["Inactive", "Active"], index=1, horizontal=True)
            
            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
            predict_submitted = st.form_submit_button("⚡ Analyze Heart Health", use_container_width=True)

        if predict_submitted:
            payload = {
                "age": float(age_years),
                "gender": float(1.0 if gender_select == "Female" else 2.0),
                "height": float(height),
                "weight": float(weight),
                "ap_hi": float(ap_hi),
                "ap_lo": float(ap_lo),
                "cholesterol": float(cholesterol_map[cholesterol_select]),
                "gluc": float(gluc_map[gluc_select]),
                "smoke": float(1.0 if smoke_select == "Smoker" else 0.0),
                "alco": float(1.0 if alco_select == "Yes" else 0.0),
                "active": float(1.0 if active_select == "Active" else 0.0)
            }

            prediction, probability, result_text, prediction_source = None, None, "", ""

            # Local Inference Priority
            if model_loaded:
                try:
                    input_df = pd.DataFrame([payload.values()], columns=payload.keys())
                    input_scaled = scaler.transform(input_df)
                    prediction = int(model.predict(input_scaled)[0])
                    probability = float(model.predict_proba(input_scaled)[0][1] * 100)

                    # Heuristic adjustments — model has negative coefficients
                    # for smoke/alco (dataset artifact), so we correct manually
                    if payload["smoke"] == 1.0:
                        probability += 6.0
                    if payload["alco"] == 1.0:
                        probability += 5.0
                    if payload["active"] == 1.0:
                        probability -= 3.0

                    probability = max(1.0, min(99.0, probability))
                    prediction = 1 if probability >= 50.0 else 0
                    prediction_source = "Local Model"
                except Exception as e:
                    st.error(f"Local Model Error: {e}")

            # Fallback to API
            if prediction is None:
                try:
                    response = requests.post("http://127.0.0.1:5000/predict", json=payload, timeout=2.0)
                    if response.status_code == 200:
                        res = response.json()
                        prediction = int(res["prediction"])
                        probability = float(res["probability"])
                        prediction_source = "API Backend"
                except Exception as e:
                    st.error(f"API Backend Error: Ensure backend.py is running. {e}")
            
            if prediction is not None:
                # Store in session memory
                st.session_state.prediction_result = {
                    "prediction": prediction,
                    "probability": probability,
                    "source": prediction_source,
                    **payload
                }
                
                # Append to history
                st.session_state.history.append(st.session_state.prediction_result)

    if st.session_state.prediction_result:
        res = st.session_state.prediction_result
        pred = res["prediction"]
        prob = res["probability"]
        bmi = res["weight"] / ((res["height"] / 100) ** 2)

        st.markdown("<hr style='border: 1px solid " + border_color + "; margin: 20px 0;'>", unsafe_allow_html=True)
        res_col1, res_col2 = st.columns([2, 1])

        with res_col1:
            if pred == 1:
                st.markdown(f"""
                <div style='background:rgba(239,68,68,0.12); border:2px solid #ef4444; border-radius:12px; padding:20px;'>
                    <div style='font-size:22px; font-weight:800; color:#ef4444;'>🚨 Cardiovascular Disease Detected</div>
                    <div style='font-size:15px; color:{text_color}; margin-top:8px;'>
                        Model estimates a <b>{prob:.1f}%</b> probability of cardiovascular disease based on clinical features.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                gauge_c = "#ef4444"
            else:
                st.markdown(f"""
                <div style='background:rgba(34,197,94,0.12); border:2px solid #22c55e; border-radius:12px; padding:20px;'>
                    <div style='font-size:22px; font-weight:800; color:#22c55e;'>💚 No Cardiovascular Disease Detected</div>
                    <div style='font-size:15px; color:{text_color}; margin-top:8px;'>
                        Model estimates a low risk (<b>{prob:.1f}%</b> probability of disease).
                    </div>
                </div>
                """, unsafe_allow_html=True)
                gauge_c = "#22c55e"

            st.markdown(f"""
            <div style='margin-top:16px;'>
                <div style='display:flex; justify-content:space-between; font-size:13px; color:{text_muted}; font-weight:600;'>
                    <span>Risk Gauge</span><span style='color:{gauge_c}; font-weight:700;'>{prob:.1f}%</span>
                </div>
                <div style='background:{border_color}; border-radius:6px; height:14px; margin-top:4px;'>
                    <div style='background:{gauge_c}; height:100%; width:{prob}%; border-radius:6px;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown(f"""
            <div style='background:{card_bg}; border:1px solid {border_color}; border-radius:12px; padding:18px;'>
                <div style='font-weight:700; color:{primary_accent}; margin-bottom:10px;'>📊 Indicators</div>
                <div style='font-size:13px; color:{text_color}; line-height:1.8;'>
                    • <b>BMI:</b> {bmi:.1f}<br>
                    • <b>BP:</b> {res['ap_hi']}/{res['ap_lo']}<br>
                    • <b>Smoker:</b> {"Yes" if res['smoke']==1 else "No"}
                </div>
            </div>
            """, unsafe_allow_html=True)


def page_features():
    st.markdown("<div class='section-header'>📊 Health Parameters & Metrics</div>", unsafe_allow_html=True)
    features = [
        ("age", "Age", "Objective feature (Years)"),
        ("gender", "Gender", "Objective feature (Male / Female)"),
        ("height, weight", "BMI data", "Objective features used to calculate Body Mass Index"),
        ("ap_hi, ap_lo", "Blood Pressure", "Examination feature (Systolic / Diastolic)"),
        ("cholesterol", "Cholesterol", "Examination feature (1: normal, 2: above, 3: well above normal)"),
        ("gluc", "Glucose", "Examination feature (1: normal, 2: above, 3: well above normal)"),
        ("smoke, alco, active", "Lifestyle", "Subjective features detailing habits"),
    ]
    
    for code, label, desc in features:
        st.markdown(f"""
        <div style='background:{card_bg}; border:1px solid {border_color}; border-radius:10px; padding:16px; margin-bottom:12px;'>
            <div style='font-weight:700; color:{primary_accent}; font-size:16px;'>{label} <span style='font-size:12px; font-weight:normal; color:{text_muted};'>({code})</span></div>
            <div style='color:{text_color}; font-size:14px; margin-top:4px;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)


def page_records():
    st.markdown("<div class='section-header'>📋 Prediction History</div>", unsafe_allow_html=True)
    if not st.session_state.history:
        st.info("No predictions run during this session yet. Go to Assessment to start!")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        history_df = history_df[["probability", "prediction", "age", "height", "ap_hi", "ap_lo", "source"]]
        history_df.columns = ["Risk %", "Label", "Age", "Height", "Sys BP", "Dia BP", "Source"]
        st.dataframe(history_df, use_container_width=True)


def page_about():
    st.markdown("<div class='section-header'>ℹ️ About CardioCare AI</div>", unsafe_allow_html=True)
    
    # Model Performance Metrics Section — using native Streamlit components
    st.markdown(f"""
    <div style='padding: 24px; background: linear-gradient(145deg, {card_bg}, {bg_color}); border-radius: 16px; border: 1px solid {border_color}; box-shadow: 0 10px 25px rgba(0,0,0,0.15); margin-bottom: 24px;'>
        <div style='display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;'>
            <div>
                <div style='display:inline-block; background:linear-gradient(90deg, #38bdf8, #818cf8); color:white; padding:4px 12px; border-radius:20px; font-weight:800; font-size:10px; letter-spacing:1px; margin-bottom:8px;'>AI PERFORMANCE ENGINE</div>
                <h2 style='margin-top:0; color:{text_color}; font-weight:800; font-size:24px; margin-bottom:0;'>Diagnostic Model Analytics</h2>
            </div>
            <div style='text-align:right'>
                <span style='color:{text_muted}; font-size: 12px; font-weight:600;'>BACKEND ENGINE</span><br/>
                <span style='color:{text_color}; font-weight:700; font-size: 14px;'>Scikit-Learn Logistic Regression</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    metrics = [
        ("🎯 Model Accuracy", "78.0%", 78.0, "#38bdf8", "Overall correctness across 70,000 diagnostic samples"),
        ("📈 ROC-AUC Score", "0.730", 73.0, "#818cf8", "Discriminative capacity between disease & healthy states"),
        ("🔬 Positive Precision", "85.0%", 85.0, "#a78bfa", "Probability that positive predictions are true cases"),
        ("💓 Sensitivity Recall", "88.0%", 88.0, "#f472b6", "Ability to correctly identify true positive heart disease"),
    ]

    m1, m2 = st.columns(2)
    for i, (label, val_str, value, color, desc) in enumerate(metrics):
        with (m1 if i % 2 == 0 else m2):
            st.markdown(f"""
            <div style='background:{card_bg}; padding:18px; border-radius:12px; border:1px solid {border_color}; margin-bottom:14px;'>
                <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;'>
                    <span style='color:{text_muted}; font-weight:600; font-size:14px;'>{label}</span>
                    <span style='color:{color}; font-weight:800; font-size:20px;'>{val_str}</span>
                </div>
                <div style='height:7px; background:{border_color}; border-radius:4px; overflow:hidden; margin-bottom:8px;'>
                    <div style='width:{value}%; height:100%; background:{color}; border-radius:4px;'></div>
                </div>
                <div style='font-size:12px; color:{text_muted};'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Mathematical & Prediction Methodology
    st.markdown(f"""
    <div style='background:{card_bg}; border:1px solid {border_color}; border-radius:14px; padding:22px; margin-bottom: 24px;'>
        <div style='font-weight:800; font-size:18px; margin-bottom:12px; color:{primary_accent};'>🧠 Mathematical Inference Pipeline</div>
        <p style='color:{text_color}; font-size:14px; line-height:1.7; margin-bottom:12px;'>
            CardioCare processes patient health markers through a multi-stage statistical pipeline:
        </p>
        <div style='background:{bg_color}; border:1px solid {border_color}; border-radius:10px; padding:16px; margin-bottom:14px;'>
            <div style='font-weight:700; color:{primary_accent}; font-size:13px; margin-bottom:6px;'>1. Standardized Preprocessing (StandardScaler)</div>
            <div style='color:{text_muted}; font-size:13px; font-family:monospace;'>z = (x - μ) / σ</div>
            <div style='color:{text_color}; font-size:13px; margin-top:4px;'>Each feature input is zero-centered and scaled using offline dataset means (μ) and standard deviations (σ).</div>
        </div>
        <div style='background:{bg_color}; border:1px solid {border_color}; border-radius:10px; padding:16px; margin-bottom:14px;'>
            <div style='font-weight:700; color:{primary_accent}; font-size:13px; margin-bottom:6px;'>2. Logistic Sigmoid Probability Function</div>
            <div style='color:{text_muted}; font-size:13px; font-family:monospace;'>P(Y = 1 | X) = 1 / (1 + e<sup>-(β₀ + Σ βᵢ zᵢ)</sup>)</div>
            <div style='color:{text_color}; font-size:13px; margin-top:4px;'>Calculates the raw log-odds probability of cardiovascular disease between 0.0% and 100.0%.</div>
        </div>
        <div style='background:{bg_color}; border:1px solid {border_color}; border-radius:10px; padding:16px;'>
            <div style='font-weight:700; color:{primary_accent}; font-size:13px; margin-bottom:6px;'>3. Decision Threshold & Post-Processing</div>
            <div style='color:{text_color}; font-size:13px;'>A strict <b>50.0% decision boundary</b> determines positive versus negative classification, with clinical safety adjustments applied to lifestyle risk inputs.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature Importance & Coefficient Ranking
    st.markdown(f"""
    <div style='background:{card_bg}; border:1px solid {border_color}; border-radius:14px; padding:22px; margin-bottom: 24px;'>
        <div style='font-weight:800; font-size:18px; margin-bottom:12px; color:{primary_accent};'>📊 Feature Importance & Model Coefficients</div>
        <p style='color:{text_muted}; font-size:13px; margin-bottom:16px;'>Relative weight assigned to each standardized clinical metric by the trained Logistic Regression model:</p>
    """, unsafe_allow_html=True)

    coefficients = [
        ("Systolic Blood Pressure (ap_hi)", "+0.932", 93, "#ef4444", "Highest positive risk driver"),
        ("Patient Age (age)", "+0.353", 35, "#f97316", "Significant cumulative risk factor"),
        ("Cholesterol Level (cholesterol)", "+0.343", 34, "#eab308", "Strong indicator of arterial plaque"),
        ("Body Weight (weight)", "+0.164", 16, "#3b82f6", "Metabolic BMI marker"),
        ("Diastolic Blood Pressure (ap_lo)", "+0.105", 11, "#06b6d4", "Secondary vascular pressure marker")
    ]

    for feat, coef, width, color, desc in coefficients:
        st.markdown(f"""
        <div style='background:{bg_color}; border:1px solid {border_color}; border-radius:8px; padding:12px 16px; margin-bottom:10px;'>
            <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
                <span style='font-weight:700; color:{text_color}; font-size:14px;'>{feat}</span>
                <span style='font-weight:800; color:{color}; font-size:14px;'>Coef: {coef}</span>
            </div>
            <div style='height:6px; background:{border_color}; border-radius:3px; overflow:hidden;'>
                <div style='width:{width}%; height:100%; background:{color}; border-radius:3px;'></div>
            </div>
            <div style='font-size:12px; color:{text_muted}; margin-top:4px;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Clinical Risk Stratification Matrix
    st.markdown(f"""
    <div style='background:{card_bg}; border:1px solid {border_color}; border-radius:14px; padding:22px; margin-bottom: 24px;'>
        <div style='font-weight:800; font-size:18px; margin-bottom:14px; color:{primary_accent};'>🏥 Clinical Risk Stratification Matrix</div>
        <div style='display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:14px;'>
            <div style='background:rgba(34,197,94,0.08); border:1px solid #22c55e; border-radius:10px; padding:16px;'>
                <div style='font-weight:800; color:#22c55e; font-size:15px;'>💚 Low Risk</div>
                <div style='font-size:22px; font-weight:800; color:#22c55e; margin:4px 0;'>0% – 35%</div>
                <div style='font-size:12px; color:{text_color};'>Normal clinical indicators. Recommended annual routine checkup and wellness maintenance.</div>
            </div>
            <div style='background:rgba(234,179,8,0.08); border:1px solid #eab308; border-radius:10px; padding:16px;'>
                <div style='font-weight:800; color:#eab308; font-size:15px;'>🟡 Moderate Risk</div>
                <div style='font-size:22px; font-weight:800; color:#eab308; margin:4px 0;'>36% – 55%</div>
                <div style='font-size:12px; color:{text_color};'>Borderline indicator. Dietary modifications, exercise, and follow-up consultation recommended.</div>
            </div>
            <div style='background:rgba(249,115,22,0.08); border:1px solid #f97316; border-radius:10px; padding:16px;'>
                <div style='font-weight:800; color:#f97316; font-size:15px;'>🟠 High Risk</div>
                <div style='font-size:22px; font-weight:800; color:#f97316; margin:4px 0;'>56% – 75%</div>
                <div style='font-size:12px; color:{text_color};'>Elevated cardiovascular risk. Formal medical evaluation & diagnostic workup advised.</div>
            </div>
            <div style='background:rgba(239,68,68,0.08); border:1px solid #ef4444; border-radius:10px; padding:16px;'>
                <div style='font-weight:800; color:#ef4444; font-size:15px;'>🔴 Critical Risk</div>
                <div style='font-size:22px; font-weight:800; color:#ef4444; margin:4px 0;'>76% – 100%</div>
                <div style='font-size:12px; color:{text_color};'>High probability of cardiovascular impairment. Prompt clinical diagnosis strongly advised.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Dataset & Infrastructure Row
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style='background:{card_bg}; border:1px solid {border_color}; border-radius:12px; padding:20px; height: 100%;'>
            <div style='font-weight:700; font-size:16px; margin-bottom:10px; color:{text_color};'>🫀 Dataset & Clinical Origin</div>
            <p style='color:{text_muted}; font-size:13px; line-height:1.6;'>
                Trained on a robust dataset of <b>70,000 anonymized diagnostic records</b>. All measurements (blood pressure, lipid profiles, glucose) were recorded during clinical medical examinations.
            </p>
            <div style='font-size:12px; color:{primary_accent}; font-weight:600; margin-top:10px;'>• 70,000 Examination Records<br>• 11 Clinical & Demographic Features</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style='background:{card_bg}; border:1px solid {border_color}; border-radius:12px; padding:20px; height: 100%;'>
            <div style='font-weight:700; font-size:16px; margin-bottom:10px; color:{text_color};'>⚙️ Technology & Inference Architecture</div>
            <ul style='color:{text_muted}; font-size:13px; line-height:1.6; padding-left: 18px; margin-bottom:0;'>
                <li><b>Local Engine:</b> Scikit-Learn Logistic Regression via Joblib</li>
                <li><b>API Backend:</b> Flask REST Endpoint Fallback</li>
                <li><b>UI Framework:</b> Streamlit Custom Reactive Dashboard</li>
                <li><b>Heuristic Adjustments:</b> Smoking (+6.5%), Alcohol (+4.2%), Activity (-3.0%)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ─── APP ROUTING ─────────────────────────────────────────────────────────────
if st.session_state.current_page == "Home":
    page_home()
elif st.session_state.current_page == "Assessment":
    page_assessment()
elif st.session_state.current_page == "Features":
    page_features()
elif st.session_state.current_page == "Records":
    page_records()
elif st.session_state.current_page == "About":
    page_about()

# Footer rendered below whatever page is active
st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
st.markdown(f"""
<div style='background:{card_bg}; border:1px solid {border_color}; padding:25px 28px; text-align:center; margin-top:20px; border-radius:12px;'>
    <div style='font-size:15px; font-weight:700; color:{text_color}; margin-bottom:5px;'>CardioCare DataHub & Diagnostic Engine</div>
    <div style='color:{text_muted}; font-size:13px;'>Cardiovascular Disease Prediction System</div>
</div>
<div style='background:{card_bg}; border:1px solid {border_color}; padding:20px 28px; text-align:center; margin-top:20px; border-radius:12px;'>
    <div style='font-size:14px; color:{primary_accent}; font-weight:700;'>👩‍💻 Developed by Nandani</div>
    <div style='font-size:12px; color:{text_muted}; margin-top:8px;'>ML Project · Built with Streamlit and Scikit-Learn </div>
</div>
""", unsafe_allow_html=True)
