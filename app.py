import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image

# Load the trained ML model
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Valid teams list
valid_teams = [
    'Chennai Super Kings', 'Delhi Capitals', 'Kolkata Knight Riders', 'Mumbai Indians',
    'Punjab Kings', 'Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad',
    'Gujarat Titans', 'Lucknow Super Giants'
]

# Helper function for encoding inputs
def encode_inputs(input_data):
    input_df = pd.DataFrame([input_data])
    input_encoded = pd.get_dummies(input_df)
    train_columns = model.feature_names_in_
    input_aligned = input_encoded.reindex(columns=train_columns, fill_value=0)
    return input_aligned.to_numpy()

# Page Config
st.set_page_config(page_title="IPL Score Predictor", layout="wide", page_icon="🏏")

# Custom Header & Styling
st.markdown('''
    <style>
        .main-title {
            font-size: 3em;
            color: #42d0fa;
            text-align: center;
            font-weight: bold;
            text-shadow: 2px 2px #2c3e50;
        }
        .subtitle {
            text-align: center;
            font-size: 1.2em;
            color: #FFFFFF;
        }
        .stButton>button {
            background-color: #e74c3c;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 0.5em 1em;
            font-size: 1em;
        }
        .stSlider>div>div>div>div {
            background: linear-gradient(to right, #2980b9, #6dd5fa);
        }
        .stSidebar .css-ng1t4o {
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 10px;
        }
    </style>'''
, unsafe_allow_html=True)

# Titles
st.markdown('<div class="main-title">TATA IPL 2025 Score Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict the final score of an IPL innings based on live stats!</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar Inputs
st.sidebar.header("📊 Match Input Details")

with st.sidebar:
    batting_team = st.selectbox("🏏 Batting Team", options=valid_teams)
    bowling_team = st.selectbox("🏋️ Bowling Team", options=[team for team in valid_teams if team != batting_team])
    runs = st.slider("🏃 Runs Scored", 0, 300, 80)
    wickets = st.slider("🛌 Wickets Lost", 0, 9, 2)
    overs = st.slider("⏳ Overs Bowled", 0.1, 20.0, 10.0, step=0.1)
    runs_last5 = st.slider("🔢 Runs in Last 5 Overs", 0, 100, 40)
    wickets_last5 = st.slider("🪨 Wickets in Last 5 Overs", 0, 5, 1)
    predict_btn = st.button("Predict Score ✨")

# Prediction logic
if predict_btn:
    input_data = {
        'runs': runs,
        'wickets': wickets,
        'overs': overs,
        'runs_last5': runs_last5,
        'wickets_last5': wickets_last5,
        'batting_team': batting_team,
        'bowling_team': bowling_team
    }

    features = encode_inputs(input_data)
    prediction = model.predict(features)

    lower_bound = int(prediction[0])
    upper_bound = lower_bound + 6

    st.success(f"🌟 Predicted Final Score: {lower_bound} - {upper_bound} runs")

    # Optional display
    st.markdown("### 😃 Match Situation")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🏃 Runs", runs)
        st.metric("🛌 Wickets", wickets)

    with col2:
        st.metric("⏳ Overs", overs)
        st.metric("🔢 Runs (Last 5)", runs_last5)

    with col3:
        st.metric("🏏 Batting", batting_team)
        st.metric("🤾‍♀️ Bowling", bowling_team)

    st.markdown("---")
    st.caption("Cricket Score Predictor 📊 | Powered by Machine Learning | IPL 2025")
    st.caption(":copyright: copyRights reserved By Zigbe !!")
