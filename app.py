import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page config
st.set_page_config(page_title="Laptop Price Predictor")

# Load CSS and Header
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
with open("header.html") as f:
    st.markdown(f.read(), unsafe_allow_html=True)

# Load data
data = pd.read_csv("traineddata.csv")

st.markdown('<div class="main-container">', unsafe_allow_html=True)

# --- Model Selection (Only Random Forest available) ---
st.markdown("### 🔍 Using Random Forest Prediction Model")

# --- Input Form Layout ---
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox('💼 Brand', sorted(data['Company'].unique()))
    type_name = st.selectbox('🧳 Laptop Type', sorted(data['TypeName'].unique()))
    ram = st.selectbox('🧠 RAM (GB)', sorted(data['Ram'].unique()))
    cpu = st.selectbox('🖥️ CPU', sorted(data['CPU_name'].unique()))
    hdd = st.selectbox('💽 HDD (GB)', sorted(data['HDD'].unique()))
    ssd = st.selectbox('⚡ SSD (GB)', sorted(data['SSD'].unique()))
    resolution = st.selectbox('🖼️ Screen Resolution', [
        '1920x1080', '1366x768', '1600x900', '3840x2160',
        '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
    ])

with col2:
    os = st.selectbox('🧭 Operating System', sorted(data['OpSys'].unique()))
    gpu = st.selectbox('🎮 GPU Brand', sorted(data['Gpu brand'].unique()))
    weight = st.number_input('⚖️ Weight (kg)', min_value=0.8, max_value=5.0, step=0.1)
    touchscreen = st.selectbox('🖱️ Touchscreen', ['No', 'Yes'])
    ips = st.selectbox('🌈 IPS Display', ['No', 'Yes'])
    screen_size = st.number_input('📏 Screen Size (inches)', min_value=10.0, max_value=20.0, step=0.1)

# --- Prediction ---
if st.button("🎯 Predict Laptop Price"):
    try:
        X_res, Y_res = map(int, resolution.split('x'))
        ppi = ((X_res**2 + Y_res**2) ** 0.5) / screen_size

        input_df = pd.DataFrame([[
            company, type_name, ram, weight, 1 if touchscreen == 'Yes' else 0,
            1 if ips == 'Yes' else 0, ppi, cpu, hdd, ssd, gpu, os
        ]], columns=[
            'Company', 'TypeName', 'Ram', 'Weight', 'TouchScreen',
            'IPS', 'PPI', 'CPU_name', 'HDD', 'SSD', 'Gpu brand', 'OpSys'
        ])

        pipe = pickle.load(open("pipe.pkl", "rb"))
        log_price = pipe.predict(input_df)[0]
        final_price = int(np.exp(log_price))

        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.success(f"💰 Estimated Price: ₹{final_price - 1000:,} – ₹{final_price + 1000:,}")
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")

# Footer
st.markdown('</div>', unsafe_allow_html=True)

with open("footer.html") as f:
    st.markdown(f.read(), unsafe_allow_html=True)
