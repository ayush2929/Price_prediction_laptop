import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ✅ Always first
st.set_page_config(page_title="Laptop Price Predictor", layout="wide")

# --- Load Styling & HTML ---
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

with open("header.html") as f:
    st.markdown(f.read(), unsafe_allow_html=True)

# --- Load Dropdown Data ---
data = pd.read_csv("traineddata.csv")

# --- Input Form ---
st.markdown("<div class='form-container'>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox('💼 Brand', sorted(data['Company'].unique()))
    type_name = st.selectbox('💻 Laptop Type', sorted(data['TypeName'].unique()))
    ram = st.selectbox('🧠 RAM (GB)', sorted(data['Ram'].unique()))
    cpu = st.selectbox('🖥️ CPU', sorted(data['CPU_name'].unique()))
    hdd = st.selectbox('💽 HDD (GB)', sorted(data['HDD'].unique()))
    ssd = st.selectbox('⚡ SSD (GB)', sorted(data['SSD'].unique()))

with col2:
    os = st.selectbox('🧭 Operating System', sorted(data['OpSys'].unique()))
    gpu = st.selectbox('🎮 GPU Brand', sorted(data['Gpu brand'].unique()))
    weight = st.number_input('⚖️ Weight (kg)', min_value=0.8, max_value=5.0, step=0.1)
    touchscreen = st.selectbox('🖱️ Touchscreen', ['No', 'Yes'])
    ips = st.selectbox('🌈 IPS Display', ['No', 'Yes'])
    screen_size = st.number_input('📏 Screen Size (inches)', min_value=10.0, max_value=20.0, step=0.1)

resolution = st.selectbox('🖼️ Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160',
    '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
])
st.markdown("</div>", unsafe_allow_html=True)

# --- Predict Button ---
if st.button("💡 Predict Laptop Price"):
    try:
        # Convert categorical values
        touchscreen = 1 if touchscreen == 'Yes' else 0
        ips = 1 if ips == 'Yes' else 0
        X_res, Y_res = map(int, resolution.split('x'))
        ppi = ((X_res**2 + Y_res**2) ** 0.5) / screen_size

        # Prepare input
        input_df = pd.DataFrame([[
            company, type_name, ram, weight, touchscreen, ips, ppi,
            cpu, hdd, ssd, gpu, os
        ]], columns=[
            'Company', 'TypeName', 'Ram', 'Weight', 'TouchScreen',
            'IPS', 'PPI', 'CPU_name', 'HDD', 'SSD', 'Gpu brand', 'OpSys'
        ])

        # Predict using Random Forest
        pipe = pickle.load(open("pipe.pkl", "rb"))
        log_price = pipe.predict(input_df)[0]
        final_price = int(np.exp(log_price))

        # Display Result
        st.markdown("<div class='output-box'>", unsafe_allow_html=True)
        st.success(f"💰 Estimated Laptop Price: ₹{final_price - 1000} - ₹{final_price + 1000}")
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")

# --- Footer HTML ---
with open("footer.html") as f:
    st.markdown(f.read(), unsafe_allow_html=True)
