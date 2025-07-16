import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load custom CSS
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load custom HTML header
with open("header.html") as f:
    st.markdown(f.read(), unsafe_allow_html=True)

# Load pipeline and data
with open('pipe.pkl', 'rb') as file1:
    pipe = pickle.load(file1)

data = pd.read_csv("traineddata.csv")

# --- Layout Start ---
st.markdown("<div class='form-container'>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    company = st.selectbox('💼 Brand', data['Company'].unique())
    type = st.selectbox('💻 Laptop Type', data['TypeName'].unique())
    ram = st.selectbox('🧠 RAM (GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
    cpu = st.selectbox('🖥️ CPU', data['CPU_name'].unique())
    hdd = st.selectbox('💽 HDD (GB)', [0, 128, 256, 512, 1024, 2048])
    ssd = st.selectbox('⚡ SSD (GB)', [0, 8, 128, 256, 512, 1024])

with col2:
    os = st.selectbox('🧭 Operating System', data['OpSys'].unique())
    gpu = st.selectbox('🎮 GPU Brand', data['Gpu brand'].unique())
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
if st.button('🎯 Predict Laptop Price'):
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    try:
        ppi = ((X_res**2 + Y_res**2) ** 0.5) / float(screen_size)
    except ZeroDivisionError:
        st.error("Screen size cannot be zero!")
        st.stop()

    query = pd.DataFrame([[
        company, type, int(ram), float(weight), touchscreen, ips,
        float(ppi), cpu, int(hdd), int(ssd), gpu, os
    ]], columns=[
        'Company', 'TypeName', 'Ram', 'Weight', 'TouchScreen',
        'IPS', 'PPI', 'CPU_name', 'HDD', 'SSD', 'Gpu brand', 'OpSys'
    ])

    log_price = pipe.predict(query)[0]
    final_price = int(np.exp(log_price))

    st.markdown("<div class='output-box'>", unsafe_allow_html=True)
    st.success(f"💰 Estimated Laptop Price: ₹{final_price - 1000} - ₹{final_price + 1000}")
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
with open("footer.html") as f:
    st.markdown(f.read(), unsafe_allow_html=True)
