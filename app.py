import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Încărcăm modelul
with open("model_beton.pkl", "rb") as f:
    data = pickle.load(f)

models = data["models"]
scaler = data["scaler"]
input_cols = data["input_cols"]
output_cols = data["output_cols"]

st.set_page_config(page_title="Simulator Rețete Beton Ceramic", layout="wide")

st.title("🏗️ Simulator Predictiv: Beton cu Agregat Ceramic")
st.markdown("Introduceți cantitățile componentelor pentru a estima proprietățile betonului.")

# Sidebar pentru input-uri
st.sidebar.header("Parametri Rețetă (Kg)")

def user_input_features():
    ciment = st.sidebar.number_input("Ciment", value=1.475, step=0.01)
    a04_r = st.sidebar.number_input("Agr. 0-4 Râu", value=3.945, step=0.01)
    a04_c = st.sidebar.number_input("Agr. 0-4 Ceramic", value=0.0, step=0.01)
    a48_r = st.sidebar.number_input("Agr. 4-8 Râu", value=1.835, step=0.01)
    a48_c = st.sidebar.number_input("Agr. 4-8 Ceramic", value=0.0, step=0.01)
    a816_r = st.sidebar.number_input("Agr. 8-16 Râu", value=3.395, step=0.01)
    a816_c = st.sidebar.number_input("Agr. 8-16 Ceramic", value=0.0, step=0.01)
    apa = st.sidebar.number_input("Apă", value=0.85, step=0.01)
    
    data_dict = {
        "Ciment": ciment, "Agregat_0_4_rau": a04_r, "Agregat_0_4_ceramica": a04_c,
        "Agregat_4_8_rau": a48_r, "Agregat_4_8_ceramica": a48_c,
        "Agregat_8_16_rau": a816_r, "Agregat_8_16_ceramica": a816_c, "Apa": apa
    }
    return pd.DataFrame([data_dict])

df_input = user_input_features()

# Afișare input
st.subheader("Compoziție selectată")
st.table(df_input)

if st.button("Calculează Predicții"):
    # Procesare date
    X_scaled = scaler.transform(df_input[input_cols])
    
    # Predicții
    res = {}
    for col in output_cols:
        res[col] = models[col].predict(X_scaled)[0]
    
    # Afișare rezultate sub formă de carduri
    st.subheader("📊 Rezultate Estimate")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Densitate Proaspătă", f"{res['dens_proaspat']:.3f} g/cm³")
        st.metric("Densitate Întărită", f"{res['dens_intarit']:.3f} g/cm³")
    
    with col2:
        st.metric("Rezistență 7 zile", f"{res['res7']:.2f} kN/m³")
        st.metric("Rezistență 14 zile", f"{res['res14']:.2f} kN/m³")
        
    with col3:
        st.success(f"Rezistență 28 zile: {res['res28']:.2f} kN/m³")
        
    st.info("💡 Notă: Predicțiile se bazează pe modelul Gradient Boosting antrenat pe datele experimentale.")
