
import streamlit as st
import pandas as pd
import pickle

with open("model_beton.pkl", "rb") as f:
    data = pickle.load(f)

st.title("🏗️ Calculator Predictiv Beton Ceramic")
st.sidebar.header("Introducere Cantități (Kg)")

user_input = {}
for col in data['input_cols']:
    user_input[col] = st.sidebar.number_input(col, value=0.0)

if st.button("Analizează Rețeta"):
    df_in = pd.DataFrame([user_input])
    X_sc = data['scaler'].transform(df_in)
    
    st.subheader("Rezultate Estimate:")
    cols = st.columns(3)
    for i, col_out in enumerate(data['output_cols']):
        val = data['models'][col_out].predict(X_sc)[0]
        cols[i%3].metric(col_out.replace('_',' ').upper(), f"{val:.2f}")
    