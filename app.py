import streamlit as st

st.set_page_config(page_title="MusicAbility", layout="centered")
st.title("🎵 MusicAbility (Demo)")
st.write("Primero verificamos que la app corre local.")
text = st.text_input("Escribe una idea musical:")
if st.button("Mostrar"):
    st.success(f"Texto recibido: {text}")