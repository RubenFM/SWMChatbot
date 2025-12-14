import streamlit as st
import langchain as lg
st.set_page_config(page_title="Aplicación Streamlit", page_icon="⚙️", layout="centered")

st.title("Normativas de natación")
prompt = st.chat_input("Say something")
if prompt:
    st.write(f"User has sent the following prompt: {prompt}")
