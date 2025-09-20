import streamlit as st
from app03-chain import getAnswer

st.title("Ask query")
ip_text = st.text_input("Question:")
if ip_text:
    response = getAnswer(ip_text)
    st.write(response)
    