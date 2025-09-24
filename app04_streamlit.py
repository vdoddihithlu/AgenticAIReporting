import streamlit as st

 
#from app03_chain import getAnswer
#
#st.title("Ask query")
#ip_text = st.text_input("Question:")
#if ip_text:
#    response = getAnswer(ip_text)
#    st.write(response)

    
################# run ###########################
# python -m streamlit run app04_streamlit.py
# crtl C to close    

from app06_history import getAnswer
session_id = "default"

st.title("Ask questions & followup")
if "history" not in st.session_state:
    st.session_state.history = []

# user input
ip_text = st.text_input("Question:", key=f"q_{len(st.session_state.history)}")

if ip_text:
    response = getAnswer(ip_text, session_id)

    # store result
    st.session_state.history.append((ip_text, response))

    # clear input by forcing rerun
    st.rerun()

# show history
for q, a in st.session_state.history:
    st.write(f"**Q:** {q}")
    st.write(f"**A:** {a}")