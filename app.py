# ssh -T git@github.com
# streamlit run app.py --server.port 8080
import streamlit as st


st.title("lanchain-streamlit-app")
prompt = st.chat_input("What is up?")
print(prompt)
