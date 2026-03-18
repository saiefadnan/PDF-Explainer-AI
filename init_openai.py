import streamlit as st
from openai import OpenAI

# Get OpenAI API key from Streamlit secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY not found in secrets!")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)