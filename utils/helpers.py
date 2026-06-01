import streamlit as st
from groq import Groq
from tavily import TavilyClient

def get_groq_client():
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        return Groq(api_key=api_key)
    except Exception as e:
        st.error("Failed to initialize Groq client. Please check your GROQ_API_KEY in .streamlit/secrets.toml.")
        raise e

def get_tavily_client():
    try:
        api_key = st.secrets["TAVILY_API_KEY"]
        return TavilyClient(api_key=api_key)
    except Exception as e:
        st.error("Failed to initialize Tavily client. Please check your TAVILY_API_KEY in .streamlit/secrets.toml.")
        raise e
