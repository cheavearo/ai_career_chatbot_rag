import streamlit as st
import os
from career_chatbot.ui.uiconfigfile import Config

# Streamlit Page Loading Class

class LoadStreamlitUI:
    def __init__(self):
        self.config = Config()
        self.user_controls = {}

    def load_streamlit_ui(self):
        st.set_page_config(page_title = self.config.get_page_title(), page_icon= "assets/logo.png", layout="wide")
        
        st.header("🤖 " + "Vearo | " + self.config.get_page_title())
        

        

        with st.sidebar:
            self.user_controls["OPENAI_API_KEY"] =st.session_state["OPENAI_API_KEY"]=st.text_input("OPENAI_API_KEY",type="password")
            if not self.user_controls["OPENAI_API_KEY"]:
                st.warning("⚠️ Please enter your Groq API key to proceed. Don't have? refer : https://console.groq.com/keys")
                return None
        
        # Set in environment so downstream code can use it
        os.environ["OPENAI_API_KEY"] = self.user_controls["OPENAI_API_KEY"]

        return self.user_controls


