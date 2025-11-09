import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

class OpenAILLM:
    def __init__(self,user_controls_input):
        self.user_controls_input = user_controls_input
    
    def get_llm_model(self):
        try:
            groq_api_key = self.user_controls_input["OPENAI_API_KEY"]
            if groq_api_key=='' and os.environ["OPENAI_API_KEY"] =='':
                st.error("Please Enter the OpenAI AI Key")
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        except Exception as e:
            raise ValueError(f"Error Occured With Exception: {e}")
        
        return llm
        
