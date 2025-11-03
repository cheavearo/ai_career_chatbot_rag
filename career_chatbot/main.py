import streamlit as st
from career_chatbot.ui.streamlit.loadui import LoadStreamlitUI

def laod_langgraph_app():
    """
    Load and runs the Langgraph AgenticAI with Streamlit UI.
    This function initializes the UI, handle user input, configure the LLM model,
    sets up the graph based on the selected use case, and displays the output while
    implementing exception handling for robustness.
    """

    # Load UI
    ui = LoadStreamlitUI()
    user_input = ui.load_streamlit_ui()

    if not user_input:
        st.error("Error: Failed ot laod user input from the UI.")
        return
    user_input = st.chat_input("Enter your messages:")