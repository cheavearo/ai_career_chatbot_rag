import streamlit as st
from career_chatbot.ui.streamlit.loadui import LoadStreamlitUI
from career_chatbot.LLMS.openaillm import OpenAILLM
from career_chatbot.graphs.graph_builder import GraphBuilder
from career_chatbot.ui.streamlit.display_result import DisplayResultStreamlit

def laod_langgraph_app():
    """
    Load and runs the Langgraph AgenticAI with Streamlit UI.
    This function initializes the UI, handle user input, configure the LLM model,
    sets up the graph based on the selected use case, and displays the output while
    implementing exception handling for robustness.
    """

    # Load UI and get user input
    ui = LoadStreamlitUI()
    user_input = ui.load_streamlit_ui()

    if not user_input or not user_input.get("OPENAI_API_KEY"):
      #st.error("Error: Failed to laod user from the UI.")
      return st.stop()

    # Get user message (main chat input)
    user_message = st.chat_input("Enter your messages:")
    
    if not user_message:
      st.warning("Please type a question to get started!")
      # Stop processing further
      return
    usecase = "rag_chatbot"
    #if not user_input or not user_input.get("OPENAI_API_KEY"):
          #st.stop()

  
    try:
      # Configure the LLM's
      obj_llm_config = OpenAILLM(user_controls_input=user_input)
      model = obj_llm_config.get_llm_model()
      if not model:
            st.error("Error: LLM model could not be initialized")
            return
      # Build graph
      graph_builder = GraphBuilder(model)
      graph = graph_builder.setup_graph(usecase)
            
            
      # Display results
      DisplayResultStreamlit(usecase, graph, user_message).display_result_on_ui()

    except Exception as e:
      st.error(f"Error: Graph set up failed- {e}")
      return