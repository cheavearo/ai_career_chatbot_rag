from langgraph.graph import StateGraph, START, END
from career_chatbot.state.state import State
from career_chatbot.LLMS.openaillm import OpenAILLM
from career_chatbot.data_loader.data_loader import RAGIngestionPipeline
from career_chatbot.nodes.rag_chatbot_node import RAGChatbotNode


class GraphBuilder:
    def __init__(self, model):
        self.llm = model
        self.graph_builder = StateGraph(State)

    def rag_chatbot_build_graph(self):
        """
        Builds an advanced RAG (Retrieval-Augmented Generation) chatbot using LangGraph.
        This method initializes a chatbot node using 'RAGChatBotNode' class
        and integrates it into the graph. The chatbot node is set as both the entry and
        exit point of the graph
        """

        self.rag_chatbot_node_obj =RAGChatbotNode(self.llm)

        # Add Nodes
        self.graph_builder.add_node("web_search",self.rag_chatbot_node_obj.web_search)
        self.graph_builder.add_node("retrieve", self.rag_chatbot_node_obj.retrieve)
        self.graph_builder.add_node("grade_documents", self.rag_chatbot_node_obj.grade_documents)
        self.graph_builder.add_node("generate", self.rag_chatbot_node_obj.generate)
        self.graph_builder.add_node("transform_query", self.rag_chatbot_node_obj.transform_query)

        # Edges
        self.graph_builder.add_conditional_edges(
            START,
            self.rag_chatbot_node_obj.route_question,
            {
                "web_search": "web_search",
                "vector_store": "retrieve"
            },
        )
        self.graph_builder.add_edge("web_search", "generate")
        self.graph_builder.add_edge("retrieve", "grade_documents")
        self.graph_builder.add_conditional_edges(
            "grade_documents",
            self.rag_chatbot_node_obj.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        self.graph_builder.add_edge("transform_query", "retrieve")
        self.graph_builder.add_conditional_edges(
            "generate",
            self.rag_chatbot_node_obj.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "transform_query",
            },
        )
        return self.graph_builder

    
    # Compile the graph
    def setup_graph(self, usecase ="rag_chatbot"):
        """Sets up the graph."""
        if usecase == "rag_chatbot":
            self.rag_chatbot_build_graph()
        return self.graph_builder.compile()




from dotenv import load_dotenv
load_dotenv()


#os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['LANGSMITH_API_KEY'] = os.getenv("LANGSMITH_API_KEY")
import os



  

    

## Below code is for the LangGraph studio
llm = OpenAILLM(
    user_controls_input={
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")
    }
).get_llm_model()

## get the graph
graph_builder=GraphBuilder(llm)
graph=graph_builder.rag_chatbot_build_graph().compile()
    


