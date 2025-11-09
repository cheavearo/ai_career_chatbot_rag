from typing_extensions import TypedDict, List
from langgraph.graph.message import add_messages
from typing import Annotated

class State(TypedDict):
    """ 
    Represent the structure of the state used in graph.
    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """
    #add_messages: Annotated[List, add_messages]
    question: str
    generation: str
    documents: List[str]