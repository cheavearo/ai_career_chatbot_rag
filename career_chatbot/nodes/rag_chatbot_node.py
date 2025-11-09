from career_chatbot.state.state import State
from typing import Literal, List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch
from career_chatbot.data_loader.data_loader import RAGIngestionPipeline
from dotenv import load_dotenv
load_dotenv()
import os
from langsmith import Client


from langchain_core.output_parsers import StrOutputParser

os.environ['TAVILY_API_KEY']= os.getenv("TAVILY_API_KEY")

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource"""
    
    datasource: Literal["vector_store", "web_search"] = Field(
        ...,
        description="Given a user question choose to vectorstore or web search."
    )

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Docuements are relevant to the question, 'yes' or 'no'." 
    )

class GradeHallucinations(BaseModel):
    """Binary score on retrieved present in generated answer"""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )




class RAGChatbotNode:
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """
    def __init__(self, model):

        self.llm = model
        self.output_parser = StrOutputParser()

   
    
    def retrieve(self, state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )


        retrieval_grader = grade_prompt | structured_llm_grader
        # Prompt
        print("---RETRIEVE---")
        question = state["question"]
        retrieve_vector_store_obj=RAGIngestionPipeline()
        vector_store = retrieve_vector_store_obj.load_vectorstore()
        retriever=vector_store.as_retriever(search_kwargs={"k": 5})
        # Retrieve method:
        # https://docs.langchain.com/oss/python/integrations/vectorstores/chroma
        # https://reference.langchain.com/python/integrations/langchain_chroma/?_gl=1*11kaqwq*_gcl_au*NDI3ODU1MDAzLjE3NTYwMjc2MzU.*_ga*MTkwMTY3MjQ0OS4xNzU2MDI3NjM1*_ga_47WX3HKKY2*czE3NjI1MDc4MjEkbzM0JGcxJHQxNzYyNTA4NzIyJGo2JGwwJGgw
        documents = retriever.invoke(question)
        #documents = retriever.get_relevant_documents(question)
        state["documents"] = documents
        return {"documents": documents, "question": question}

    def generate(self, state)->dict:       
        
        """
        Generate answer
        
        Args:
            state (dict): The current graph state
        
        Returns:
            state (dict): New key added to state, generation, that contain LLM generation.
        """
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        #generation = state["generation"]
        # documents =  state.get("documents", [])
        if not isinstance(documents, list):
            documents = [documents]
        # Prompts
        client = Client()
        prompt = client.pull_prompt("rlm/rag-prompt") 
        # RAG chain
        rag_chain = prompt | self.llm | self.output_parser

        # RAG generation
        generation = rag_chain.invoke({"context": format_docs(documents), "question":question})
        return {"documents": documents, "question":question, "generation":generation}
    
    def grade_documents(self,state)->dict:
        """
        Determine whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state
        
        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """
        structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
        print("---CHECK DOUMENTS RELVEVANT TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        # Prompt
        system = """
        You are a grader assessing relevance of a retrieved document to a user question. \n
        If the document contains keyword(s) or sematic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
        
        """
        # Grade prompt
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \\n {document} \n\n User questions: {question}"),
            ]
        )
        # Retrieval Grader
        retrieval_grader = grade_prompt | structured_llm_grader
        for d in documents:
            score = retrieval_grader.invoke(
                {"question":question, "document":d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question":question}
    
    def transform_query(self, state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state
        
        Returns:
            state (dict): Updates question key with a re-phrased question
        """
        print("---TRANSORM QUERY---")
        question = state["question"]
        documents = state["documents"]
        # Prompt
        system ="""
        You are a question re-writer that converts an input question to a better version that is optimzed \n
        for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.
        Return only improved question, any additional comments are not allowed.
        """
        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system),
                (
                    "human",
                    "Here is the initial question: \n\n {question} \n Formulate an improved question.",
                ),
            ]
        )
        question_rewriter = re_write_prompt | self.llm | self.output_parser

        # Re-write question
        better_question = question_rewriter.invoke({"question":question})

        return {"documents": documents,"question":better_question}
    
    def web_search(self, state):
        """
        Web search based on the re-phrased question.

        Args:
            state (dict): The current graph state
        
        Returns:
            state (dict): Updates documents key with appended web results
        """

        question = state["question"]
        print("---WEB SEARCH---")
        if isinstance(question, tuple):
            question = str(question[0]) if question[0] else ""
        question = str(question).strip()
        web_search_tool = TavilySearch(
            max_results=3,
            topic="general",
            include_answer=False,
            include_raw_content=False
        )

        try:
            # Web search
            result = web_search_tool.invoke({"query": question})
            print("Raw web search result:", result)  # Debug the response
            # Check if 'results' key exists
            if "results" not in result or not result["results"]:
                print("No search results returned.")
                return {"documents": [], "question": question}
            contents = [r.get("content", "") for r in result["results"] if "content" in r]
            web_results = Document(page_content="\n".join(contents))
            return {"documents": [web_results], "question": question}
        except Exception as e:
            print("Error during web search:", e)
            return {"documents": [], "question": question}


    
    ### Edge functions

    def route_question(self, state):
        """
        route question to web search or RAG.

        Args:
            state (dict): The current graph state
        
        Returns:
            Next node to call
        """
        
        # LLM function call
        structure_llm_router = self.llm.with_structured_output(RouteQuery)
        # Prompt
        system = """
        You are an expert at routing a user question to a vectorstore or web search.
        The vectorstore contain documents related to agents, prompt engineering, and adversarial attacks.
        Use the vectorstore for question on these topics. Otherwise, use web-search.

        """
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{question}"),
            ]
        )
        question_router = route_prompt | structure_llm_router
        question = state["question"]
        print("---ROUTE QUESTION---")
        source = question_router.invoke({"question":question})
        if source.datasource == "web_search":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "web_search"
        if source.datasource == "vector_store":
            print("---ROUTE QUESTION TO RAG---")
            return "vector_store"
    
    def decide_to_generate(self, state):
        """
        Determine whether to generate an answer or re-generate a question.

        Args:
            state (dict): The current graph state
        
        Returns:
            str: Binary decision for next node to call
        """
        state["question"]
        print("---ACCESS THE GRADED DOCUMENT---")
        filtered_documents = state["documents"]

        if not filtered_documents:
            ## All documents have been filtered check_relevant
            ## We will generate a new query
            print("---DECISION: ALL DOCUMENT ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
            return "transform_query"
        else:
            print("---DECISION: GENERATE---")
            return "generate"
    
    def grade_generation_v_documents_and_question(self,state):
        """
        Determine whether to generation is grounded in the document and answer question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """
        print("---CHECK HALLUCINATION---")
        question =state["question"]
        document = state["documents"]
        generation = state["generation"]
        structured_llm_grader = self.llm.with_structured_output(GradeHallucinations)
        #Prompt
        system = """
        You are a grader assessing whether an LLM generation is grounded in /supported by a set of retrieved facts.
        Give a binary score 'yes' or 'no'. 'Yes'means that the answer is grounded in / supported by a set of facts.
        """
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),

            ]
        )
        hallucination_grader = hallucination_prompt | structured_llm_grader

        score = hallucination_grader.invoke(
            {"documents":document, "generation":generation}
        )
        grade = score.binary_score

        # Grade answer
        structured_llm_answer_grader = self.llm.with_structured_output(GradeAnswer)
        graded_answer_system ="""
        You are a grader assessing whether an answer address / resolves a question \n
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question.
        """
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", graded_answer_system),
                ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
        )
        answer_grader = answer_prompt | structured_llm_answer_grader
        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            score = answer_grader.invoke({"question": question,"generation": generation})
            grade = score.binary_score
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUEMENTS, RE-TRY---")
            return "not supported"