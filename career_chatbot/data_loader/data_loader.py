from langchain_community.document_loaders import PyPDFium2Loader, WebBaseLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from pydantic import BaseModel
from career_chatbot.data_loader.rag_config import Config

class RAGIngestionPipeline:
    def __init__(self):
        """
        Hnadle the ingestion of PDF, websites , or local text files into a Chroma vector database.
        """
        self.config = Config()
        self.rag_config = self.config.to_pydantic()
        self.data_path = self.rag_config.DATA_PATH
        self.chroma_path = self.rag_config.CHROMA_PATH
        self.collection_name = self.rag_config.COLLECTION_NAME
        self.embedding_model = self.rag_config.EMBEDDING_MODEL
        self.chunk_size = self.rag_config.CHUNK_SIZE
        self.chunk_overlap = self.rag_config.CHUNK_OVERLAP

        # Initialize embeddings and splitter
        self.embedding = OpenAIEmbeddings(model=self.embedding_model)
        self.text_spliter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
        )


        


