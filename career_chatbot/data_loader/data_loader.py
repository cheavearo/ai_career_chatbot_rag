from langchain_community.document_loaders import PyMuPDFLoader, WebBaseLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
from pydantic import BaseModel
from career_chatbot.data_loader.rag_config import Config
import os

load_dotenv()

class RAGIngestionPipeline:
    def __init__(self):
        """Handle ingestion of PDFs, websites, or local text files into a Chroma vector database."""
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
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""],
        )

    # ------------------- LOADERS -------------------
    def load_pdfs(self):
        pdf_docs = []
        for file in os.listdir(self.data_path):
            if file.endswith(".pdf"):
                loader = PyMuPDFLoader(os.path.join(self.data_path, file))
                pdf_docs.extend(loader.load())
        return pdf_docs
    
    def load_texts(self):
        text_docs = []
        for file in os.listdir(self.data_path):
            if file.endswith((".txt", ".md", ".csv")):
                loader = TextLoader(os.path.join(self.data_path, file), encoding="utf-8")
                text_docs.extend(loader.load())
        return text_docs
    
    def load_website(self, urls: list[str]):
        web_docs = []
        for url in urls:
            loader = WebBaseLoader(url)
            web_docs.extend(loader.load())
        return web_docs

    # ------------------- PROCESSING -------------------
    def split_documents(self, docs):
        print(f"Splitting {len(docs)} documents into chunks...")
        chunks = self.text_splitter.split_documents(docs)
        print(f"✅ Split {len(docs)} documents into {len(chunks)} chunks")
        return chunks

    def store_in_chroma(self, documents):
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding,
            collection_name=self.collection_name,
            persist_directory=self.chroma_path,
        )
        
        print(f"✅ Stored {len(documents)} chunks into Chroma collection '{self.collection_name}'")

    def load_vectorstore(self):
        vector_store = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.chroma_path, 
            embedding_function=self.embedding,
            
        )
        print(vector_store)
        return vector_store

    
        
    def run(self, urls: list[str]=None):
        all_docs = []

        # Load from local sources
        pdfs = self.load_pdfs()
        texts = self.load_texts()
        all_docs.extend(pdfs)
        all_docs.extend(texts)

        # Optionally load from URLs
        if urls:
            web_docs = self.load_website(urls)
            all_docs.extend(web_docs)

        if not all_docs:
            print("⚠️ No documents found to process.")
            return None

        chunks = self.split_documents(all_docs)
        self.store_in_chroma(chunks)

        return {
            "pdf_count": len(pdfs),
            "text_count": len(texts),
            "web_count": len(urls) if urls else 0,
            "chunk_count": len(chunks),
            "collection_name": self.collection_name,
        }


