from pydantic import BaseModel
from configparser import ConfigParser
from pydantic import Field, BaseModel
from typing import Optional
import os

class RAGConfig(BaseModel):
    DATA_PATH: str = Field(..., description="Path to data directory")
    CHROMA_PATH: str = Field(..., description="Path to Chroma database")
    COLLECTION_NAME: str = Field(..., description="Chroma collection name")
    EMBEDDING_MODEL: str = Field(..., description="Embedding model name")
    CHUNK_SIZE: int = Field(gt=0, description="Text chunk size")
    CHUNK_OVERLAP: int = Field(ge=0, description="Text chunk overlap")


class Config:
    def __init__(self, config_file="./career_chatbot/data_loader/ragconfigfile.ini")->RAGConfig:
        self.config = ConfigParser()
        if not os.path.exists(config_file):
            raise FileExistsError(f"Config file not found: {config_file}")
        self.config.read(config_file)

    
    def get_data_path(self)->str:
        return self.config["DEFAULT"].get("DATA_PATH", "")
    
    def get_chroma_path(self)->str:
        return self.config["DEFAULT"].get("CHROMA_PATH", "")
    
    def get_collection_name(self)->str:
        return self.config["DEFAULT"].get("COLLECTION_NAME", "")
    
    def get_embedding_model(self)->str:
        return self.config["DEFAULT"].get("EMBEDDING_MODEL", "")
    
    def get_chunk_size(self)->int:
        return self.config["DEFAULT"].get("CHUNK_SIZE", "")
    
    def get_chunk_overlap(self)->int:
        return self.config["DEFAULT"].get("CHUNK_OVERLAP", "")
    
    def to_pydantic(self) -> RAGConfig:
        """Covert INI config values to a validated Pydantic model."""
        
        return RAGConfig(
            DATA_PATH=self.get_data_path(),
            CHROMA_PATH=self.get_chroma_path(),
            COLLECTION_NAME=self.get_collection_name(),
            EMBEDDING_MODEL=self.get_embedding_model(),
            CHUNK_SIZE=self.get_chunk_size(),
            CHUNK_OVERLAP=self.get_chunk_overlap(),
        )
