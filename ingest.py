"""
Data ingestion module for RAG-based chat system.
Handles CSV parsing, text chunking, and embedding generation.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import uuid

class DocumentIngester:
    def __init__(self, embedding_model=None, chunk_size=1000, chunk_overlap=200):
        """
        Initialize the document ingester with configurable chunking parameters.
        
        Args:
            embedding_model: The embedding model to use (defaults to OpenAI)
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        os.environ.pop('HTTP_PROXY', None)
        os.environ.pop('HTTPS_PROXY', None)
        self.chunk_overlap = chunk_overlap
        # Initialize OpenAIEmbeddings without proxies parameter
        self.embedding_model = embedding_model or OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame containing the CSV data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path, quoting=1, engine='python', on_bad_lines='skip')
            return df
        except Exception as e:
            raise Exception(f"Error loading CSV: {str(e)}")
    def preprocess_dataframe(self, df: pd.DataFrame, candidate_text_columns: List[str] = ["text", "markdown", "content"],
                            url_column: Optional[str] = None,
                            title_column: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Preprocess the dataframe into a list of document dictionaries.
        """
        documents = []
    
        for idx, row in df.iterrows():
            doc_text = None
            # Try candidate columns in order: "text", "markdown", then "content"
            for col in candidate_text_columns:
                if col in row.index and not pd.isna(row[col]) and str(row[col]).strip() != "":
                    doc_text = str(row[col])
                    break
            # Fallback: if no candidate produced valid text, use the first column
            if not doc_text:
                first_col = df.columns[0]
                if not pd.isna(row[first_col]) and str(row[first_col]).strip() != "":
                    doc_text = str(row[first_col])
                else:
                    continue
    
            metadata = {"source_idx": idx}
            if url_column and url_column in row.index and not pd.isna(row[url_column]):
                metadata["url"] = row[url_column]
            if title_column and title_column in row.index and not pd.isna(row[title_column]):
                metadata["title"] = row[title_column]
            documents.append({
                "text": str(doc_text),
                "metadata": metadata
            })
        return documents

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split documents into chunks for better retrieval.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of chunked document dictionaries
        """
        chunked_documents = []
        for doc in documents:
            text = str(doc["text"])
            text_chunks = self.text_splitter.split_text(text)
            for chunk in text_chunks:
                # Create a copy of the metadata to avoid modifying the original
                chunk_metadata = doc["metadata"].copy()
                # Add a unique chunk ID
                chunk_metadata["chunk_id"] = str(uuid.uuid4())
                chunked_documents.append({
                    "text": chunk,
                    "metadata": chunk_metadata
                })
    
        return chunked_documents
    
    def create_vector_store(self, chunked_documents: List[Dict[str, Any]], 
                           collection_name: str) -> Chroma:
        """
        Create a vector store from chunked documents.
        
        Args:
            chunked_documents: List of chunked document dictionaries
            collection_name: Name for the vector store collection
            
        Returns:
            Chroma vector store containing the document embeddings
        """
        texts = [doc["text"] for doc in chunked_documents]
        metadatas = [doc["metadata"] for doc in chunked_documents]
        
        # Create a persistent Chroma vector store
        persist_directory = f"./data/chroma/{collection_name}"
        os.makedirs(persist_directory, exist_ok=True)
    
        vector_store = Chroma.from_texts(
            texts=texts,
            embedding=self.embedding_model,
            metadatas=metadatas,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
     
        # Persist the vector store to disk
        vector_store.persist()
        
        return vector_store
    
    def process_csv(self, csv_path: str, text_column: str, 
                   url_column: Optional[str] = None,
                   title_column: Optional[str] = None,
                   collection_name: Optional[str] = None) -> Chroma:
        """
        Process a CSV file end-to-end: load, preprocess, chunk, and create vector store.
        
        Args:
            csv_path: Path to the CSV file
            text_column: Name of the column containing the text content
            url_column: Optional name of the column containing URLs
            title_column: Optional name of the column containing titles
            collection_name: Name for the vector store collection (defaults to CSV filename)
            
        Returns:
            Chroma vector store containing the document embeddings
        """
        # Default collection name to the CSV filename without extension
      
        if collection_name is None:
            collection_name = os.path.splitext(os.path.basename(csv_path))[0]
   
        # Load and process the CSV
        df = self.load_csv(csv_path)
      
        documents = self.preprocess_dataframe(
            df,
            candidate_text_columns=["text", "markdown", "content"],
            url_column=url_column,
            title_column=title_column
        )
    
        chunked_documents = self.chunk_documents(documents)
    
        vector_store = self.create_vector_store(chunked_documents, collection_name)
       
        
        return vector_store

if __name__ == "__main__":
    ingester = DocumentIngester()
    
    vector_store = ingester.process_csv(
        csv_path="kizen_ex.csv",
        text_column="text",
        url_column="url",
        title_column="metadata/title"
    )
    
    print(f"Vector store created with {vector_store._collection.count()} documents")
