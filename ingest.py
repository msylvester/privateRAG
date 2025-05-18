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
import os
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
        print(30)
        # Initialize OpenAIEmbeddings without proxies parameter
        self.embedding_model = embedding_model or OpenAIEmbeddings()
       
        print(31)
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
            print(f'the file path {file_path}')
            df = pd.read_csv(file_path)
            print(f"Loaded CSV with {len(df)} rows and columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            raise Exception(f"Error loading CSV: {str(e)}")
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str, 
                             url_column: Optional[str] = None,
                             title_column: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Preprocess the dataframe into a list of document dictionaries.
        
        Args:
            df: DataFrame containing the data
            text_column: Name of the column containing the text content
            url_column: Optional name of the column containing URLs
            title_column: Optional name of the column containing titles
            
        Returns:
            List of document dictionaries with text and metadata
        """
        documents = []
        
        for idx, row in df.iterrows():
            if text_column not in row or pd.isna(row[text_column]) or row[text_column] == "":
                continue
                
            metadata = {"source_idx": idx}
            
            if url_column and url_column in row and not pd.isna(row[url_column]):
                metadata["url"] = row[url_column]
                
            if title_column and title_column in row and not pd.isna(row[title_column]):
                metadata["title"] = row[title_column]
            
            documents.append({
                "text": row[text_column],
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
            text_chunks = self.text_splitter.split_text(doc["text"])
            
            for chunk in text_chunks:
                # Create a copy of the metadata to avoid modifying the original
                chunk_metadata = doc["metadata"].copy()
                # Add a unique chunk ID
                chunk_metadata["chunk_id"] = str(uuid.uuid4())
                
                chunked_documents.append({
                    "text": chunk,
                    "metadata": chunk_metadata
                })
        
        print(f"Created {len(chunked_documents)} chunks from {len(documents)} documents")
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
        
        print(f"Created vector store with {len(texts)} documents in collection '{collection_name}'")
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
        print(177)
        if collection_name is None:
            collection_name = os.path.splitext(os.path.basename(csv_path))[0]
        print(180)
        # Load and process the CSV
        #df = self.load_csv(csv_path)
        df = self.load_csv('test_csv.csv')
        print(183)
        documents = self.preprocess_dataframe(df, text_column, url_column, title_column)
        print(185)
        chunked_documents = self.chunk_documents(documents)
        print(187)
        vector_store = self.create_vector_store(chunked_documents, collection_name)
        print(189)
        
        return vector_store

# Example usage
if __name__ == "__main__":
    # This is just for testing - the actual API will be used through the main app
    ingester = DocumentIngester()
    
    # Process a small example CSV file
    vector_store = ingester.process_csv(
        csv_path="kizen_ex.csv",
        text_column="content",
        url_column="url",
        title_column="title"
    )
    
    print(f"Vector store created with {vector_store._collection.count()} documents")
