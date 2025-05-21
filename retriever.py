"""
Retrieval module for RAG-based chat system.
Handles semantic search and context retrieval from vector stores.
"""

from typing import List, Dict, Any, Optional, Union
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI
import os

class DocumentRetriever:
    def __init__(self, collection_name: str, 
                embedding_model=None,
                use_compression: bool = False,
                compression_llm=None,
                top_k: int = 4):
        """
        Initialize the document retriever.
        
        Args:
            collection_name: Name of the vector store collection
            embedding_model: The embedding model to use (defaults to OpenAI)
            use_compression: Whether to use contextual compression
            compression_llm: LLM to use for compression (if enabled)
            top_k: Number of documents to retrieve
        """

        self.collection_name = collection_name
    
        os.environ.pop('HTTP_PROXY', None)
        os.environ.pop('HTTPS_PROXY', None)
        # Initialize OpenAIEmbeddings without proxies parameter
        self.embedding_model = embedding_model or OpenAIEmbeddings()
        self.top_k = top_k
        self.use_compression = use_compression
        self.compression_llm = compression_llm or ChatOpenAI(temperature=0)
        
        # Load the vector store
        self._load_vector_store()
        
        # Set up the retriever
        self._setup_retriever()
    
    def _load_vector_store(self):
        """Load the Chroma vector store from disk."""
        persist_directory = f"./data/chroma/{self.collection_name}"
        
        if not os.path.exists(persist_directory):
            raise FileNotFoundError(f"Vector store not found: {persist_directory}")
        
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding_model,
            collection_name=self.collection_name
        )
        
        print(f"Loaded vector store with {self.vector_store._collection.count()} documents")
    
    def _setup_retriever(self):
        """Set up the retriever with optional compression."""
        base_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )
        
        if self.use_compression:
            # Use LLM to extract only the relevant parts of retrieved documents
            compressor = LLMChainExtractor.from_llm(self.compression_llm)
            self.retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
        else:
            self.retriever = base_retriever
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The search query

        Returns:
            List of retrieved documents with text and metadata
        """
        # invoke() is now the recommended method
   
        docs = self.retriever.invoke(query)

        # Convert to a more usable format
        results = []
        for doc in docs:
            results.append({
                "text": doc.page_content,
                "metadata": doc.metadata
            })

        return results

    # def retrieve(self, query: str) -> List[Dict[str, Any]]:
    #     """
    #     Retrieve relevant documents for a query.
        
    #     Args:
    #         query: The search query
            
    #     Returns:
    #         List of retrieved documents with text and metadata
    #     """
    #     docs = self.retriever.get_relevant_documents(query)
        
    #     # Convert to a more usable format
    #     results = []
    #     for doc in docs:
    #         results.append({
    #             "text": doc.page_content,
    #             "metadata": doc.metadata
    #         })
        
    #     return results
    
    def retrieve_with_scores(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents with similarity scores.
        
        Args:
            query: The search query
            
        Returns:
            List of retrieved documents with text, metadata, and similarity scores
        """
        # This bypasses the retriever to get scores directly from the vector store
        docs_and_scores = self.vector_store.similarity_search_with_score(
            query=query,
            k=self.top_k
        )
        
        results = []
        for doc, score in docs_and_scores:
            results.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            })
        
        return results

# Example usage
if __name__ == "__main__":
    # This is just for testing - the actual API will be used through the main app
    retriever = DocumentRetriever(collection_name="kizen_ex")
    
    # Test retrieval
    results = retriever.retrieve("What is Kizen?")
    
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"Text: {result['text'][:100]}...")
        print(f"Metadata: {result['metadata']}")
        print("-" * 50)
