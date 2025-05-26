"""
Chat agent module for RAG-based chat system.
Handles agent creation, loading, and query processing.
"""

import os
import json
from typing import List, Dict, Any, Optional, Union
import uuid
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

from retriever import DocumentRetriever
from ingest import DocumentIngester

class ChatAgent:
    def __init__(self, 
                agent_id: Optional[str] = None,
                agent_name: Optional[str] = None,
                collection_name: Optional[str] = None,
                llm=None,
                retriever=None,
                memory=None):
        """
        Initialize a chat agent.
        
        Args:
            agent_id: Unique identifier for the agent (generated if not provided)
            agent_name: Human-readable name for the agent
            collection_name: Name of the vector store collection to use
            llm: Language model to use for generation
            retriever: Document retriever to use
            memory: Conversation memory to use
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.agent_name = agent_name or f"Agent-{self.agent_id[:8]}"
        self.collection_name = collection_name
        
        # Set up the language model
        self.llm = llm or ChatOpenAI(temperature=0.2)
        
        # Set up the retriever if collection name is provided
        self.retriever = retriever
        if collection_name and not retriever:
            self.retriever = DocumentRetriever(collection_name=collection_name)
        
        # Set up conversation memory (disabled for single-input prompt)
        self.memory = None
        
        # Set up the prompt template (single input variable for memory compatibility)
        self._setup_prompt_template()
        
        # Set up the chain
        self._setup_chain()
    
    def _setup_prompt_template(self):
        """Set up the prompt template for the agent."""
        self.prompt_template = PromptTemplate(
            template="{input}",
            input_variables=["input"]
        )
    
    def _setup_chain(self):
        """Set up the LLM chain for the agent."""
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            verbose=True # memory is None
        )
    
    def _get_context(self, query: str, include_scores: bool = False) -> str:
        """
        Retrieve context for a query.
        
        Args:
            query: The user's query
            include_scores: Whether to include similarity scores
            
        Returns:
            String containing the retrieved context
        """
        if not self.retriever:
            return "No context available."
        
        retrieved_docs = self.retriever.retrieve(query, include_scores=include_scores)
        
        if not retrieved_docs:
            return "No relevant information found."
        
        # Format the retrieved documents into a context string
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            source_info = ""
            if "url" in doc["metadata"]:
                source_info = f" (Source: {doc['metadata']['url']})"
            
            score_info = ""
            if include_scores and "score" in doc:
                score_info = f" [Relevance: {doc['score']:.4f}]"
            
            context_parts.append(f"Document {i+1}{source_info}{score_info}:\n{doc['text']}\n")
        
        return "\n".join(context_parts)
    
    def query(self, question: str, show_ranking: bool = False) -> Dict[str, Any]:
        """
        Process a user query and generate a response.
        
        Args:
            question: The user's question
            show_ranking: Whether to show document ranking scores
            
        Returns:
            Dictionary containing the response and metadata
        """
        # Get context for the question
        context = self._get_context(question, include_scores=show_ranking)

        prompt_string = (
            f"You are a helpful assistant named {self.agent_name} that answers questions based on the provided context.\n"
            f"\n"
            f"Context information is below:\n"
            f"{context}\n"
            f"\n"
            f"Given the context information and not prior knowledge, answer the question.\n"
            f"If you don't know the answer, just say \"I don't have enough information to answer that question.\"\n"
            f"Don't try to make up an answer.\n"
            f"\n"
            f"Question: {question}\n"
            f"\n"
            f"Answer:\n"
        )

        response = self.chain.invoke({"input": prompt_string})

        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "question": question,
            "answer": response["text"],
            "has_context": bool(context and context != "No context available." and context != "No relevant information found."),
            "context_with_scores": show_ranking
        }
    
    def save(self, directory: str = "./data/agents") -> str:
        """
        Save the agent configuration to disk.
        
        Args:
            directory: Directory to save the agent configuration
            
        Returns:
            Path to the saved configuration file
        """
        os.makedirs(directory, exist_ok=True)
        
        config = {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "collection_name": self.collection_name
        }
        
        config_path = os.path.join(directory, f"{self.agent_id}.json")
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        return config_path
    
    @classmethod
    def load(cls, agent_id: str, directory: str = "./data/agents") -> "ChatAgent":
        """
        Load an agent from a configuration file.
        
        Args:
            agent_id: ID of the agent to load
            directory: Directory containing agent configurations
            
        Returns:
            Loaded ChatAgent instance
        """
        config_path = os.path.join(directory, f"{agent_id}.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Agent configuration not found: {config_path}")
        
        with open(config_path, "r") as f:
            config = json.load(f)
        
        return cls(
            agent_id=config["agent_id"],
            agent_name=config["agent_name"],
            collection_name=config["collection_name"]
        )

class AgentManager:
    def __init__(self, agents_directory: str = "./data/agents"):
        """
        Initialize the agent manager.
        
        Args:
            agents_directory: Directory containing agent configurations
        """
        self.agents_directory = agents_directory
        os.makedirs(agents_directory, exist_ok=True)
        self.agents = {}
        self._load_agents()
    
    def _load_agents(self):
        """Load all agents from the agents directory."""
        for filename in os.listdir(self.agents_directory):
            if filename.endswith(".json"):
                agent_id = filename[:-5]  # Remove .json extension
                try:
                    self.agents[agent_id] = ChatAgent.load(agent_id, self.agents_directory)
                    pass
                except Exception as e:
                    pass
    
    def create_agent(self, url: str, agent_name: Optional[str] = None, collection_name: Optional[str] = None) -> ChatAgent:
        """
        Create a new agent for a document URL.
        
        Args:
            url: URL of the document to create an agent for
            agent_name: Optional name for the agent
            collection_name: Optional name for the vector store collection.
                             If provided, it indicates an external ingestion (e.g., scraped data)
                             has already created the embeddings.
                             
        Returns:
            Newly created ChatAgent instance
        """
        from urllib.parse import urlparse
        from ingest import DocumentIngester
        import uuid

        # Generate a collection name if not provided
        if collection_name is None:
            domain = urlparse(url).netloc
            collection_name = f"{domain.replace('.', '_')}{uuid.uuid4().hex[:8]}"
            print(f'the collection name {collection_name}')
            
            # Load and process the CSV data
            import pandas as pd
            ingester = DocumentIngester()
            try:
                # Read the CSV file
                df = pd.read_csv("greenhouse_jobs.csv")
                
                # Create a single text document from all field-value pairs
                job_text = "\n".join([f"{row['Field']}: {row['Value']}" for _, row in df.iterrows()])
                
                # Create a document for ingestion with metadata
                documents = [{
                    "text": job_text, 
                    "metadata": {
                        "source": "greenhouse_jobs.csv",
                        "url": url,
                        "title": "Job Description"
                    }
                }]
                
                # Use the ingester to chunk and embed the document
                chunked_docs = ingester.chunk_documents(documents)
                vector_store = ingester.create_vector_store(chunked_docs, collection_name=collection_name)
            except Exception as e:
                raise Exception(f"Error creating agent via CSV ingestion: {str(e)}")
        
        # Create the agent with the collection name (either provided or generated)
        agent = ChatAgent(
            agent_name=agent_name or f"Agent for {urlparse(url).netloc}",
            collection_name=collection_name
        )
        
        # Save the agent configuration
        agent.save(self.agents_directory)
        
        # Add to the loaded agents
        self.agents[agent.agent_id] = agent
        return agent
    
    def get_agent(self, agent_id: str) -> ChatAgent:
        """
        Get an agent by ID.
        
        Args:
            agent_id: ID of the agent to get
            
        Returns:
            ChatAgent instance
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent not found: {agent_id}")
        
        return self.agents[agent_id]
    
    def list_agents(self) -> List[Dict[str, str]]:
        """
        List all available agents.
        
        Returns:
            List of dictionaries containing agent information
        """
        return [
            {"agent_id": agent_id, "agent_name": agent.agent_name}
            for agent_id, agent in self.agents.items()
        ]
    
    def delete_agent(self, agent_id: str) -> bool:
        """
        Delete an agent.
        
        Args:
            agent_id: ID of the agent to delete
            
        Returns:
            True if the agent was deleted, False otherwise
        """
        if agent_id not in self.agents:
            return False
        
        # Remove the agent configuration file
        config_path = os.path.join(self.agents_directory, f"{agent_id}.json")
        if os.path.exists(config_path):
            os.remove(config_path)
        
        # Remove from loaded agents
        del self.agents[agent_id]
        
        return True

if __name__ == "__main__":
    manager = AgentManager()
    
    agent = manager.create_agent(
        url="https://example.com",
        agent_name="Example Agent"
    )
    
    response = agent.query("What is Kizen?")

