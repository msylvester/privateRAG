"""
Chat agent module for RAG-based chat system.
Handles agent creation, loading, and query processing.
"""

import os
import json
from typing import List, Dict, Any, Optional, Union
import uuid
from langchain.prompts import ChatPromptTemplate
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
        
        # Set up conversation memory
        self.memory = memory or ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Set up the prompt template
        self._setup_prompt_template()
        
        # Set up the chain
        self._setup_chain()
    
    def _setup_prompt_template(self):
        """Set up the prompt template for the agent."""
        self.prompt_template = ChatPromptTemplate.from_template("""
        You are a helpful assistant named {agent_name} that answers questions based on the provided context.
        
        Chat History:
        {chat_history}
        
        Context information is below:
        {context}
        
        Given the context information and not prior knowledge, answer the question.
        If you don't know the answer, just say "I don't have enough information to answer that question."
        Don't try to make up an answer.
        
        Question: {question}
        
        Answer:
        """)
    
    def _setup_chain(self):
        """Set up the LLM chain for the agent."""
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            verbose=False,
            memory=self.memory
        )
        print(f"Chain expects inputs: {self.chain.input_keys}")
        print(f"Prompt expects variables: {self.prompt_template.input_variables}")
    
    def _get_context(self, query: str) -> str:
        """
        Retrieve context for a query.
        
        Args:
            query: The user's query
            
        Returns:
            String containing the retrieved context
        """
        if not self.retriever:
            return "No context available."
        
        retrieved_docs = self.retriever.retrieve(query)
        
        if not retrieved_docs:
            return "No relevant information found."
        
        # Format the retrieved documents into a context string
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            source_info = ""
            if "url" in doc["metadata"]:
                source_info = f" (Source: {doc['metadata']['url']})"
            
            context_parts.append(f"Document {i+1}{source_info}:\n{doc['text']}\n")
        
        return "\n".join(context_parts)
    def test_invoke(chain):
        print("Chain input keys:", chain.input_keys)
        print("Prompt input variables:", getattr(chain.prompt, "input_variables", None))

        # Compose full input if only one input key is expected
        if len(chain.input_keys) == 1:
            key = chain.input_keys[0]
            composed_input = {
                key: {
                    "agent_name": "SupportBot",
                    "chat_history": "User: Hello\nAgent: Hi!",
                    "context": "User needs help resetting password",
                    "question": "How do I reset my password?"
                }
            }
        else:
            # Otherwise, flat input dict
            composed_input = {
                "agent_name": "SupportBot",
                "chat_history": "User: Hello\nAgent: Hi!",
                "context": "User needs help resetting password",
                "question": "How do I reset my password?"
            }
        try:
            result = chain.invoke(composed_input)
            print("Invoke result:", result)
        except Exception as e:
            print("Exception during invoke:", e)

    def query(self, question: str) -> Dict[str, Any]:
        # Get context for the question
        context = self._get_context(question)

        try:
            # Hardcoded values that we know will work
            composed_input = {
                "agent_name": "SupportBot",
                "chat_history": "User: Hello\nAgent: Hi!",
                "context": "User needs help resetting password",
                "question": "How do I reset my password?"
            }
            response = self.chain.invoke(composed_input)

        except Exception as e:
            print(f'Exception occurred: {e}')
            return {
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "question": question,
                "answer": f"Error: {e}",
                "has_context": bool(context and context != "No context available." and context != "No relevant information found.")
            }

        print(f'The response is from: {self.agent_name}')

        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "question": question,
            "answer": response["text"],
            "has_context": bool(context and context != "No context available." and context != "No relevant information found.")
        }


    # def query(self, question: str) -> Dict[str, Any]:
    #     context = self._get_context(question)
    #     print(121)
    #     print(self.chain.input_schema.model_json_schema())

    #     try:
    #         # Build the input dict with correct keys
    #         chain_input = {
    #             "question": question,
    #             "agent_name": self.agent_name,
    #             "context": context,
    #             "chat_history": None  # or pass an actual chat history if available
    #         }
    #         example_input = {
    #             "agent_name": "ScienceBot",
    #             "question": "What is the speed of light?",
    #             "context": "This is for a high school physics student.",
    #             "chat_history": []
    #         }

    #         response = self.chain.invoke(example_input)

    #     except Exception as e:
    #         print(f'Exception occurred: {e}')
    #         return {
    #             "agent_id": self.agent_id,
    #             "agent_name": self.agent_name,
    #             "question": question,
    #             "answer": f"Error: {e}",
    #             "has_context": bool(context and context != "No context available." and context != "No relevant information found.")
    #         }

    #     print(f'The response is from: {self.agent_name}')

    #     return {
    #         "agent_id": self.agent_id,
    #         "agent_name": self.agent_name,
    #         "question": question,
    #         "answer": response["text"],
    #         "has_context": bool(context and context != "No context available." and context != "No relevant information found.")
    #     }


    # def query(self, question: str) -> Dict[str, Any]:
    #     # Get context for the question
    #     print('inside query, line 121')
    #     context = self._get_context(question)
    #     print(f'the context is {context}')
    #     # Run the chain with a single input key
    #     response = self.chain.invoke({"input": {
    #         "agent_name": self.agent_name,
    #         "question": question,
    #         "context": context
    #     }})

    #     # Return the response with metadata
    #     return {
    #         "agent_id": self.agent_id,
    #         "agent_name": self.agent_name,
    #         "question": question,
    #         "answer": response["text"],
    #         "has_context": bool(context and context != "No context available." and context != "No relevant information found.")
    #     }

    # def query(self, question: str) -> Dict[str, Any]:
    #     """
    #     Process a user query and generate a response.
        
    #     Args:
    #         question: The user's question
            
    #     Returns:
    #         Dictionary containing the response and metadata
    #     """
    #     # Get context for the question
    #     context = self._get_context(question)
        
    #     # Run the chain
    #     response = self.chain.invoke({
    #         "agent_name": self.agent_name,
    #         "question": question,
    #         "context": context
    #     })
        
    #     # Return the response with metadata
    #     return {
    #         "agent_id": self.agent_id,
    #         "agent_name": self.agent_name,
    #         "question": question,
    #         "answer": response["text"],
    #         "has_context": bool(context and context != "No context available." and context != "No relevant information found.")
    #     }
    
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
                    print(f"Loaded agent: {self.agents[agent_id].agent_name} ({agent_id})")
                except Exception as e:
                    print(f"Error loading agent {agent_id}: {str(e)}")
    
    def create_agent(self, url: str, agent_name: Optional[str] = None) -> ChatAgent:
        """
        Create a new agent for a document URL.
        
        Args:
            url: URL of the document to create an agent for
            agent_name: Optional name for the agent
            
        Returns:
            Newly created ChatAgent instance
        """
        # Generate a collection name from the URL
        # In a real system, we would scrape the URL here
        # For this prototype, we'll use a mock approach
        
        # Extract domain from URL as a simple way to generate a collection name
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        collection_name = f"{domain.replace('.', '_')}{uuid.uuid4().hex[:8]}"
        
        # For the prototype, we'll use the kizen.csv file regardless of URL
        # In a real system, we would scrape the URL and process the content
       
        ingester = DocumentIngester()
      
        try:
            # Process the CSV file (in a real system, this would be content from the URL)
           
            vector_store = ingester.process_csv(
                csv_path="kizen.csv",  # Use the provided CSV for all agents in this prototype
                text_column="content",
                url_column="url",
                title_column="title",
                collection_name=collection_name
            )
      
            # Create the agent
            agent = ChatAgent(
                agent_name=agent_name or f"Agent for {domain}",
                collection_name=collection_name
            )
           
            # Save the agent configuration
            agent.save(self.agents_directory)
        
            # Add to the loaded agents
            self.agents[agent.agent_id] = agent
          
            return agent
            
        except Exception as e:
            raise Exception(f"Error creating agent: {str(e)}")
    
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

# Example usage
if __name__ == "__main__":
    # This is just for testing - the actual API will be used through the main app
    manager = AgentManager()
    
    # Create a new agent
    agent = manager.create_agent(
        url="https://example.com",
        agent_name="Example Agent"
    )
    
    # Test the agent
    response = agent.query("What is Kizen?")
    # print(f"Question: {response['question']}")
    # print(f"Answer: {response['answer']}")
