# test_invoke.py
from chat_agent import AgentManager
import sys

def test_agent_invoke():
    # Initialize the agent manager to load existing agents
    agent_manager = AgentManager()

    # List available agents
    agents = agent_manager.list_agents()
    if not agents:
        print("No agents found. Please create an agent first using app.py")
        return

    # Print available agents
    print("Available agents:")
    for i, agent in enumerate(agents):
        print(f"{i+1}. {agent['agent_name']} (ID: {agent['agent_id']})")

    # Get agent selection from user
    try:
        selection = int(input("\nSelect an agent number to test: ")) - 1
        if selection < 0 or selection >= len(agents):
            print("Invalid selection")
            return
    except ValueError:
        print("Please enter a valid number")
        return

    # Get the selected agent
    agent_id = agents[selection]["agent_id"]
    agent = agent_manager.get_agent(agent_id)

    # Get test question from user
    question = input("\nEnter a test question: ")

    # Test the agent
    print("\nSending query to agent...")
    response = agent.query(question)

    # Print the response
    print(f"\nQuestion: {response['question']}")
    print(f"Answer: {response['answer']}")
    print(f"Has context: {response['has_context']}")

def main():
    test_agent_invoke()

if __name__ == "__main__":
    main()"""
Test script for invoking LLM chains directly without using the ChatAgent query method.
This helps debug chain input format issues.
"""

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

def test_chain_invoke():
    """
    Test direct invocation of an LLM chain with various input formats
    to diagnose input format issues.
    """
    # Set up components similar to ChatAgent
    llm = ChatOpenAI(temperature=0.2)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Set up the prompt template
    prompt_template = ChatPromptTemplate.from_template("""
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
    
    # Set up the chain
    chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        verbose=True,  # Set to True for debugging
        memory=memory
    )
    
    # Print chain and prompt information
    print("Chain expects inputs:", chain.input_keys)
    print("Prompt expects variables:", prompt_template.input_variables)
    
    # Test with a flat dictionary (standard approach)
    print("\n--- Testing with flat dictionary ---")
    try:
        response = chain.invoke({
            "agent_name": "TestBot",
            "context": "The capital of France is Paris.",
            "question": "What is the capital of France?",
            "chat_history": ""
        })
        print("Success! Response:", response)
    except Exception as e:
        print("Error with flat dictionary:", e)
    
    # Test with a nested dictionary (sometimes required)
    print("\n--- Testing with nested dictionary ---")
    try:
        # If chain expects a single input key, try nesting all variables
        if len(chain.input_keys) == 1:
            key = chain.input_keys[0]
            response = chain.invoke({
                key: {
                    "agent_name": "TestBot",
                    "context": "The capital of France is Paris.",
                    "question": "What is the capital of France?",
                    "chat_history": ""
                }
            })
            print("Success with nested dictionary! Response:", response)
    except Exception as e:
        print("Error with nested dictionary:", e)
    
    # Test with minimal inputs
    print("\n--- Testing with minimal inputs ---")
    try:
        # Try with just the essential inputs
        response = chain.invoke({
            "agent_name": "TestBot",
            "context": "Test context",
            "question": "Test question?"
        })
        print("Success with minimal inputs! Response:", response)
    except Exception as e:
        print("Error with minimal inputs:", e)

if __name__ == "__main__":
    test_chain_invoke()
