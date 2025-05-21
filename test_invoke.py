from chat_agent import AgentManager, ChatAgent # Ensure ChatAgent is imported if type hinting agent
import sys

def test_agent_direct_chain_call():
    # Initialize the agent manager to load existing agents
    agent_manager = AgentManager()

    # List available agents
    agents_list = agent_manager.list_agents() # Renamed to avoid conflict
    if not agents_list:
        print("No agents found. Please create an agent first using app.py")
        return

    # Print available agents
    print("Available agents:")
    for i, agent_info in enumerate(agents_list):
        print(f"{i+1}. {agent_info['agent_name']} (ID: {agent_info['agent_id']})")

    # Get agent selection from user
    try:
        selection_idx = int(input("\nSelect an agent number to test: ")) - 1
        if not (0 <= selection_idx < len(agents_list)):
            print("Invalid selection")
            return
    except ValueError:
        print("Please enter a valid number")
        return

    # Get the selected agent
    agent_id_to_load = agents_list[selection_idx]["agent_id"]
    agent: ChatAgent = agent_manager.get_agent(agent_id_to_load)

    if not agent:
        print(f"Could not load agent with ID: {agent_id_to_load}")
        return

    print(f"\nSelected Agent: {agent.agent_name} (ID: {agent.agent_id})")
    if not agent.chain:
        print("Agent's chain is not initialized. Cannot proceed.")
        return

    print(f"Agent's chain input keys (expected by invoke): {agent.chain.input_keys}")
    print(f"Agent's chain memory keys: {agent.chain.memory.memory_key}")
    print(f"Agent's prompt input variables: {agent.chain.prompt.input_variables}")

    # Get test question from user
    question = input("\nEnter a test question: ")

    # Perform the query logic locally, calling the agent's chain directly
    print("\nInvoking agent's chain directly...")
    try:
        # 1. Get context
        context = agent._get_context(question)

        # 2. Prepare inputs for the chain.
        # LLMChain with memory expects input keys NOT part of memory.
        # 'chat_history' is in memory_keys, so it should be excluded from inputs to invoke.
        inputs_for_chain = {
        "input": {
            "agent_name": agent.agent_name,
            "context": context,
            "question": question
        }
    }


        print(f"Inputs provided to agent.chain.invoke: {inputs_for_chain}")
        # Add this after loading the agent but before invoking the chain
        print("\nDiagnosing chain input requirements:")
        print(f"Chain input keys: {agent.chain.input_keys}")
        if hasattr(agent.chain, 'input_schema'):
            print(f"Chain input schema: {agent.chain.input_schema.model_json_schema()}")
        if hasattr(agent.chain.prompt, 'input_variables'):
            print(f"Prompt input variables: {agent.chain.prompt.input_variables}")
        if hasattr(agent.chain, 'memory') and agent.chain.memory:
            print(f"Memory keys: {agent.chain.memory.memory_key}")
            print(f"Memory output key: {getattr(agent.chain.memory, 'return_messages', False)}")


        # 3. Invoke the chain
        response_payload = agent.chain.invoke(inputs_for_chain)

        # Assuming response_payload is a dictionary with a "text" key for the answer
        answer = response_payload.get("text", "Error: 'text' key not found in chain response.")

        # The memory should be updated automatically by the LLMChain if configured correctly.

        # 4. Format and print the response (similar to how agent.query would)
        response = {
            "agent_id": agent.agent_id,
            "agent_name": agent.agent_name,
            "question": question,
            "answer": answer,
            "has_context": bool(context and context != "No context available." and context != "No relevant information found.")
        }

    except Exception as e:
        print(f"Exception occurred during direct chain invocation: {e}")
        import traceback
        traceback.print_exc()
        response = {
            "agent_id": agent.agent_id if agent else "N/A",
            "agent_name": agent.agent_name if agent else "N/A",
            "question": question,
            "answer": f"Error: {e}",
            "has_context": False # Or determine based on context if available
        }

    # Print the response
    print(f"\nQuestion: {response['question']}")
    print(f"Answer: {response['answer']}")
    print(f"Has context: {response['has_context']}")

def main():
    test_agent_direct_chain_call()

if __name__ == "__main__":
    main()
