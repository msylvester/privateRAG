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
    main()