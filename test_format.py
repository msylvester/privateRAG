from chat_agent import AgentManager, ChatAgent # Ensure ChatAgent is imported if type hinting agent
import sys

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def get_test_chain():
    # Simple prompt template with variables matching your schema keys
    template = """
Agent name: {agent_name}
Chat history: {chat_history}
Context: {context}
Question: {question}

Answer:
"""
    prompt = PromptTemplate(
        input_variables=["agent_name", "chat_history", "context", "question"],
        template=template
    )

    # Initialize an OpenAI LLM (make sure OPENAI_API_KEY is set in your env)
    llm = OpenAI(temperature=0)

    # Create the LLMChain with the prompt and LLM
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

def test_invoke(chain):

    composed_input = {
        "agent_name": "SupportBot",
        "chat_history": "User: Hello\nAgent: Hi!",
        "context": "User needs help resetting password",
        "question": "How do I reset my password?"
    }

    try:
        result = chain.invoke(composed_input)
    except Exception as e:
        pass

def main():
    chain = get_test_chain()
    test_invoke(chain)

if __name__ == "__main__":
    main()
