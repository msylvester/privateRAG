
# 🤖 RAG Chat System

## 📚 Overview

**RAG Chat System** is a Retrieval-Augmented Generation (RAG) based AI chat platform that enables users to create intelligent agents trained on custom documentation sources. These agents can then answer domain-specific questions using contextual memory and source retrieval.


![RAG Success Screenshot](images/john_kizen.png)
---

## ✨ Features

* 🧠 **Multi-Agent Support** – Create multiple chat agents for different documentation sources
* 💬 **Contextual Chat** – Agents remember past interactions for coherent conversations
* 🔍 **Source-Aware RAG Responses** – Answers are based on relevant documents and cite sources when possible
* 📂 **Agent Management** – Create, select, and delete agents as needed
* 🌐 **Interactive UI** – Streamlit-powered interface for intuitive usage

---

## 🛠️ Technical Components

* **Document Ingestion**: Parses and chunks documents into vector embeddings
* **Vector Storage**: Stores embeddings for efficient similarity-based retrieval
* **Retriever**: Finds the most relevant document chunks for any query
* **LLM Integration**: Uses OpenAI to generate responses based on retrieved context
* **Web Interface**: Streamlit app for managing and interacting with agents

---

## 🚀 Getting Started

### ✅ Prerequisites

* Python `3.8+`
* OpenAI API Key (set as an environment variable)

### 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd privateRAG

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY=your-api-key
```

### Cleaning the Data 
The project has a python script to clean the data. 
```
 python clean_text_column.py your_data.csv
```
  
Cleaned output is then run in the chat_agent (chat_agent.py) with   
```bash
csv_path="kizen_cleaned.csv",  # Use the provided CSV for all agents in this prototype
```

### ▶️ Running the App

```bash
streamlit run app.py
```

---

## 🧑‍💻 Usage Guide

1. 🆕 **Create** a new agent by entering a document URL and naming your agent
2. 🔄 **Select** an existing agent from the sidebar
3. 💬 **Chat** with the agent by asking domain-specific questions
4. 🗑️ **Delete** agents when they are no longer needed

---

## 🧪 Testing

The project includes test scripts to verify core functionality:

| Script           | Description                           |
| ---------------- | ------------------------------------- |
| `test_invoke.py` | Test direct chain invocation logic    |
 |

### Run Tests

```bash
python test_invoke.py

```

---

## 📁 Project Structure

```plaintext
rag-chat-system/
├── app.py             # Main Streamlit application
├── chat_agent.py      # Agent creation and management
├── ingest.py          # Document ingestion and embedding
├── retriever.py       # Vector store retrieval logic
├── test_invoke.py     # Direct invocation test
└── requirements.txt   # Python dependencies
```

---

## 🔮 Future Improvements

* 📱 Mobile-responsive UI
* 📄 Support for PDFs, HTML, and other document formats
* 🔌 Integration with additional LLM providers
* 📊 Usage analytics and performance metrics
* 🔐 User authentication and personalized agents

---


## 👥 Contributors

* \[Mike Sylvester]

---

