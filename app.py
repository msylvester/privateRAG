"""
Main application for RAG-based chat system.
Provides a Streamlit UI for interacting with chat agents.
"""

import streamlit as st
import pandas as pd
import os
from typing import Dict, List, Any, Optional
import psycopg2  # Import psycopg2

from ingest import DocumentIngester
from retriever import DocumentRetriever
from chat_agent import ChatAgent, AgentManager

# Set up the page
st.set_page_config(
    page_title="RAG Chat System",
    page_icon="🤖",
    layout="wide"
)

# Add Scale button at the top
if st.button("Scale"):
    url = st.session_state.get("new_agent_url", "")
    if url:
        with st.spinner("Scraping job data from Greenhouse..."):
            try:
                from greenhouse_job_scraper import GreenhouseJobScraper
                scraper = GreenhouseJobScraper()
                soup = scraper.fetch_page(url)
                if soup:
                    job_details = scraper.scrape_job_details(url)
                    if job_details:
                        # Convert job details to text format
                        job_text = f"Title: {job_details.get('title', 'N/A')}\n"
                        job_text += f"Company: {job_details.get('company', 'N/A')}\n"
                        job_text += f"Location: {job_details.get('location', 'N/A')}\n"
                        job_text += f"URL: {job_details.get('url', 'N/A')}\n"
                        job_text += f"Job ID: {job_details.get('job_id', 'N/A')}\n\n"
                        job_text += f"Description:\n{job_details.get('description', 'N/A')}"
                        
                        # Create a download button for the text file
                        st.download_button(
                            label="Download Job Data",
                            data=job_text,
                            file_name="greenhouse_job_data.txt",
                            mime="text/plain"
                        )
                        st.success("Job data scraped successfully!")
                    else:
                        st.error("No job details found.")
                else:
                    st.error("Failed to fetch the Greenhouse job page.")
            except Exception as e:
                st.error(f"Error scraping job data: {str(e)}")
    else:
        st.warning("Please enter a URL in the 'Document URL' field below.")

# Initialize session state
if "agent_manager" not in st.session_state:
    st.session_state.agent_manager = AgentManager()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

if "current_agent_id" not in st.session_state:
    st.session_state.current_agent_id = None

# --- Fix input clearing: signal and control on rerun ---
if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

# Function to create a new agent
def create_new_agent(url: str, agent_name: str) -> None:
    """Create a new agent from a URL."""
    with st.spinner("Creating agent... This may take a while as we process the documentation."):
        try:
            from greenhouse_job_scraper import GreenhouseJobScraper
            scraper = GreenhouseJobScraper()
            job_details = scraper.scrape_job_details(url)
            if not job_details:
                st.error("Failed to scrape job details.")
            else:
                job_text = scraper.get_job_text(job_details)
                scraper.save_to_txt(job_details)
                agent = st.session_state.agent_manager.create_agent(url, agent_name)
                st.session_state.current_agent_id = agent.agent_id
                st.session_state.chat_history[agent.agent_id] = []
                st.success(f"Agent '{agent_name}' created successfully!")
        except Exception as e:
            st.error(f"Error creating agent: {str(e)}")

# Function to handle sending a message
def send_message(agent_id: str, message: str) -> None:
    """Send a message to an agent and get a response."""
    if not message.strip():
        return
    # Add user message to chat history
    st.session_state.chat_history[agent_id].append({"role": "user", "content": message})
    agent = st.session_state.agent_manager.get_agent(agent_id)
    with st.spinner("Thinking..."):
        response = agent.query(message, show_ranking=st.session_state.get("show_ranking", False))
        st.session_state.chat_history[agent.agent_id].append({"role": "assistant", "content": response["answer"]})

# Function to switch agents
def switch_agent(agent_id: str) -> None:
    """Switch to a different agent."""
    st.session_state.current_agent_id = agent_id
    if agent_id not in st.session_state.chat_history:
        st.session_state.chat_history[agent_id] = []

# Function to delete an agent
def delete_agent(agent_id: str) -> None:
    """Delete an agent."""
    if st.session_state.agent_manager.delete_agent(agent_id):
        if agent_id in st.session_state.chat_history:
            del st.session_state.chat_history[agent_id]
        if st.session_state.current_agent_id == agent_id:
            st.session_state.current_agent_id = None
        st.success("Agent deleted successfully!")
    else:
        st.error("Failed to delete agent.")

# --- Database connection and data fetching ---
def fetch_jobs_data():
    """Fetches data from the 'jobs' table in the PostgreSQL database."""
    try:
        # Get the database URL from the environment variable
        postgres_url = os.environ.get("POSTGRES_URL")

        if not postgres_url:
            st.error("POSTGRES_URL environment variable not set.")
            return None

        # Establish a connection to the PostgreSQL database
        conn = psycopg2.connect(postgres_url)
        cur = conn.cursor()

        # Execute a query to fetch all data from the 'jobs' table
        cur.execute("SELECT name, id FROM jobs")  # Select name and id
        data = cur.fetchall()

        # Close the cursor and connection
        cur.close()
        conn.close()

        return data

    except Exception as e:
        st.error(f"Error fetching data from the database: {e}")
        return None

def fetch_skillz_data():
    """Fetches data from the 'skillz' table in the PostgreSQL database."""
    try:
        # Get the database URL from the environment variable
        postgres_url = os.environ.get("POSTGRES_URL")

        if not postgres_url:
            st.error("POSTGRES_URL environment variable not set.")
            return None

        # Establish a connection to the PostgreSQL database
        conn = psycopg2.connect(postgres_url)
        cur = conn.cursor()

        # Execute a query to fetch all data from the 'skillz' table
        cur.execute("SELECT skill_name, company_name FROM skillz")  # Select skill_name and company_name
        data = cur.fetchall()

        # Close the cursor and connection
        cur.close()
        conn.close()

        return data

    except Exception as e:
        st.error(f"Error fetching data from the database: {e}")
        return None

def fetch_unified_data():
    """Fetches and joins data from both jobs and skillz tables."""
    try:
        # Get the database URL from the environment variable
        postgres_url = os.environ.get("POSTGRES_URL")

        if not postgres_url:
            st.error("POSTGRES_URL environment variable not set.")
            return None

        # Establish a connection to the PostgreSQL database
        conn = psycopg2.connect(postgres_url)
        cur = conn.cursor()

        # Execute a query to join the jobs and skillz tables
        # This query assumes there's a relationship between company_name in skillz 
        # and name in jobs. Adjust the JOIN condition as needed.
        query = """
        SELECT j.id, j.name as job_name, s.skill_name, s.company_name
        FROM jobs j
        LEFT JOIN skillz s ON j.name = s.company_name
        ORDER BY j.name, s.skill_name
        """
        
        cur.execute(query)
        data = cur.fetchall()

        # Close the cursor and connection
        cur.close()
        conn.close()

        # Convert to DataFrame with appropriate column names
        if data:
            df = pd.DataFrame(data, columns=["job_id", "job_name", "skill_name", "company_name"])
            return df
        return None

    except Exception as e:
        st.error(f"Error fetching unified data from the database: {e}")
        return None

# Main layout
st.title("RAG Chat System")

# Sidebar for agent management
with st.sidebar:
    st.header("Agent Management")

    # Add option to show ranking scores
    if "show_ranking" not in st.session_state:
        st.session_state["show_ranking"] = False
    show_ranking = st.checkbox("Show document ranking scores", value=st.session_state["show_ranking"])
    st.session_state["show_ranking"] = show_ranking

    # Add option to show unified view
    show_unified_view = st.checkbox("Show Jobs & Skills View", value=False)

    with st.expander("Create New Agent", expanded=True):
        new_agent_url = st.text_input("Document URL", placeholder="https://example.com/docs", key="new_agent_url")
        new_agent_name = st.text_input("Agent Name", placeholder="My Documentation Agent")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Create Agent"):
                if new_agent_url and new_agent_name:
                    create_new_agent(new_agent_url, new_agent_name)
                else:
                    st.warning("Please provide both a URL and a name for the agent.")
        with col2:
            if st.button("Add Skills"):
                st.info("Alert created")

    st.subheader("Your Agents")
    agents = st.session_state.agent_manager.list_agents()
    if not agents:
        st.info("No agents created yet. Create your first agent above!")
    else:
        for agent in agents:
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(f"{agent['agent_name']}", key=f"select_{agent['agent_id']}"):
                    switch_agent(agent['agent_id'])
            with col2:
                if st.button("🗑️", key=f"delete_{agent['agent_id']}" ):
                    delete_agent(agent['agent_id'])
    st.divider()
    st.markdown("### About")
    st.markdown("""
    This is a prototype RAG-based chat system that allows you to:
    - Create agents from documentation sources
    - Chat with agents to get information from the documentation
    - Manage multiple agents for different documentation sources
    """)

# Main chat interface
if st.session_state.current_agent_id:
    agent = st.session_state.agent_manager.get_agent(st.session_state.current_agent_id)
    st.header(f"Chat with {agent.agent_name}")
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history[agent.agent_id]:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**{agent.agent_name}:** {message['content']}")

    # --- Updated input clearing logic ---
    ui_value = "" if st.session_state.clear_input else st.session_state.get("user_input", "")
    user_input = st.text_input("Your message:", key="user_input", value=ui_value)
    send_pressed = st.button("Send")
    if send_pressed:
        send_message(agent.agent_id, user_input)
        st.session_state.clear_input = True
        st.rerun()
    # After a rerun, reset the flag
    if st.session_state.clear_input:
        st.session_state.clear_input = False
else:
    st.info("Select an agent from the sidebar or create a new one to start chatting.")

# Sample data display (for development purposes)
if st.sidebar.checkbox("Show Sample Data", value=False):
    st.subheader("Sample Data Preview")
    try:
        sample_df = pd.read_csv("kizen_cleaned.csv")
        st.dataframe(sample_df.head())
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")

# Display the unified view if the checkbox is selected
if show_unified_view:
    st.subheader("Jobs & Skills View")
    unified_data = fetch_unified_data()
    if unified_data is not None:
        # Add a filter for company/job name
        if not unified_data.empty:
            job_filter = st.selectbox(
                "Filter by Job/Company:", 
                options=["All"] + sorted(unified_data["job_name"].unique().tolist())
            )
            
            # Apply filter if not "All"
            if job_filter != "All":
                filtered_df = unified_data[unified_data["job_name"] == job_filter]
            else:
                filtered_df = unified_data
                
            # Display the filtered data
            st.dataframe(filtered_df)
            
            # Show a count of skills per job
            if st.checkbox("Show Skills Count per Job"):
                skill_counts = unified_data.groupby("job_name")["skill_name"].count().reset_index()
                skill_counts.columns = ["Job/Company", "Number of Skills"]
                st.bar_chart(skill_counts.set_index("Job/Company"))
    else:
        st.info("No unified data available or connection issue.")

# Run the Streamlit app
if __name__ == "__main__":
    # This is handled by Streamlit's execution model
    pass
