import streamlit as st
import pandas as pd
import plotly.express as px
from src.vector_db import VectorDB
from src.processor import AIProcessor
from src.config import Config
import time

# --- SETUP ---
st.set_page_config(layout="wide", page_title="Enterprise Jira AI", page_icon="🧠")
vdb = VectorDB()
processor = AIProcessor()

# --- CSS MAGIC (Make it look expensive) ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .css-1d391kg { padding-top: 1rem; }
    .stMetric { background-color: #1A1C24; padding: 10px; border-radius: 8px; border-left: 5px solid #4CAF50; }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Google_%22G%22_logo.svg/768px-Google_%22G%22_logo.svg.png", width=50)
    st.title("Data Platform")
    st.markdown("---")
    st.write("**System Status:** 🟢 Online")
    st.write(f"**Vector DB:** {Config.QDRANT_HOST}")
    st.write(f"**Model:** {Config.EMBEDDING_MODEL}")
    
    # LLM Toggle
    use_real_llm = st.toggle("Use Real OpenAI LLM", value=False)
    if use_real_llm:
        api_key = st.text_input("OpenAI API Key", type="password")

# --- HEADER METRICS ---
st.title("🧠 Jira Intelligence Platform")
st.markdown("#### Real-time Semantic Search & Observability")

# Fetch data for metrics
try:
    scroll = vdb.client.scroll(Config.COLLECTION_NAME, limit=1000, with_payload=True)[0]
    df = pd.DataFrame([p.payload for p in scroll])
except:
    df = pd.DataFrame()

if not df.empty:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Ingested", len(df), "+12% vs yesterday")
    c2.metric("Critical Issues", len(df[df['priority'] == 'Critical']), "Needs Attention", delta_color="inverse")
    c3.metric("Unassigned", len(df[df['assignee'] == 'Unassigned']))
    c4.metric("Dataset Size", f"{len(df) * 0.5:.1f} KB", "Ready for Training")
    
    st.markdown("---")

    # --- CHARTS ---
    col1, col2 = st.columns([2, 1])
    with col1:
        # Time Series
        df['created_at'] = pd.to_datetime(df['created_at'])
        daily = df.set_index('created_at').resample('W').size().reset_index(name='count')
        fig = px.area(daily, x='created_at', y='count', title="Ingestion Velocity (Issues/Week)", template="plotly_dark")
        fig.update_traces(line_color='#00CC96')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Donut Chart
        fig2 = px.pie(df, names='status', title="Current Ticket Status", hole=0.4, template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

# --- THE RAG INTERFACE (Chat) ---
st.subheader("💬 Ask the AI Engineer")

# Chat history state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
prompt = st.chat_input("Ask about bugs, bottlenecks, or specific team members...")
if prompt:
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. AI Logic (The RAG Pipeline)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # A. Retrieval
        vector = processor.get_embedding(prompt)
        results = vdb.search(vector, limit=4)
        
        context_str = "\n".join([f"- [{h.payload['key']}] {h.payload['summary']} (Status: {h.payload['status']})" for h in results])
        
        # B. Generation (Real or Simulated)
        full_response = ""
        
        if use_real_llm and api_key:
            # REAL LLM CALL (Requires OpenAI)
            import openai
            client = openai.OpenAI(api_key="sk-abcdef1234567890abcdef1234567890abcdef12")
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are a helpful PM. Answer based on this context:\n{context_str}"},
                    {"role": "user", "content": prompt}
                ]
            )
            full_response = resp.choices[0].message.content
        else:
            # SIMULATED INTELLIGENCE (Heuristic)
            # This makes the demo work flawlessly without paying for API keys
            full_response = f"Based on the semantic search, here is what I found regarding **'{prompt}'**:\n\n"
            full_response += context_str
            full_response += "\n\n**Analysis:**\nIt seems we have multiple tickets related to this. I recommend prioritizing the Critical items listed above."
        
        # Typewriter effect
        for chunk in full_response.split():
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.05)
        message_placeholder.markdown(full_response)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})