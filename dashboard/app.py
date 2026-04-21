import streamlit as st
import pandas as pd
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="LLM Data Platform",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 LLM Data Platform")
st.markdown("**RAG Pipeline + AI SQL Agent powered by Claude**")
st.divider()


# ── Check API Key ──
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    st.error("❌ ANTHROPIC_API_KEY not found! Add it to your .env file.")
    st.stop()


# ── Initialize components ──
@st.cache_resource
def initialize_rag():
    from ingestion.document_loader import DocumentLoader
    from embeddings.embedder import TextEmbedder
    from embeddings.vector_store import VectorStore
    from retrieval.rag_pipeline import RAGPipeline

    with st.spinner("📚 Loading documents..."):
        loader = DocumentLoader()
        docs   = loader.load_all()

    with st.spinner("🔢 Generating embeddings..."):
        embedder = TextEmbedder()
        store    = VectorStore()

        # Only embed if store is empty
        if store.count() == 0:
            texts      = [doc["content"] for doc in docs]
            embeddings = embedder.embed_batch(texts)
            store.add_documents(docs, embeddings)

    rag = RAGPipeline(embedder, store)
    return rag, store


@st.cache_resource
def initialize_sql_agent():
    from agents.sql_agent import SQLAgent
    return SQLAgent()


# ── Tabs ──
tab1, tab2, tab3 = st.tabs([
    "💬 RAG Chat",
    "🔍 SQL Agent",
    "📊 Vector Store Explorer"
])


# ─────────────────────────────────────────────
# TAB 1: RAG CHAT
# ─────────────────────────────────────────────
with tab1:
    st.subheader("💬 Ask Questions About Your Documents")
    st.caption("Powered by ChromaDB vector search + Claude RAG")

    # Initialize RAG
    try:
        rag, store = initialize_rag()
        st.success(f"✅ Vector store ready with {store.count()} document chunks")
    except Exception as e:
        st.error(f"Error initializing RAG: {e}")
        st.stop()

    # Chat history
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []

    # Display chat history
    for msg in st.session_state.rag_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📄 Sources"):
                    for src in msg["sources"]:
                        st.caption(f"• {src}")

    # Sample questions
    st.markdown("**Try asking:**")
    col1, col2, col3 = st.columns(3)
    sample_questions = [
        "What is the Sharpe ratio?",
        "Explain momentum investing",
        "What is a data warehouse?",
        "How is machine learning used in finance?",
        "What are the risk factors for companies?",
        "What is algorithmic trading?"
    ]

    for i, col in enumerate([col1, col2, col3]):
        with col:
            if st.button(sample_questions[i], use_container_width=True):
                st.session_state.rag_input = sample_questions[i]
            if st.button(sample_questions[i+3], use_container_width=True):
                st.session_state.rag_input = sample_questions[i+3]

    # Chat input
    query = st.chat_input("Ask anything about the documents...")

    # Handle sample question clicks
    if "rag_input" in st.session_state and st.session_state.rag_input:
        query = st.session_state.rag_input
        st.session_state.rag_input = None

    if query:
        # Add user message
        st.session_state.rag_messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents and generating answer..."):
                result = rag.ask(query)

            st.write(result["answer"])

            with st.expander("📄 Sources used"):
                for doc in result["context"]:
                    st.caption(f"• {doc['id']} (similarity: {1-doc['distance']:.3f})")

        # Save assistant message
        st.session_state.rag_messages.append({
            "role":    "assistant",
            "content": result["answer"],
            "sources": result["sources"]
        })

        st.rerun()


# ─────────────────────────────────────────────
# TAB 2: SQL AGENT
# ─────────────────────────────────────────────
with tab2:
    st.subheader("🔍 Natural Language SQL Agent")
    st.caption("Ask questions in plain English — Claude writes and runs the SQL")

    try:
        agent = initialize_sql_agent()
        st.success("✅ SQL Agent ready | 50 companies, 6 years of financials")
    except Exception as e:
        st.error(f"Error initializing SQL agent: {e}")
        st.stop()

    # Sample questions
    st.markdown("**Try asking:**")
    sql_samples = [
        "Show top 5 companies by revenue",
        "Which sectors have the highest average growth?",
        "Show companies founded after 2000 with revenue above 10000",
        "What is the average revenue by sector?",
        "Show companies with negative growth",
        "Which company has the most employees?"
    ]

    cols = st.columns(3)
    for i, col in enumerate(cols):
        with col:
            if st.button(sql_samples[i], key=f"sql_{i}", use_container_width=True):
                st.session_state.sql_query = sql_samples[i]
            if st.button(sql_samples[i+3], key=f"sql_{i+3}", use_container_width=True):
                st.session_state.sql_query = sql_samples[i+3]

    # SQL query history
    if "sql_messages" not in st.session_state:
        st.session_state.sql_messages = []

    question = st.chat_input("Ask a question about the company data...")

    if "sql_query" in st.session_state and st.session_state.sql_query:
        question = st.session_state.sql_query
        st.session_state.sql_query = None

    if question:
        with st.spinner("🤔 Generating SQL and running query..."):
            result = agent.ask(question)

        st.session_state.sql_messages.append(result)

    # Display results
    for result in reversed(st.session_state.sql_messages):
        st.markdown(f"**❓ {result['question']}**")

        with st.expander("🔧 Generated SQL"):
            st.code(result["sql"], language="sql")

        if not result["results"].empty and "error" not in result["results"].columns:
            st.dataframe(result["results"], hide_index=True, use_container_width=True)
            st.caption(f"💬 {result['explanation']}")
        else:
            st.error("Query failed — try rephrasing your question")

        st.divider()


# ─────────────────────────────────────────────
# TAB 3: VECTOR STORE EXPLORER
# ─────────────────────────────────────────────
with tab3:
    st.subheader("📊 Vector Store Explorer")
    st.caption("Explore what's in the knowledge base")

    try:
        rag, store = initialize_rag()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("📄 Total Documents", store.count())
        with col2:
            st.metric("🔢 Embedding Dimension", "384")

        st.divider()

        # Search the vector store
        st.markdown("**Semantic Search**")
        search_query = st.text_input(
            "Search for similar documents:",
            placeholder="e.g. machine learning finance"
        )

        n_results = st.slider("Number of results", 1, 10, 5)

        if search_query:
            from embeddings.embedder import TextEmbedder
            embedder = TextEmbedder()
            query_emb = embedder.embed_text(search_query)
            results   = store.search(query_emb, n_results=n_results)

            st.markdown(f"**Top {len(results)} results for: '{search_query}'**")
            for i, doc in enumerate(results):
                similarity = 1 - doc["distance"]
                with st.expander(
                    f"#{i+1} | {doc['id']} | Similarity: {similarity:.3f}"
                ):
                    st.write(doc["content"][:500] + "...")
                    st.json(doc["metadata"])

    except Exception as e:
        st.error(f"Error: {e}")


# ── Sidebar ──
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **LLM Data Platform**

    Built with:
    - 🤖 Claude (Anthropic)
    - 🗄️ ChromaDB (Vector DB)
    - 🔢 Sentence Transformers
    - 🐍 Python
    - 📊 Streamlit

    **Architecture:**
    1. Load documents
    2. Chunk + embed text
    3. Store in ChromaDB
    4. Retrieve by similarity
    5. Generate with Claude

    **[View on GitHub](https://github.com/saimanjunathk/llm-data-platform)**
    """)

    if st.button("🗑️ Clear Chat History"):
        st.session_state.rag_messages = []
        st.session_state.sql_messages = []
        st.rerun()