import streamlit as st
import uuid
from dotenv import load_dotenv

from flipkart.data_ingestion import DataIngestor
from flipkart.rag_agent import RAGAgentBuilder

load_dotenv()

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="AI Shopping Assistant",
    page_icon="🛒",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Custom CSS — professional retail look
# ---------------------------
st.markdown("""
<style>
/* Font & base */
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

/* Header banner */
.header-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.header-banner h1 {
    color: #ffffff;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.3px;
}
.header-banner p {
    color: #94a3b8;
    margin: 4px 0 0 0;
    font-size: 0.9rem;
}
.header-badge {
    background: #e94560;
    color: white;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 20px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    display: inline-block;
    margin-top: 6px;
}

/* Suggested questions */
.suggest-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}

/* Source pill */
.source-pill {
    display: inline-block;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.75rem;
    color: #475569;
    margin: 2px 3px 2px 0;
    font-weight: 500;
}

/* Metric cards */
.metric-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px 18px;
    text-align: center;
}
.metric-card .value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #0f3460;
}
.metric-card .label {
    font-size: 0.75rem;
    color: #64748b;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}

/* Sidebar polish */
[data-testid="stSidebar"] {
    background: #f8fafc;
}
.sidebar-section {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 12px;
}
.sidebar-section h4 {
    font-size: 0.8rem;
    font-weight: 700;
    color: #374151;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin: 0 0 8px 0;
}
.sidebar-section p {
    font-size: 0.82rem;
    color: #6b7280;
    margin: 3px 0;
}

/* Thinking indicator */
.thinking-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #0f3460;
    margin: 0 2px;
    animation: bounce 1.2s infinite ease-in-out;
}
.thinking-dot:nth-child(2) { animation-delay: 0.2s; }
.thinking-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
    40% { transform: scale(1.0); opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 8px 0 16px 0;">
        <div style="font-size:2.2rem;">🛒</div>
        <div style="font-weight:700; font-size:1rem; color:#0f3460;">AI Shopping Assistant</div>
        <div style="font-size:0.75rem; color:#94a3b8; margin-top:2px;">Powered by RAG + LLM</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="sidebar-section"><h4>🧠 Tech Stack</h4><p>🔗 LangChain v1 + LangGraph</p><p>🗄️ AstraDB Vector Store</p><p>⚡ Groq LLM Inference</p><p>🔍 MultiQuery Retriever</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section"><h4>📦 Data</h4><p>📊 Product reviews dataset</p><p>⭐ Ratings & summaries indexed</p><p>🔄 Semantic search enabled</p></div>', unsafe_allow_html=True)

    st.divider()

    if st.button("🔄 New Conversation", use_container_width=True, type="primary"):
        st.session_state.clear()
        st.rerun()

# ---------------------------
# Session State
# ---------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.request_count = 0
    st.session_state.prediction_count = 0

# ---------------------------
# Load Agent (Cached)
# ---------------------------
@st.cache_resource(show_spinner="🔍 Loading product knowledge base...")
def load_agent():
    vector_store = DataIngestor().ingest(load_existing=True)
    builder = RAGAgentBuilder(vector_store)
    agent = builder.build_agent()
    return agent, builder.query_capture  # agent + shared capture list

rag_agent, query_capture = load_agent()

# ---------------------------
# Header Banner
# ---------------------------
st.markdown("""
<div class="header-banner">
    <div style="font-size:2.4rem;">🛒</div>
    <div>
        <h1>AI Shopping Assistant</h1>
        <p>Ask about products, compare specs, read real customer reviews</p>
        <span class="header-badge">AI Powered</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Suggested Questions (shown only when no conversation yet)
# ---------------------------
SUGGESTED_QUESTIONS = [
    "🎧 Best Bluetooth headphones under 2000?",
    "📱 Which products have the best customer reviews for sound quality ?",
    "⭐ What are the highest rated earbuds?",
    "🔋 Which headphones have the best battery life?",
    "💻 Suggest headsets with good sound quality",
]

st.markdown('<div class="suggest-label">Try asking...</div>', unsafe_allow_html=True)
cols = st.columns(2)
for i, q in enumerate(SUGGESTED_QUESTIONS):
    col = cols[i % 2]
    if col.button(q, key=f"suggest_{i}", use_container_width=True):
        st.session_state["pending_input"] = q

# ---------------------------
# Display Chat History
# ---------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            sources_html = "".join(
                f'<span class="source-pill">📦 {s}</span>' for s in msg["sources"]
            )
            st.markdown(
                f'<div style="margin-top:10px;"><span style="font-size:0.72rem;color:#94a3b8;font-weight:600;text-transform:uppercase;letter-spacing:0.4px;">Sources</span><br>{sources_html}</div>',
                unsafe_allow_html=True,
            )

# ---------------------------
# User Input
# ---------------------------
user_input = st.chat_input("Ask me about any product...") or st.session_state.pop("pending_input", None)

if user_input:
    st.session_state.request_count += 1

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ---------------------------
    # Assistant Response with streaming
    # ---------------------------
    with st.chat_message("assistant"):

        # Animated thinking indicator while first token arrives
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown(
            '<span class="thinking-dot"></span><span class="thinking-dot"></span><span class="thinking-dot"></span>',
            unsafe_allow_html=True,
        )

        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        state = {"first_token_received": False}
        query_capture.clear()  # reset before each new query

        def token_stream():
            for token, metadata in rag_agent.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
                stream_mode="messages",
            ):
                if metadata.get("langgraph_node") == "model":
                    content = token.content
                    if not state["first_token_received"] and content:
                        thinking_placeholder.empty()
                        state["first_token_received"] = True
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                yield block["text"]
                            elif isinstance(block, str):
                                yield block
                    elif isinstance(content, str) and content:
                        yield content

        reply = st.write_stream(token_stream())
        thinking_placeholder.empty()
        st.session_state.prediction_count += 1

        if not reply:
            reply = "I couldn't find relevant product information. Please try rephrasing your question."
            st.markdown(reply)

        # Extract source product names from "Sources: ..." line in reply
        sources = []
        if isinstance(reply, str) and "Sources:" in reply:
            for line in reply.split("\n"):
                if line.strip().startswith("Sources:"):
                    raw = line.replace("Sources:", "").strip().strip("`")
                    sources = [s.strip() for s in raw.split(",") if s.strip()]
                    break

        if sources:
            sources_html = "".join(
                f'<span class="source-pill">📦 {s}</span>' for s in sources
            )
            st.markdown(
                f'<div style="margin-top:10px;"><span style="font-size:0.72rem;color:#94a3b8;font-weight:600;text-transform:uppercase;letter-spacing:0.4px;">Sources</span><br>{sources_html}</div>',
                unsafe_allow_html=True,
            )

        # Query expansion visualiser
        if query_capture:
            with st.expander("🔍 Query Expansion — see how your question was interpreted", expanded=False):
                st.markdown("**Original query:**")
                st.code(user_input, language=None)
                st.markdown("**Expanded queries sent to vector search:**")
                for i, q in enumerate(query_capture, 1):
                    st.markdown(
                        f'<div style="padding:6px 12px;margin:4px 0;background:#f1f5f9;border-left:3px solid #0f3460;border-radius:4px;font-size:0.85rem;color:#1e293b;">'
                        f'<span style="font-weight:600;color:#0f3460;">#{i}</span> {q}</div>',
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "sources": sources,
    })

# ---------------------------
# Footer Metrics
# ---------------------------
st.divider()
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f'<div class="metric-card"><div class="value">{st.session_state.request_count}</div><div class="label">Queries</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="value">{st.session_state.prediction_count}</div><div class="label">Responses</div></div>', unsafe_allow_html=True)
with c3:
    turn = len([m for m in st.session_state.messages if m["role"] == "user"])
    st.markdown(f'<div class="metric-card"><div class="value">{turn}</div><div class="label">Turns</div></div>', unsafe_allow_html=True)
