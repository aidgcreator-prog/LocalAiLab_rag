# === FULLY OPTIMIZED STREAMLIT APP (GEMMA 26B + OPTIONAL FAST MODEL) ===
# Focus: Performance + Stability + Streaming

import streamlit as st
import asyncio
from typing import Any

# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# MODEL LOADERS
# =========================
@st.cache_resource
def get_fast_llm():
    """Lightweight model (optional but strongly recommended)"""
    from langchain_ollama import OllamaLLM
    try:
        return OllamaLLM(model="gemma4:26b", temperature=0.3)
    except:
        return None  # fallback if not installed


@st.cache_resource
def get_heavy_llm():
    """Your existing Gemma 26B model"""
    from langchain_ollama import OllamaLLM
    return OllamaLLM(
        model="gemma4:26b",
        temperature=0.2,
        num_predict=256  # LIMIT TOKENS FOR SPEED
    )

# =========================
# FAST PATH LOGIC
# =========================
def is_simple_query(query: str) -> bool:
    q = query.lower()

    if len(q) < 50:
        return True

    keywords = [
        "what is", "define", "explain", "list",
        "difference", "summary"
    ]

    return any(k in q for k in keywords)

# =========================
# STREAMING HANDLER
# =========================
def stream_llm_response(llm, prompt: str):
    placeholder = st.empty()
    full_text = ""

    try:
        for chunk in llm.stream(prompt):
            full_text += chunk
            placeholder.markdown(full_text)
    except Exception as e:
        placeholder.error(f"Error: {e}")

    return full_text

# =========================
# MEMORY MANAGEMENT
# =========================
def trim_history(max_messages=10):
    if len(st.session_state.messages) > max_messages:
        st.session_state.messages = st.session_state.messages[-max_messages:]

# =========================
# UI
# =========================
st.set_page_config(page_title="Fast DeepAgents", layout="wide")

st.title("⚡ Optimized DeepAgents (Gemma 26B)")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
max_mem = st.sidebar.slider("Max Memory", 5, 20, 10)

use_fast_model = st.sidebar.checkbox("Use Fast Model (if available)", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Strategy")
st.sidebar.markdown("- Fast model → simple queries")
st.sidebar.markdown("- Gemma 26B → complex queries")

# =========================
# DISPLAY CHAT
# =========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# INPUT
# =========================
user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        # Decide model
        use_fast = is_simple_query(user_input) and use_fast_model

        fast_llm = get_fast_llm() if use_fast else None

        if fast_llm:
            llm = fast_llm
            st.caption("⚡ Fast model")
        else:
            llm = get_heavy_llm()
            st.caption("🧠 Gemma 26B")

        # Run streaming response
        response = stream_llm_response(llm, user_input)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

    trim_history(max_mem)

# =========================
# FOOTER
# =========================
st.sidebar.markdown("---")
st.sidebar.markdown("### Performance Tips")
st.sidebar.markdown("- Install gemma4:26b for speed")
st.sidebar.markdown("- Keep memory low (<10 messages)")
st.sidebar.markdown("- Avoid long prompts")
