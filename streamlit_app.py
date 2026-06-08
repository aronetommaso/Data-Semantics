import os
import sys
from pathlib import Path

import streamlit as st

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import router
from graphrag_terminal import GraphRAGAnswerer, GraphRAGConfig


st.set_page_config(
    page_title="ACLED LLM Interface",
    page_icon="",
    layout="wide",
)


def apply_runtime_config(groq_api_key: str, graphdb_repo_url: str) -> None:
    os.environ["GROQ_API_KEY"] = groq_api_key
    os.environ["GRAPHDB_REPO_URL"] = graphdb_repo_url
    router.GROQ_API_KEY = groq_api_key


@st.cache_resource(show_spinner=False)
def load_pipeline(groq_api_key: str, graphdb_repo_url: str):
    apply_runtime_config(groq_api_key, graphdb_repo_url)
    sparql_module = router.load_sparql_module()
    sparql_module.GROQ_API_KEY = groq_api_key
    sparql_module.GRAPHDB_REPO_URL = graphdb_repo_url
    graphrag_answerer = GraphRAGAnswerer(GraphRAGConfig())
    return graphrag_answerer, sparql_module


with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("Groq API Key", key="groq_api_key", type="password")
    graphdb_repo_url = st.text_input(
        "GraphDB repository URL",
        value=os.getenv("GRAPHDB_REPO_URL", "http://localhost:7200/repositories/acled-kg"),
    )
    st.divider()
    st.caption("The router automatically chooses between SPARQL Query Answering and GraphRAG.")
    st.caption("The key is used only for the current Streamlit session.")


st.title("ACLED LLM Interface")
st.caption("A Streamlit chat interface for your Groq-powered SPARQL + GraphRAG system")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi. Ask me a question about the ACLED Knowledge Graph data or the GraphRAG reports.",
        }
    ]

if "payloads" not in st.session_state:
    st.session_state.payloads = []

for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and index > 0:
            payload_index = (index - 1) // 2
            if payload_index < len(st.session_state.payloads):
                payload = st.session_state.payloads[payload_index]
                with st.expander("Response details"):
                    st.write(f"Route: `{payload.get('route', 'unknown')}`")
                    st.write(f"Status: `{payload.get('status', 'unknown')}`")
                    if payload.get("retrieval", {}).get("sparql_query"):
                        st.code(payload["retrieval"]["sparql_query"], language="sparql")
                    elif payload.get("retrieval"):
                        st.json(payload["retrieval"])


prompt = st.chat_input("Ask a question about the ACLED data...")

if prompt:
    if not groq_api_key:
        st.info("Please enter your Groq API Key in the sidebar to continue.")
        st.stop()

    apply_runtime_config(groq_api_key, graphdb_repo_url)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Querying the router, GraphDB, and GraphRAG..."):
            try:
                graphrag_answerer, sparql_module = load_pipeline(groq_api_key, graphdb_repo_url)
                payload = router.route_question(prompt, graphrag_answerer, sparql_module)
            except Exception as exc:
                payload = {
                    "route": "error",
                    "status": "error",
                    "answer": f"Execution error: {exc}",
                }

        answer = payload.get("answer", "No answer was generated.")
        st.write(answer)
        with st.expander("Response details"):
            st.write(f"Route: `{payload.get('route', 'unknown')}`")
            st.write(f"Status: `{payload.get('status', 'unknown')}`")
            if payload.get("retrieval", {}).get("sparql_query"):
                st.code(payload["retrieval"]["sparql_query"], language="sparql")
            elif payload.get("retrieval"):
                st.json(payload["retrieval"])

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.payloads.append(payload)
