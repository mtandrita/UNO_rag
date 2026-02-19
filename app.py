import streamlit as st
import os
from typing import TypedDict, List
from langgraph.graph import START, StateGraph, END
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma

# --- 1. PAGE CONFIG & EXTERNAL CSS ---
st.set_page_config(page_title="UNO Agentic RAG", page_icon="⚡", layout="wide")

# Link the local style.css file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

if os.path.exists("style.css"):
    local_css("style.css")

# --- 2. MODELS ---
llm = ChatOllama(model="llama3.2", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# --- 3. AGENT STATE DEFINITION ---
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    iteration: int

# --- 4. GRAPH NODES (High Speed Logic) ---
def retrieve(state: GraphState):
    docs = st.session_state.vectorstore.similarity_search(state["question"], k=4)
    return {"documents": docs, "question": state["question"], "iteration": state.get("iteration", 0)}

def grade_documents(state: GraphState):
    """Batch grades all docs in one LLM call to save time"""
    full_text = "\n\n".join([d.page_content for d in state["documents"]])
    prompt = f"Analyze if these excerpts are relevant to the question: '{state['question']}'. Answer ONLY 'yes' or 'no'."
    res = llm.invoke(f"{prompt}\n\nContext:\n{full_text}")
    
    if "yes" in res.content.lower():
        return "generate"
    return "transform_query"

def transform_query(state: GraphState):
    # Rewrites question for better semantic matching
    res = llm.invoke(f"Rewrite this search query to be more precise for a PDF search: {state['question']}")
    return {"question": res.content, "iteration": state.get("iteration", 0) + 1}

def generate(state: GraphState):
    context = "\n\n".join([d.page_content for d in state["documents"]])
    prompt = (
        f"You are a professional AI Analyst. Use the following context to answer the user question.\n"
        f"If the data contains tables or lists, format them clearly using Markdown.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['question']}\n"
        f"Answer:"
    )
    res = llm.invoke(prompt)
    return {"generation": res.content}

# --- 5. BUILD LANGGRAPH WORKFLOW ---
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("transform_query", transform_query)
workflow.add_node("generate", generate)

workflow.add_edge(START, "retrieve")
workflow.add_conditional_edges(
    "retrieve", 
    grade_documents, 
    {"generate": "generate", "transform_query": "transform_query"}
)
# Retry logic: only allow 1 retry to keep it fast
workflow.add_conditional_edges(
    "transform_query", 
    lambda x: "retrieve" if x["iteration"] < 2 else "generate"
)
workflow.add_edge("generate", END)
rag_app = workflow.compile()

# --- 6. PRODUCTION UI LAYOUT ---
st.title("⚡ UNO Agentic RAG")
st.markdown("---")

with st.sidebar:
    st.header("📂 Document Center")
    uploaded_file = st.file_uploader("Upload Source PDF", type="pdf", label_visibility="collapsed")
    
    if uploaded_file:
        with st.spinner("⚡ Powering up Knowledge Base..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Layout-aware loader (Great for Tables!)
            loader = PyMuPDFLoader("temp.pdf")
            docs = loader.load()
            
            # Optimized splitting for structured data
            splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
            st.session_state.vectorstore = Chroma.from_documents(
                documents=splitter.split_documents(docs), embedding=embeddings
            )
            st.success("Indexing Complete!")
    
    st.markdown("---")
    st.write("🤖 **Model:** Llama 3.2")
    st.write("🧠 **Engine:** LangGraph Agent")

# --- 7. CHAT & EXECUTION ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Query your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        with st.spinner("Agentic Workflow in progress..."):
            final_ans = ""
            # Stream the graph to show the reasoning trace
            for output in rag_app.stream({"question": prompt}):
                for node, data in output.items():
                    status_placeholder.markdown(f"<div class='status-box'>📍 Currently in node: {node.upper()}</div>", unsafe_allow_html=True)
                    if node == "generate":
                        final_ans = data["generation"]
            
            st.write(final_ans)
            st.session_state.messages.append({"role": "assistant", "content": final_ans})