# ⚡ Nexus Agentic RAG
### *Local Multi-Modal Knowledge Agent with LangGraph Self-Correction*

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Ollama](https://img.shields.io/badge/Ollama-Llama3.2-orange.svg)
![Framework](https://img.shields.io/badge/Framework-LangGraph-green.svg)

Nexus is a production-grade **Agentic Retrieval-Augmented Generation (RAG)** system designed to run entirely on local hardware. Unlike standard linear RAG pipelines, Nexus uses a **state-machine architecture** to evaluate its own search results, self-correct through query transformation, and preserve complex document layouts like tables and structured data.

---

## 🚀 Key Features

- **Agentic Self-Correction:** Uses **LangGraph** to grade retrieved context. If relevance is low, the agent autonomously rewrites the query and retries the search.
- **Layout-Aware Parsing:** Powered by **PyMuPDF**, ensuring tables and numbered lists remain structured for the LLM.
- **100% Privacy:** No data leaves your machine. All inference is handled locally via **Ollama**.
- **Interactive UI:** A high-contrast, dark-themed **Streamlit** dashboard with a real-time "Reasoning Trace" to watch the agent think.

---

## 🏗️ The Architecture

The system operates as a Directed Acyclic Graph (DAG):

1. **Retrieve:** Semantic search against **ChromaDB** using `nomic-embed-text`.
2. **Grade:** A logic node evaluates chunks for "hallucination risk" and relevance.
3. **Transform Query:** If the Grade node fails, the LLM generates a more precise search query.
4. **Generate:** The final response is synthesized only once the context quality is verified.

---

## 🛠️ Tech Stack

- **Orchestration:** LangGraph, LangChain
- **LLM Engine:** Ollama (Llama 3.2)
- **Vector Database:** ChromaDB
- **Frontend:** Streamlit
- **Document Loading:** PyMuPDF (fitz)

---

## 📥 Installation & Setup

### 1. Install Ollama & Models
Download [Ollama](https://ollama.com/) and run the following in your terminal:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text

### 2. Clone the repository:
git clone [https://github.com/YOUR_USERNAME/nexus-agentic-rag.git](https://github.com/YOUR_USERNAME/nexus-agentic-rag.git)
cd nexus-agentic-rag

###3.Install Dependencies
pip install streamlit langchain langchain-community langgraph langchain-ollama pymupdf chromadb

###4. Run the App
streamlit run app.py
```

## Project Structure: 
├── app.py           # Main Streamlit application & LangGraph logic
├── style.css        # Custom high-contrast UI styling
├── temp.pdf         # Temporary storage for uploaded documents
└── README.md        # Project documentation
