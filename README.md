# RAG-Based Knowledge Assistant

A document-based AI knowledge assistant built using **Python, LangChain, ChromaDB, Gemini, and Streamlit**.

The application allows users to upload PDF documents and ask questions about their content. It uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from the uploaded documents and generate context-based answers using Google's Gemini LLM.

---

## 🚀 Features

- 📄 Upload multiple PDF documents
- 🔍 Extract text from PDF documents
- ✂️ Split documents into smaller chunks
- 🧠 Generate embeddings using Gemini
- 🗄️ Store document embeddings in ChromaDB
- 🔎 Perform semantic similarity search
- 🤖 Generate answers using Gemini LLM
- 💬 Interactive chat interface using Streamlit
- 🧾 Display document sources and page numbers
- 🔄 Create a separate vector database for each document-processing session
- 🧹 Clear chat history
- 🔐 API key stored securely using environment variables

---

## 🏗️ Architecture

```text
                PDF Documents
                     │
                     ▼
              Text Extraction
                     │
                     ▼
               Text Chunking
                     │
                     ▼
             Gemini Embeddings
                     │
                     ▼
                 ChromaDB
                     │
                     │
User Question ───────┘
       │
       ▼
Semantic Similarity Search
       │
       ▼
Relevant Document Chunks
       │
       ▼
      Gemini LLM
       │
       ▼
Context-Based Answer
       │
       ▼
   Streamlit Chat UI
