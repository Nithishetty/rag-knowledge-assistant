"""RAG chain for answering questions using documents stored in ChromaDB."""

from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"


def get_rag_chain(
    chroma_db_dir: Path = DEFAULT_CHROMA_DB_DIR,
):
    """Create the retriever and Gemini LLM."""

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        task_type="RETRIEVAL_QUERY",
    )

    vector_store = Chroma(
        persist_directory=str(chroma_db_dir),
        embedding_function=embeddings,
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
    )

    return retriever, llm


def ask_question(
    question: str,
    chroma_db_dir: Path = DEFAULT_CHROMA_DB_DIR,
) -> dict:
    """Retrieve relevant context and ask Gemini to answer."""

    retriever, llm = get_rag_chain(chroma_db_dir)

    documents = retriever.invoke(question)

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
You are a document-based knowledge assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context, say:
"I couldn't find that information in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    if isinstance(response.content, list):
        answer = "".join(
            item.get("text", "")
            for item in response.content
            if isinstance(item, dict)
        )
    else:
        answer = response.content

    return {
        "answer": answer,
        "sources": documents,
    }


if __name__ == "__main__":
    question = input(
        "Ask a question about your document: "
    )

    result = ask_question(question)

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")

    for document in result["sources"]:
        print(document.metadata)