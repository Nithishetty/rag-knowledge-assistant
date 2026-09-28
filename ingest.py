"""Ingest PDF documents into a local Chroma vector database."""

from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
DEFAULT_CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"


def ingest_documents(
    chroma_db_dir: Path = DEFAULT_CHROMA_DB_DIR,
) -> Chroma:
    """Load PDFs, create Gemini embeddings, and store them in ChromaDB."""

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in '{DOCUMENTS_DIR}'."
        )

    # Extract text from PDFs
    documents = []

    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        documents.extend(loader.load())

    print(f"Loaded {len(pdf_files)} PDF file(s).")
    print(f"Extracted {len(documents)} page(s).")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    if not chunks:
        raise ValueError(
            "No text chunks were created from the PDF. "
            "The PDF may be scanned or contain no extractable text."
        )

    # Gemini embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        task_type="RETRIEVAL_DOCUMENT",
    )

    # Create ChromaDB
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(chroma_db_dir),
    )

    print("Gemini embeddings stored successfully in ChromaDB.")

    return vector_store


if __name__ == "__main__":
    ingest_documents()