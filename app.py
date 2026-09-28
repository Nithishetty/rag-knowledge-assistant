import uuid
from pathlib import Path

import streamlit as st

from ingest import ingest_documents
from rag_chain import ask_question


PROJECT_ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
CHROMA_ROOT_DIR = PROJECT_ROOT / "chroma_databases"


st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="📚",
    layout="wide",
)


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

if "document_count" not in st.session_state:
    st.session_state.document_count = 0

if "active_chroma_dir" not in st.session_state:
    st.session_state.active_chroma_dir = None


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("📚 RAG Knowledge Assistant")

st.caption(
    "Upload one or more PDF documents and ask questions "
    "about their content."
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:

        st.write(
            f"**{len(uploaded_files)} document(s) selected**"
        )

        for uploaded_file in uploaded_files:
            st.caption(f"📄 {uploaded_file.name}")

        if st.button(
            "Process Documents",
            type="primary",
            use_container_width=True,
        ):

            try:

                DOCUMENTS_DIR.mkdir(exist_ok=True)
                CHROMA_ROOT_DIR.mkdir(exist_ok=True)

                # Remove previous PDF files.
                for existing_file in DOCUMENTS_DIR.glob("*.pdf"):
                    existing_file.unlink()

                # Save newly uploaded PDFs.
                for uploaded_file in uploaded_files:

                    file_path = (
                        DOCUMENTS_DIR
                        / uploaded_file.name
                    )

                    with open(file_path, "wb") as file:
                        file.write(
                            uploaded_file.getbuffer()
                        )

                # Create a NEW ChromaDB directory.
                database_id = uuid.uuid4().hex

                new_chroma_dir = (
                    CHROMA_ROOT_DIR
                    / database_id
                )

                with st.spinner(
                    "Processing documents..."
                ):

                    ingest_documents(
                        chroma_db_dir=new_chroma_dir
                    )

                # Make this database the active one.
                st.session_state.active_chroma_dir = (
                    new_chroma_dir
                )

                st.session_state.documents_processed = (
                    True
                )

                st.session_state.document_count = (
                    len(uploaded_files)
                )

                # New documents = new conversation.
                st.session_state.messages = []

                st.success(
                    f"Successfully processed "
                    f"{len(uploaded_files)} document(s)."
                )

                st.rerun()

            except Exception as error:

                st.error(
                    f"Error while processing documents: "
                    f"{error}"
                )


    # --------------------------------------------------
    # Knowledge base information
    # --------------------------------------------------

    if st.session_state.documents_processed:

        st.divider()

        st.subheader("Knowledge Base")

        st.metric(
            "Documents",
            st.session_state.document_count,
        )

        st.caption(
            "The current knowledge base contains "
            "the latest uploaded documents."
        )


    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# --------------------------------------------------
# Main chat area
# --------------------------------------------------

st.subheader("💬 Ask Questions")


if not st.session_state.documents_processed:

    st.info(
        "Upload one or more PDF documents from the "
        "sidebar and click **Process Documents**."
    )


# --------------------------------------------------
# Display chat history
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                f"📚 Sources ({len(message['sources'])})"
            ):

                displayed_sources = set()

                for document in message["sources"]:

                    source = Path(
                        document.metadata.get(
                            "source",
                            "Unknown document",
                        )
                    ).name

                    page = document.metadata.get(
                        "page_label",
                        document.metadata.get(
                            "page",
                            "?",
                        ),
                    )

                    source_key = (
                        source,
                        page,
                    )

                    if source_key not in displayed_sources:

                        st.write(
                            f"📄 **{source}** — Page {page}"
                        )

                        displayed_sources.add(
                            source_key
                        )


# --------------------------------------------------
# Chat input
# --------------------------------------------------

question = st.chat_input(
    "Ask something about your documents..."
)


if question:

    if not st.session_state.documents_processed:

        st.warning(
            "Please upload and process at least "
            "one PDF first."
        )

    else:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        # Recent conversation history
        previous_messages = (
            st.session_state.messages[:-1]
        )

        history_text = ""

        for message in previous_messages[-6:]:

            history_text += (
                f"{message['role']}: "
                f"{message['content']}\n"
            )

        retrieval_question = question

        if history_text:

            retrieval_question = f"""
Conversation history:
{history_text}

Current question:
{question}
"""

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching documents..."
            ):

                try:

                    result = ask_question(
                        retrieval_question,
                        st.session_state.active_chroma_dir,
                    )

                    answer = result["answer"]
                    sources = result["sources"]

                    st.markdown(answer)

                    if sources:

                        with st.expander(
                            f"📚 Sources ({len(sources)})"
                        ):

                            displayed_sources = set()

                            for document in sources:

                                source = Path(
                                    document.metadata.get(
                                        "source",
                                        "Unknown document",
                                    )
                                ).name

                                page = document.metadata.get(
                                    "page_label",
                                    document.metadata.get(
                                        "page",
                                        "?",
                                    ),
                                )

                                source_key = (
                                    source,
                                    page,
                                )

                                if (
                                    source_key
                                    not in displayed_sources
                                ):

                                    st.write(
                                        f"📄 **{source}** "
                                        f"— Page {page}"
                                    )

                                    displayed_sources.add(
                                        source_key
                                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )

                except Exception as error:

                    st.error(
                        f"Error while answering "
                        f"the question: {error}"
                    )