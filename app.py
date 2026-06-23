import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(page_title="Telecom RAG Assistant", page_icon="📄", layout="wide")
st.title("📄 PDF RAG Assistant")
st.write("Ask questions about your PDF and get answers grounded in the document.")

# ---------- Helpers ----------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def build_vector_store(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(pages)

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store, pages, chunks

def build_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    system_prompt = """You are a helpful telecom assistant.
Answer the question using ONLY the context provided below.
If the context does not contain enough information, say so clearly.

Context:
{context}
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    llm = ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0,
        reasoning_format="parsed",
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# ---------- UI ----------
with st.sidebar:
    st.header("Upload PDF")
    pdf_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "chain" not in st.session_state:
        st.session_state.chain = None

    if pdf_file and st.button("Build Knowledge Base"):
        with st.spinner("Processing PDF..."):
            vector_store, pages, chunks = build_vector_store(pdf_file)
            st.session_state.vector_store = vector_store
            st.session_state.chain = build_chain(vector_store)

            st.success(f"Loaded {len(pages)} pages and built {len(chunks)} chunks.")

            with st.expander("Preview first page"):
                st.write(pages[0].page_content[:1500])

if st.session_state.chain:
    question = st.text_input("Ask a question about the PDF")

    if question:
        with st.spinner("Thinking..."):
            answer = st.session_state.chain.invoke(question)
            st.subheader("Answer")
            st.write(answer)
else:
    st.info("Upload a PDF and click **Build Knowledge Base** to start.")
