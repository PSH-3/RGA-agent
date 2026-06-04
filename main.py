import os
import re
from typing import List

from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama


load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
CHUNK_COUNT = int(os.getenv("CHUNK_COUNT", 4))

EMBEDDING_TYPE = os.getenv("EMBEDDING_TYPE", "OLLAMA")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
LLM_URL = os.getenv("LLM_URL", "http://localhost:11434")


SYSTEM_PROMPT = """
You are a helpful AI assistant that answers questions based on the provided context.

Rules:
1. Only use information from the provided context
2. If not enough information — say it clearly
3. Be specific
4. Keep answers concise

Context:
{context}

Question:
{input}
"""

# PDF PROCESSING


def extract_text_from_pdf(path: str) -> str:
    if not path or not os.path.exists(path):
        raise ValueError("Invalid PDF_PATH")

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "".join(pages)


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    return text.strip()


def split_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


# VECTOR STORE


def build_vectorstore(chunks: List[str]) -> FAISS:
    if EMBEDDING_TYPE != "OLLAMA":
        raise NotImplementedError("Only OLLAMA embedding supported")

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    docs = [
        Document(
            page_content=chunk,
            metadata={"chunk_id": i}
        )
        for i, chunk in enumerate(chunks)
    ]

    return FAISS.from_documents(docs, embeddings)


# RAG CHAIN


def build_rag_chain(vectorstore: FAISS):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": CHUNK_COUNT}
    )

    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=LLM_URL,
        temperature=0.0
    )

    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def ask(question: str, chain, retriever):
    print(f"\nQ: {question}")

    answer = chain.invoke(question)
    print(f"A: {answer}")

    docs = retriever.invoke(question)
    print(f"\nSources ({len(docs)}):")

    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc.page_content[:150]}...")

    return answer


def main():
    text = extract_text_from_pdf(PDF_PATH)
    text = clean_text(text)

    chunks = split_text(text)
    vectorstore = build_vectorstore(chunks)

    chain, retriever = build_rag_chain(vectorstore)

    questions = [
        "Какая основная тема документа?",
        "Какая основная концепция документа??",
        "Кратко изложи суть документа."
    ]

    for q in questions:
        ask(q, chain, retriever)


if __name__ == "__main__":
    main()