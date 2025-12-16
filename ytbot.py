import streamlit as st
import os
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import YoutubeLoader

# -------------------------------------------------
# ENV SETUP
# -------------------------------------------------
load_dotenv()

# -------------------------------------------------
# FUNCTIONS
# -------------------------------------------------

def get_youtube_transcript(video_url: str) -> str:
    """Fetch transcript using LangChain YoutubeLoader"""
    try:
        loader = YoutubeLoader.from_youtube_url(
            video_url,
            add_video_info=False
        )
        docs = loader.load()

        if not docs:
            return ""

        transcript = "\n".join(doc.page_content for doc in docs)
        return transcript

    except Exception:
        return ""


def build_rag(transcript_text: str):
    """Build RAG pipeline safely"""

    if not transcript_text.strip():
        raise ValueError("Transcript is empty or unavailable for this video.")

    # Create document
    documents = [Document(page_content=transcript_text)]

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Transcript could not be split into chunks.")

    texts = [doc.page_content for doc in chunks if doc.page_content.strip()]

    if not texts:
        raise ValueError("No valid text chunks found for embedding.")

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector Store
    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=embeddings
    )

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4}
    )

    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2
    )

    # Prompt
    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Answer the question using ONLY the transcript context.

Transcript:
-----------
{context}

Question: {question}
""",
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------

st.set_page_config(page_title="YouTube RAG Assistant", layout="centered")

st.title("🎥 YouTube RAG Assistant")
st.write("Ask questions based on a YouTube video's transcript.")

youtube_url = st.text_input("🔗 Enter YouTube Video URL")
question = st.text_input("❓ Ask Your Question")

if st.button("Generate Answer"):

    if not youtube_url.strip() or not question.strip():
        st.warning("Please enter both a YouTube URL and a question.")
        st.stop()

    with st.spinner("Fetching transcript and generating answer... ⏳"):
        transcript = get_youtube_transcript(youtube_url)

        try:
            rag_chain = build_rag(transcript)
            answer = rag_chain.invoke(question)

            st.success("Answer Ready 🧠")
            st.text_area("📌 Result", value=answer, height=400)

        except ValueError as e:
            st.error(str(e))
