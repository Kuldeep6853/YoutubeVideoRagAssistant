# 📄 Multimodal RAG System (Gemini + CLIP + FAISS)

This project implements a **Multimodal Retrieval-Augmented Generation (RAG)** pipeline using:

- **Google Gemini** for reasoning and responses  
- **CLIP (Hugging Face)** for text embeddings  
- **FAISS** for vector similarity search  
- **PyMuPDF** for PDF document processing  
- **LangChain** for orchestration  

The system allows users to upload a PDF document, embed its contents, store them in a vector database, and ask natural language questions about the document.

---

## 🚀 Features

- PDF document ingestion  
- CLIP-safe text chunking (handles token limits)  
- Vector search with FAISS  
- Gemini-powered question answering  
- Fully local vector storage  
- Clean, modular, and extensible codebase  

---

## 🧱 Tech Stack

- Python 3.11  
- LangChain  
- Google Gemini API  
- Hugging Face Transformers (CLIP)  
- FAISS  
- PyMuPDF  
- Torch  

---

## 🔐 API Key Disclaimer (MANDATORY)

This project requires a **Google Gemini API key**.

- You must create your own API key from **Google AI Studio**
- **Do NOT commit your API key to GitHub**
- The API key is loaded securely using environment variables

⚠️ Any misuse or exposure of API keys is the responsibility of the user.

---

## 🛠️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

