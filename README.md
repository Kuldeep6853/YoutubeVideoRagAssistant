##🎥 YouTube RAG Assistant
---

A Streamlit-based Retrieval-Augmented Generation (RAG) application that allows users to ask questions about a YouTube video and get AI-generated answers based only on the video transcript.

The app fetches subtitles from YouTube, embeds the transcript using Hugging Face models, stores them in FAISS, and uses Google Gemini to generate accurate answers.


🚀 Features
---

📜 Fetches YouTube video transcripts automatically

🔍 Semantic search using FAISS vector database

🧠 AI-powered answers using Google Gemini

🧩 Text chunking for long transcripts

⚡ Fast and lightweight Streamlit UI

❌ Prevents hallucinations by answering only from transcript content


🧱 Tech Stack
---

- Python 3.11

- Streamlit

- LangChain

- Google Gemini API

- Hugging Face Embeddings

- FAISS

YouTube Transcript Loader

## 🔐 API Key Disclaimer (MANDATORY)

This project requires a **Google Gemini API key**.

- You must create your own API key from **Google AI Studio**
- **Do NOT commit your API key to GitHub**
- The API key is loaded securely using environment variables

⚠️ Any misuse or exposure of API keys is the responsibility of the user.

---

## 🛠️ Setup Instructions

 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2️⃣ Create Virtual Environment
```bash
python -m venv venv
```
 Activate Virtual Environment (Windows)
 ```bash
venv\Scripts\activate
```

3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

4️⃣ Configure Environment Variables

Create a .env file in the root directory and add the following:
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_hugging_face_api_key_here
```

▶️ Run the Application
```bash
streamlit run file_name.py
```




