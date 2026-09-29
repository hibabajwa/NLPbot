# NLPBot 🤖

An intelligent NLP Course Assistant built using Retrieval-Augmented Generation (RAG).

NLPBot answers questions using personal NLP lecture materials and retrieves relevant information from the course PDFs before generating a response.

## Features

- 📚 Uses 11 NLP lecture PDFs as the knowledge base
- 🔎 Semantic search using FAISS
- 🧠 Sentence Transformers for embeddings
- 🤖 Groq LLM for answer generation
- 💬 Conversational chat history
- 📖 Displays relevant lecture sources
- 🌐 Streamlit web interface
- 🚫 Avoids answering questions outside the provided lecture material

## Tech Stack

- Python
- Streamlit
- FAISS
- Sentence Transformers
- LangChain Text Splitters
- Groq API
- PyPDF

## RAG Pipeline

1. Load NLP lecture PDFs
2. Extract text from the documents
3. Split text into smaller chunks
4. Generate embeddings using `all-MiniLM-L6-v2`
5. Store embeddings in a FAISS vector database
6. Retrieve relevant chunks for a user query
7. Pass retrieved context to the LLM
8. Generate a context-based answer

## Project Structure

```text
NLPBot/
├── data/
│   └── NLP lecture PDFs
├── vectorstore/
│   ├── index.faiss
│   └── chunks.pkl
├── app.py
├── ingest.py
├── rag.py
├── prompt.py
├── requirements.txt
└── .gitignore
## Deployment

The application is deployed using Streamlit Community Cloud.

### Live Application

https://hibabajwa-nlpbot-app-i32u00.streamlit.app/

### Deployment Steps

1. Upload the project to GitHub.
2. Connect the GitHub repository to Streamlit Community Cloud.
3. Set Python version to 3.12.
4. Add the `GROQ_API_KEY` as a Streamlit secret.
5. Set `app.py` as the main file.
6. Deploy the application.
