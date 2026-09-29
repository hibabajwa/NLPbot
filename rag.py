import os
import pickle

import faiss
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from prompt import build_prompt


# ==========================================
# Load environment variables
# ==========================================

load_dotenv()


# ==========================================
# Vector database settings
# ==========================================

VECTORSTORE_FOLDER = "vectorstore"


# ==========================================
# Load FAISS index
# ==========================================

index = faiss.read_index(
    f"{VECTORSTORE_FOLDER}/index.faiss"
)


# ==========================================
# Load document chunks
# ==========================================

with open(
    f"{VECTORSTORE_FOLDER}/chunks.pkl",
    "rb"
) as file:
    chunks = pickle.load(file)


# ==========================================
# Load embedding model
# ==========================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==========================================
# Initialize Groq client
# ==========================================

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# ==========================================
# Retrieve relevant documents
# ==========================================

def retrieve_documents(query, top_k=5):

    # Convert the user's query into an embedding
    query_embedding = embedding_model.encode(
        [query]
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    # Search the FAISS vector database
    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for distance, index_position in zip(
        distances[0],
        indices[0]
    ):

        # Skip invalid results
        if index_position == -1:
            continue

        chunk = chunks[index_position]

        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "distance": float(distance)
        })

    return results


# ==========================================
# Generate answer using RAG
# ==========================================

def generate_answer(
    question,
    chat_history=None,
    top_k=5
):

    # --------------------------------------
    # Build a context-aware search query
    # --------------------------------------

    search_query = question

    if chat_history:

        previous_user_questions = [
            message["content"]
            for message in chat_history
            if message["role"] == "user"
        ]

        if previous_user_questions:

            last_question = previous_user_questions[-1]

            search_query = (
                f"{last_question} {question}"
            )

    # --------------------------------------
    # Retrieve relevant lecture chunks
    # --------------------------------------

    retrieved_documents = retrieve_documents(
        search_query,
        top_k
    )

    # --------------------------------------
    # Build the RAG prompt
    # --------------------------------------

    prompt = build_prompt(
        question,
        retrieved_documents,
        chat_history
    )

    # --------------------------------------
    # Send prompt to Groq
    # --------------------------------------

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    # --------------------------------------
    # Extract the generated answer
    # --------------------------------------

    answer = response.choices[0].message.content

    return answer, retrieved_documents