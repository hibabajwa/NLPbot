import os
import pickle

import faiss
import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_FOLDER = "data"
VECTORSTORE_FOLDER = "vectorstore"


def load_pdfs():
    documents = []

    for filename in os.listdir(DATA_FOLDER):

        if filename.lower().endswith(".pdf"):

            filepath = os.path.join(DATA_FOLDER, filename)

            print(f"Loading: {filename}")

            reader = PdfReader(filepath)

            text = ""

            for page in reader.pages:
                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

            documents.append({
                "source": filename,
                "text": text
            })

    return documents


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = []

    for document in documents:

        document_chunks = splitter.split_text(
            document["text"]
        )

        for chunk in document_chunks:

            chunks.append({
                "text": chunk,
                "source": document["source"]
            })

    return chunks


def create_vectorstore(chunks):

    print("Loading embedding model...")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    os.makedirs(
        VECTORSTORE_FOLDER,
        exist_ok=True
    )

    faiss.write_index(
        index,
        f"{VECTORSTORE_FOLDER}/index.faiss"
    )

    with open(
        f"{VECTORSTORE_FOLDER}/chunks.pkl",
        "wb"
    ) as file:

        pickle.dump(
            chunks,
            file
        )

    print("\nVector database created successfully!")
    print(f"Total chunks: {len(chunks)}")


if __name__ == "__main__":

    documents = load_pdfs()

    print(
        f"\nLoaded {len(documents)} PDF files."
    )

    chunks = split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    create_vectorstore(
        chunks
    )