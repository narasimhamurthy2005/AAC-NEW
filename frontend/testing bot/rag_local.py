import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All

# -----------------------------
# 1. Load the embedding model
# -----------------------------
# Load the SentenceTransformer model once globally when the server starts
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# 2. Load saved chunks + embeddings
# -----------------------------
try:
    # Assuming these files are in the same directory as this script
    chunks = np.load("chunks.npy", allow_pickle=True)
    embeddings = np.load("embeddings.npy")
except FileNotFoundError:
    print("FATAL ERROR: chunks.npy or embeddings.npy not found.")
    chunks = []
    embeddings = np.array([])
    faiss_ready = False
else:
    # -----------------------------
    # 3. Build FAISS index
    # -----------------------------
    if embeddings.size > 0:
        dimension = embeddings.shape[1]  # embedding dimension
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        print(f"✅ FAISS index loaded with {len(chunks)} chunks")
        faiss_ready = True
    else:
        print("Warning: Embeddings file was empty.")
        faiss_ready = False


# -----------------------------
# 4. Load GPT4All model
# -----------------------------
# Load GPT4All model once globally
model_filename = "gpt4all-falcon-newbpe-q4_0.gguf"
model_path = os.path.join(os.getcwd(), "models")

try:
    llm = GPT4All(model_filename, model_path=model_path)
    print(f"✅ GPT4All Model loaded: {model_filename}")
except Exception as e:
    print(f"FATAL ERROR: Could not load GPT4All model. Ensure model file is in the 'models' folder. Error: {e}")
    llm = None


# -----------------------------
# 5. RAG Query Function
# -----------------------------
def rag_query(query, top_k=3, distance_threshold=1.0):
    """
    Perform a RAG query and return the generated answer.
    """
    if not faiss_ready or not llm:
        return "🤖 Initialization error: RAG components are not ready."
        
    # Encode query
    query_vector = embedder.encode([query])

    # Search FAISS index
    distances, indices = index.search(query_vector, top_k)

    # If closest match is too far, return fallback
    if distances[0][0] > distance_threshold:
        return "🤖 I'm sorry, I don't have relevant information on that topic in my knowledge base."

    # Retrieve top matching chunks and truncate for faster generation
    # Ensure indices are valid before accessing chunks
    valid_indices = [i for i in indices[0] if i < len(chunks)]
    retrieved_chunks = [chunks[i][:500] for i in valid_indices]  # first 500 chars
    context = "\n".join(retrieved_chunks)

    # Build prompt
    prompt = f"""You are a helpful assistant.
Use the following context to answer the question briefly and clearly:

Context:
{context}

Question: {query}
Answer:"""

    # Get response from GPT4All
    response = ""
    # Use a chat session for isolated query handling
    with llm.chat_session():
        response = llm.generate(prompt, max_tokens=150)
    
    return response.strip()
