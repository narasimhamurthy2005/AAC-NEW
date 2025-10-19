from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# Folder where scraped files are saved
folder = "scraped_pages"

# Load all text files from the folder
texts = []
for file in os.listdir(folder):
    if file.endswith('.txt'):
        with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
            texts.append(f.read())

if len(texts) == 0:
    print("No text files found! Check folder path.")
else:
    print(f"Found {len(texts)} text files.")

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = []
for text in texts:
    chunks.extend(text_splitter.split_text(text))

print(f"Total chunks: {len(chunks)}")

# Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

# Save embeddings and chunks
np.save('embeddings.npy', embeddings)
with open('chunks.npy', 'wb') as f:
    np.save(f, chunks)

print("Embeddings and chunks saved.")
