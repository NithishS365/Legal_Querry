from retriever import Retriever
from ollama_client import ask_ollama
import os
import json

# Initialize retriever
retriever = Retriever(
    data_dir="data",
    embedding_dir="embeddings",
    model_name="all-MiniLM-L6-v2"
)

# Build or load FAISS index
if not os.path.exists("embeddings/faiss_index"):
    print("Building FAISS index from JSON files...")
    retriever.load_data()
    retriever.build_index()
else:
    print("Loading existing FAISS index...")
    retriever.load_index()

# Take user query
query = input("Enter your legal query: ")

# Retrieve relevant documents
docs = retriever.search(query)

# Combine documents for context
context = "\n".join([json.dumps(doc, ensure_ascii=False) for doc in docs])

# Format prompt
prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

# Call LLaMA3 via Ollama
answer = ask_ollama(prompt)

# Output result
print("\nGenerated Answer:")
print(answer)
