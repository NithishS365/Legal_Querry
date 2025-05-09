from retriever import Retriever
from ollama_client import ask_ollama
import os
import json
retriever = Retriever(
    data_dir="data",
    embedding_dir="embeddings",
    model_name="all-MiniLM-L6-v2"
)
if not os.path.exists("embeddings/faiss_index"):
    print("Building FAISS index from JSON files...")
    retriever.load_data()
    retriever.build_index()
else:
    print("Loading existing FAISS index...")
    retriever.load_index()
query = input("Enter your legal query: ")
docs = retriever.search(query)
context = "\n".join([json.dumps(doc, ensure_ascii=False) for doc in docs])
prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
answer = ask_ollama(prompt)
print("\nGenerated Answer:")
print(answer)