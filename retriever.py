import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, data_dir, embedding_dir, model_name="all-MiniLM-L6-v2"):
        self.data_dir = data_dir
        self.embedding_dir = embedding_dir
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def load_data(self):
        for folder in os.listdir(self.data_dir):
            folder_path = os.path.join(self.data_dir, folder)
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.endswith(".json"):
                        file_path = os.path.join(folder_path, filename)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if isinstance(data, list):
                                    self.documents.extend(data)
                        except json.JSONDecodeError:
                            continue

    def build_index(self):
        if not self.documents:
            raise ValueError("No documents loaded. Ensure your JSON files contain valid lists.")
        embeddings = []
        for doc in tqdm(self.documents, desc="Embedding documents"):
            text = json.dumps(doc, ensure_ascii=False)
            embedding = self.model.encode(text)
            embeddings.append(embedding)

        embeddings = np.array(embeddings).astype('float32')
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

        faiss.write_index(self.index, os.path.join(self.embedding_dir, "faiss_index"))
        with open(os.path.join(self.embedding_dir, "doc_store.json"), "w", encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

    def load_index(self):
        self.index = faiss.read_index(os.path.join(self.embedding_dir, "faiss_index"))
        with open(os.path.join(self.embedding_dir, "doc_store.json"), "r", encoding='utf-8') as f:
            self.documents = json.load(f)

    def search(self, query, k=3):
        query_embedding = self.model.encode(query).astype('float32')
        D, I = self.index.search(np.array([query_embedding]), k)
        return [self.documents[i] for i in I[0]]
