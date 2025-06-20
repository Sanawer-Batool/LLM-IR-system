from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import os

# pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  

docs_path = 'trec_covid_data/trec_covid_documents.csv'
df = pd.read_csv(docs_path)
documents = df['text'].tolist()
doc_ids = df['_id'].tolist()

# Encode documents into embeddings
doc_embeddings = model.encode(documents, show_progress_bar=True)

# Build FAISS index for efficient search
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  
index.add(np.array(doc_embeddings).astype(np.float32))

print(f"Indexed {index.ntotal} documents for retrieval.")