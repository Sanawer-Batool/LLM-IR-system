from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')

model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("faiss_index.idx")  

docs_path = 'trec_covid_data/trec_covid_documents.csv'
df = pd.read_csv(docs_path)
documents = df['text'].tolist()
doc_ids = df['_id'].tolist()

def retrieve_documents(query, k=10):
    query_embedding = model.encode([query])[0]
    distances, indices = index.search(np.array([query_embedding]).astype(np.float32), k)
    results = [(doc_ids[i], 1 - d) for i, d in zip(indices[0], distances[0]) if i < len(doc_ids)]
    return results

queries_path = 'trec_covid_data/trec_covid_queries.csv'
queries_df = pd.read_csv(queries_path) if os.path.exists(queries_path) else pd.DataFrame(columns=['query_id', 'query'])

queries = queries_df['query'].tolist() if not queries_df.empty else ["Example query: What are the clinical features of Mycoplasma pneumoniae infections?"]
query_ids = queries_df['query_id'].tolist() if not queries_df.empty else [1]

if queries_df.empty:
    print("Warning: Queries are missing. Using placeholder data. Please download from https://ir.nist.gov/covidSubmit/ and save as trec_covid_queries.csv in trec_covid_data/.")
    print("Proceeding with example data for demonstration.")

# Tokenization and stop words removal
stop_words = set(stopwords.words('english'))

def tokenize_text(text):
    tokens = word_tokenize(text.lower())
    return [token for token in tokens if token.isalnum() and token not in stop_words]

# Query expansion using relevance language model
def expand_query(query, k=10, n=5):
    # Retrieve top-k documents
    retrieved_docs = retrieve_documents(query, k)
    if not retrieved_docs:
        return query
    
    # Extract text of top-k documents
    doc_texts = [documents[doc_ids.index(doc_id)] for doc_id, _ in retrieved_docs]
    
    # Compute term distribution (maximum likelihood estimation)
    all_tokens = []
    for doc_text in doc_texts:
        all_tokens.extend(tokenize_text(doc_text))
    
    term_freq = Counter(all_tokens)
    total_terms = len(all_tokens)
    term_probs = {term: freq / total_terms for term, freq in term_freq.items()}
    
    # Select top-n most probable terms for expansion
    expansion_terms = sorted(term_probs.items(), key=lambda x: x[1], reverse=True)[:n]
    expansion_terms = [term for term, _ in expansion_terms if term not in tokenize_text(query)]
    
    # Form expanded query (concatenate terms)
    expanded_query = f"{query} {' '.join(expansion_terms)}"
    return expanded_query, expansion_terms

# Evaluate and print expanded queries
for qid, query in zip(query_ids, queries):
    expanded_query, terms = expand_query(query, k=10, n=5)
    print(f"Query ID: {qid}, Original Query: {query}")
    print(f"Expanded Query: {expanded_query}")
    print(f"Expansion Terms: {terms}\n")

# Save results to a file for the report
with open('query_expansion_results.txt', 'w') as f:
    f.write("Query Expansion Results:\n")
    for qid, query in zip(query_ids, queries):
        expanded_query, terms = expand_query(query, k=10, n=5)
        f.write(f"Query ID: {qid}, Original Query: {query}\n")
        f.write(f"Expanded Query: {expanded_query}\n")
        f.write(f"Expansion Terms: {terms}\n\n")
    if queries_df.empty:
        f.write("Note: Results are based on placeholder data. Actual evaluation requires real queries.\n")

print("Query expansion completed. Results saved to 'query_expansion_results.txt'.")