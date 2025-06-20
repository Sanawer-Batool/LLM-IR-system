from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

import nltk
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
qrels_path = 'trec_covid_data/trec_covid_qrels.tsv'

queries_df = pd.read_csv(queries_path) if os.path.exists(queries_path) else pd.DataFrame(columns=['query_id', 'query'])
qrels_df = pd.read_csv(qrels_path, sep='\t', names=['query_id', 'doc_id', 'relevance']) if os.path.exists(qrels_path) else pd.DataFrame(columns=['query_id', 'doc_id', 'relevance'])

queries = queries_df['query'].tolist() if not queries_df.empty else ["Example query: What are the clinical features of Mycoplasma pneumoniae infections?"]
query_ids = queries_df['query_id'].tolist() if not queries_df.empty else [1]
qrels = qrels_df if not qrels_df.empty else pd.DataFrame(columns=['query_id', 'doc_id', 'relevance'])

if queries_df.empty or qrels_df.empty:
    print("Warning: Queries and/or relevance judgments are missing. Using placeholder data. Please download from https://ir.nist.gov/covidSubmit/ and save as trec_covid_queries.csv and trec_covid_qrels.tsv in trec_covid_data/.")
    print("Proceeding with example data for demonstration. Recall values will be 0.0 due to missing relevance judgments.")

stop_words = set(stopwords.words('english'))

def tokenize_text(text):
    tokens = word_tokenize(text.lower())
    return [token for token in tokens if token.isalnum() and token not in stop_words]

def expand_query(query, k=10, n=5):
    retrieved_docs = retrieve_documents(query, k)
    if not retrieved_docs:
        return query
    
    doc_texts = [documents[doc_ids.index(doc_id)] for doc_id, _ in retrieved_docs]
    all_tokens = []
    for doc_text in doc_texts:
        all_tokens.extend(tokenize_text(doc_text))
    
    term_freq = Counter(all_tokens)
    total_terms = len(all_tokens)
    term_probs = {term: freq / total_terms for term, freq in term_freq.items()}
    expansion_terms = sorted(term_probs.items(), key=lambda x: x[1], reverse=True)[:n]
    expansion_terms = [term for term, _ in expansion_terms if term not in tokenize_text(query)]
    
    expanded_query = f"{query} {' '.join(expansion_terms)}"
    return expanded_query, expansion_terms

# Compute recall for a single query
def compute_recall(retrieved_docs, query_id, qrels):
    relevant_docs = set(qrels[qrels['query_id'] == query_id]['doc_id'].dropna())
    retrieved_doc_ids = set(doc_id for doc_id, _ in retrieved_docs)
    if not relevant_docs:
        return 0.0
    return len(retrieved_doc_ids & relevant_docs) / len(relevant_docs)

recall_before = {}
recall_after = {}

for qid, query in zip(query_ids, queries):
    # Before expansion
    retrieved_before = retrieve_documents(query, k=10)
    recall_before[qid] = compute_recall(retrieved_before, qid, qrels)
    
    # After expansion
    expanded_query, _ = expand_query(query, k=10, n=5)
    retrieved_after = retrieve_documents(expanded_query, k=10)
    recall_after[qid] = compute_recall(retrieved_after, qid, qrels)

# Create a table of recall values
recall_table = pd.DataFrame({
    'Query ID': query_ids,
    'Original Query': [q[:50] + "..." for q in queries],
    'Recall Before': [recall_before[qid] for qid in query_ids],
    'Recall After': [recall_after[qid] for qid in query_ids]
})
print("\nRecall Table:")
print(recall_table.to_string(index=False))

# Compute average recall improvements
avg_recall_before = np.mean(list(recall_before.values())) if recall_before else 0.0
avg_recall_after = np.mean(list(recall_after.values())) if recall_after else 0.0
improvement = avg_recall_after - avg_recall_before
print(f"\nAverage Recall Before: {avg_recall_before:.4f}")
print(f"Average Recall After: {avg_recall_after:.4f}")
print(f"Average Recall Improvement: {improvement:.4f}")

# Case studies (based on placeholder data, to be updated with real data)
case_studies = """
Case Studies:
1. Improved Recall Significantly:
   - Query: Example query...
   - Before: 0.0 (no relevant docs retrieved due to missing qrels)
   - After: 0.0 (expansion ineffective with placeholder data)
   - Note: With real data, expansion might improve recall by adding relevant terms (e.g., "clinical" or "infections").

2. Minimal Effect:
   - Query: Example query...
   - Before: 0.0
   - After: 0.0
   - Note: Expansion may have minimal impact if top-k docs lack diverse relevant terms.

3. Introduced Irrelevant Documents:
   - Query: Example query...
   - Before: 0.0
   - After: 0.0
   - Note: With real data, expansion might add noisy terms, reducing precision (e.g., unrelated medical terms).
"""

print("\nCase Studies:")
print(case_studies)

# Save results to a file for the report
with open('comparative_analysis_results.txt', 'w') as f:
    f.write("Recall Table:\n")
    f.write(recall_table.to_string(index=False) + "\n\n")
    f.write(f"Average Recall Before: {avg_recall_before:.4f}\n")
    f.write(f"Average Recall After: {avg_recall_after:.4f}\n")
    f.write(f"Average Recall Improvement: {improvement:.4f}\n\n")
    f.write("Case Studies:\n")
    f.write(case_studies)
    if queries_df.empty or qrels_df.empty:
        f.write("\nNote: Results are based on placeholder data. Actual analysis requires real queries and relevance judgments.\n")

print("Comparative analysis completed. Results saved to 'comparative_analysis_results.txt'.")