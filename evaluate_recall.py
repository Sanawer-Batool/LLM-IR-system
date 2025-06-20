from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import os

model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("faiss_index.idx")  # Load the pre-computed index from Part 1

docs_path = 'trec_covid_data/trec_covid_documents.csv'
df = pd.read_csv(docs_path)
documents = df['text'].tolist()
doc_ids = df['_id'].tolist()

# Retrieval function
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

# Compute recall for a single query
def compute_recall(retrieved_docs, query_id, qrels):
    relevant_docs = set(qrels[qrels['query_id'] == query_id]['doc_id'].dropna())
    retrieved_doc_ids = set(doc_id for doc_id, _ in retrieved_docs)
    if not relevant_docs:
        return 0.0  # Recall is 0 if no relevant documents are defined
    return len(retrieved_doc_ids & relevant_docs) / len(relevant_docs)

# Evaluate recall for all queries
recall_scores = {}
for qid, query in zip(query_ids, queries):
    retrieved = retrieve_documents(query, k=10)  # Retrieve top-10 documents
    recall = compute_recall(retrieved, qid, qrels)
    recall_scores[qid] = recall
    print(f"Query ID: {qid}, Query: {query[:50]}..., Recall: {recall:.4f}")

# Report average recall
average_recall = np.mean(list(recall_scores.values())) if recall_scores else 0.0
print(f"\nAverage Recall across all queries: {average_recall:.4f}")

with open('recall_results.txt', 'w') as f:
    f.write("Recall per Query:\n")
    for qid, recall in recall_scores.items():
        f.write(f"Query ID: {qid}, Query: {queries[query_ids.index(qid)][:50]}..., Recall: {recall:.4f}\n")
    f.write(f"\nAverage Recall: {average_recall:.4f}\n")
    if queries_df.empty or qrels_df.empty:
        f.write("\nNote: Results are based on placeholder data. Actual evaluation requires real queries and relevance judgments.\n")

print("Recall evaluation completed. Results saved to 'recall_results.txt'.")