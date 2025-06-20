from datasets import load_dataset
import pandas as pd
import os
import json

# Load the TREC-COVID dataset
dataset = load_dataset("nreimers/trec-covid")
print("Available splits:", dataset.keys())
print("Features in train split:", dataset['train'].features)

# Extract all data from the train split
df_all = pd.DataFrame(dataset['train'])

# Save all columns to inspect them
print("Columns in train split:", df_all.columns.tolist())

# Extract documents (assuming _id, title, text, metadata are present)
documents = df_all[['_id', 'title', 'text', 'metadata']].copy()
documents.to_csv('trec_covid_data/trec_covid_documents.csv', index=False)

# Attempt to extract queries and qrels
queries = df_all[['query_id', 'query']].dropna() if all(col in df_all.columns for col in ['query_id', 'query']) else pd.DataFrame(columns=['query_id', 'query'])
qrels = df_all[['query_id', '_id', 'relevance']].dropna() if all(col in df_all.columns for col in ['query_id', '_id', 'relevance']) else pd.DataFrame(columns=['query_id', 'doc_id', 'relevance'])

# Save extracted data
os.makedirs('trec_covid_data', exist_ok=True)
if not queries.empty:
    queries.to_csv('trec_covid_data/trec_covid_queries.csv', index=False)
else:
    print("No 'query_id' or 'query' columns found. Queries may need manual loading.")
if not qrels.empty:
    qrels.to_csv('trec_covid_data/trec_covid_qrels.tsv', sep='\t', index=False, header=False)
else:
    print("No 'relevance' column found. Relevance judgments may need manual loading.")
    # Create empty qrels file as a placeholder
    pd.DataFrame(columns=['query_id', 'doc_id', 'relevance']).to_csv('trec_covid_data/trec_covid_qrels.tsv', sep='\t', index=False)

print("Dataset extraction attempt completed. Check 'trec_covid_data' directory for files.")