# LLM-IR System

This repository contains an implementation of a Learning-to-Rank (LTR) Information Retrieval (IR) system using Large Language Models (LLMs), developed as part of an assignment. The system processes the TREC-COVID dataset and includes retrieval, evaluation, query expansion, and comparative analysis.

## Project Structure
- `LLM-embeddings.py`: Part 1 - LLM-based retrieval system with document encoding and FAISS indexing.
- `evaluate_recall.py`: Part 2 - Evaluates retrieval performance using recall.
- `query_expansion.py`: Part 3 - Implements query expansion using Relevance Language Modeling (RLM).
- `comparative_analysis.py`: Part 5 - Compares retrieval performance before and after query expansion.

## Prerequisites
- Python 3.x
- Required libraries: `sentence-transformers`, `faiss-cpu`, `pandas`, `numpy`, `nltk`, `gensim`
  - Install with: `pip install sentence-transformers faiss-cpu pandas numpy nltk gensim`
- NLTK data: Run `python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"`

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Sanawer-Batool/LLM-IR-system.git
   cd LLM-IR
