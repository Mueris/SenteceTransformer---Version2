SentenceTransformer-V2

SentenceTransformer-V2 is an extended and experimental version of the original sentence-transformers
 library.
This project explores additional retrieval and caching capabilities on top of the original embedding pipeline to support efficient semantic search workflows.

The goal of this repository is to experiment with improvements that make semantic search pipelines more practical for real-world applications such as large-scale retrieval systems, search engines, and AI-powered document discovery.

Overview

This repository builds upon the core functionality of the original Sentence Transformers library while introducing several enhancements for embedding caching, semantic search utilities, hybrid retrieval, and reusable search indexing.

These improvements focus on:

reducing redundant embedding computations

improving query performance

enabling reusable search pipelines

providing better observability of embedding cache behavior

combining lexical and semantic retrieval techniques

Key Improvements
1. Embedding Cache System

A lightweight caching system was introduced to avoid recomputing embeddings for previously encoded inputs.

Features:

optional caching during encoding

automatic reuse of cached embeddings

transparent integration with the existing encode() method

Example:

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = ["machine learning", "artificial intelligence", "machine learning"]

embeddings = model.encode(sentences, use_cache=True)
2. Cache Statistics API

A new method was added to inspect cache usage.

model.cache_stats()

Returns:

{
  "entries": int,
  "hits": int,
  "misses": int
}

This allows developers to monitor cache efficiency and understand embedding reuse patterns.

3. Cache Management

A cache management API was introduced.

model.clear_cache()

This clears all cached embeddings and resets cache statistics.

4. Semantic Search Utility

A simplified semantic search interface was added directly to the model.

results = model.semantic_search(
    query="What is machine learning?",
    corpus=[
        "Machine learning is a field of AI.",
        "Pizza is delicious.",
        "The weather is nice today."
    ],
    top_k=2
)

Example output:

[
  {"corpus_id": 0, "score": 0.82, "text": "Machine learning is a field of AI."},
  {"corpus_id": 2, "score": 0.41, "text": "The weather is nice today."}
]
5. Batch Semantic Search

A batch search interface was added for handling multiple queries efficiently.

results = model.semantic_search_batch(
    queries=["machine learning", "pizza"],
    corpus=documents,
    top_k=3
)

This enables efficient multi-query retrieval pipelines.

6. Reusable Search Index

A reusable semantic search index was introduced to avoid recomputing corpus embeddings for repeated queries.

index = model.build_search_index(corpus)

results = index.search("machine learning", top_k=3)

The index stores:

corpus texts

corpus embeddings

model reference

and provides:

index.search(query)
index.search_batch(queries)

This allows efficient repeated queries against the same corpus.

7. Hybrid Search (BM25 + Dense Retrieval)

Hybrid search combines keyword-based BM25 scoring with dense embedding similarity.
This allows the system to capture both lexical matches and semantic similarity.

results = model.hybrid_search(
    query="machine learning",
    corpus=corpus,
    top_k=2
)

Hybrid search indices can also be reused for faster repeated queries.

index = model.build_hybrid_search_index(corpus)

results = index.search(
    "machine learning",
    top_k=2,
    bm25_weight=0.4,
    dense_weight=0.6
)
8. Query Expansion

Query expansion generates additional queries based on semantic similarity to improve retrieval coverage.

expanded = model.expand_query(
    "What is ML?",
    corpus=corpus,
    top_k=3
)

These expanded queries can be used to perform additional searches and increase recall in retrieval systems.

9. Embedded Query Expansion in Search

Query expansion can be directly integrated into semantic or hybrid search pipelines.

Example with semantic search:

results = model.semantic_search(
    "What is ML?",
    corpus,
    top_k=2,
    use_expanded_queries=True
)

Example with hybrid search:

results = model.hybrid_search(
    "machine learning",
    corpus,
    top_k=3,
    use_expanded_queries=True
)

This allows automatic query expansion without manually generating queries.

10. Similarity Selection (Cosine vs Dot Product)

The search pipeline now allows selecting the similarity function.

results = index.search(
    "machine learning",
    top_k=3,
    similarity="dot"
)

Supported options:

"cosine" (default) – uses cosine similarity

"dot" – normalizes embeddings and uses dot product

Dot product with normalized vectors is mathematically equivalent to cosine similarity but can be more efficient when embeddings are reused.

Example Workflow
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

corpus = [
    "Machine learning enables computers to learn from data.",
    "Pizza is one of the most popular foods.",
    "The weather is sunny today."
]

index = model.build_search_index(corpus)

results = index.search("What is machine learning?", top_k=2)

for r in results:
    print(r["text"], r["score"])
Differences from the Original Repository

Compared to the original repository, this version introduces:

embedding caching system

cache statistics and monitoring

cache management utilities

simplified semantic search API

batch semantic search

reusable semantic search index

hybrid search (BM25 + dense retrieval)

query expansion using semantic similarity

embedded query expansion in search pipelines

similarity selection (cosine or dot product)

These additions focus on improving the usability of the library for retrieval and search-based applications.

Intended Use

This repository is intended for:

experimentation with semantic search pipelines

research and prototyping

exploring performance improvements for embedding-based retrieval systems

Credits

This project builds upon the excellent work of the Sentence Transformers library developed by the Hugging Face community.

Original repository:
https://github.com/huggingface/sentence-transformers

All core functionality belongs to the original authors.
This repository only adds experimental improvements on top of their work.

License

This project follows the same license as the original Sentence Transformers repository.
