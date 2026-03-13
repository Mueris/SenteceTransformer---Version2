"""Tests for the embedding cache functionality in SentenceTransformer."""

from __future__ import annotations

import numpy as np
import pytest

from sentence_transformers import SentenceTransformer


class TestEmbeddingCache:
    """Test suite for embedding cache functionality."""

    @pytest.fixture
    def model(self):
        """Fixture to create a fresh model for each test."""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model

    def test_cache_stats_initial_state(self, model):
        """Test that cache stats are initialized correctly."""
        stats = model.cache_stats()
        assert stats == {"entries": 0, "hits": 0, "misses": 0}

    def test_cache_stats_after_encoding(self, model):
        """Test cache stats after encoding with cache enabled."""
        sentences = ["First sentence.", "Second sentence."]
        model.encode(sentences, use_cache=True)

        stats = model.cache_stats()
        assert stats["entries"] == 2
        assert stats["hits"] == 0
        assert stats["misses"] == 2

    def test_cache_stats_with_hits(self, model):
        """Test cache stats with cache hits."""
        # First encoding - all misses
        sentences1 = ["First sentence.", "Second sentence."]
        model.encode(sentences1, use_cache=True)

        stats1 = model.cache_stats()
        assert stats1["hits"] == 0
        assert stats1["misses"] == 2

        # Second encoding with one repeated sentence - one hit
        sentences2 = ["First sentence.", "Third sentence."]
        model.encode(sentences2, use_cache=True)

        stats2 = model.cache_stats()
        assert stats2["entries"] == 3
        assert stats2["hits"] == 1
        assert stats2["misses"] == 3

    def test_cache_stats_no_cache(self, model):
        """Test that cache stats don't change when use_cache=False."""
        sentences = ["First sentence.", "Second sentence."]
        model.encode(sentences, use_cache=False)

        stats = model.cache_stats()
        assert stats == {"entries": 0, "hits": 0, "misses": 0}

    def test_clear_cache(self, model):
        """Test clearing the cache."""
        sentences = ["First sentence.", "Second sentence."]
        model.encode(sentences, use_cache=True)

        # Verify cache has entries
        stats_before = model.cache_stats()
        assert stats_before["entries"] == 2

        # Clear cache
        model.clear_cache()

        # Verify cache is empty
        stats_after = model.cache_stats()
        assert stats_after == {"entries": 0, "hits": 0, "misses": 0}

    def test_clear_cache_resets_stats(self, model):
        """Test that clear_cache resets both entries and stats."""
        # First encoding: 1 miss
        model.encode(["Sentence one."], use_cache=True)
        # Second encoding with duplicate: 1 hit, 1 miss
        model.encode(["Sentence one.", "Sentence two."], use_cache=True)

        stats_before = model.cache_stats()
        assert stats_before["entries"] == 2
        assert stats_before["hits"] == 1
        assert stats_before["misses"] == 2  # 1 from first call + 1 new

        # Clear cache
        model.clear_cache()

        # Verify all stats are reset
        stats_after = model.cache_stats()
        assert stats_after == {"entries": 0, "hits": 0, "misses": 0}

    def test_cache_persists_across_calls(self, model):
        """Test that cache persists across multiple encode calls."""
        # First call
        embeddings1 = model.encode(["Test sentence."], use_cache=True)

        # Second call with same sentence
        embeddings2 = model.encode(["Test sentence."], use_cache=True)

        # Embeddings should be identical
        np.testing.assert_array_almost_equal(embeddings1, embeddings2)

        # Should have 1 entry, 1 hit, 1 miss
        stats = model.cache_stats()
        assert stats["entries"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_cache_with_duplicates_in_same_batch(self, model):
        """Test cache behavior with duplicates in the same batch."""
        sentences = [
            "Duplicate sentence.",
            "Unique sentence one.",
            "Duplicate sentence.",  # Duplicate
            "Unique sentence two.",
            "Duplicate sentence.",  # Duplicate
        ]

        embeddings = model.encode(sentences, use_cache=True)

        # Should have 3 entries (unique sentences)
        stats = model.cache_stats()
        assert stats["entries"] == 3
        # All are misses in first call (no prior cache)
        assert stats["hits"] == 0
        assert stats["misses"] == 5

        # Verify duplicates have identical embeddings
        np.testing.assert_array_almost_equal(embeddings[0], embeddings[2])
        np.testing.assert_array_almost_equal(embeddings[0], embeddings[4])

    def test_cache_with_prompts(self, model):
        """Test that different prompts create different cache entries."""
        model.prompts = {"query": "query: ", "document": "document: "}

        # Encode same text with different prompts
        query_emb = model.encode("What is the weather?", prompt_name="query", use_cache=True)
        doc_emb = model.encode("What is the weather?", prompt_name="document", use_cache=True)

        # Should have 2 entries (different prompts)
        stats = model.cache_stats()
        assert stats["entries"] == 2

        # Embeddings should be different
        assert not np.allclose(query_emb, doc_emb)

    def test_encode_query_with_cache(self, model):
        """Test that encode_query supports use_cache parameter."""
        queries = ["Query one?", "Query two?"]
        model.encode_query(queries, use_cache=True)

        stats = model.cache_stats()
        assert stats["entries"] == 2
        assert stats["misses"] == 2

    def test_encode_document_with_cache(self, model):
        """Test that encode_document supports use_cache parameter."""
        docs = ["Document one.", "Document two."]
        model.encode_document(docs, use_cache=True)

        stats = model.cache_stats()
        assert stats["entries"] == 2
        assert stats["misses"] == 2

    def test_cache_stats_are_cumulative(self, model):
        """Test that cache stats accumulate across multiple calls."""
        # Multiple encoding calls
        for i in range(3):
            model.encode([f"Sentence {i}."], use_cache=True)

        stats = model.cache_stats()
        assert stats["entries"] == 3
        assert stats["misses"] == 3

        # Now with some duplicates - both calls are hits
        for _ in range(2):
            model.encode(["Sentence 0."], use_cache=True)

        stats = model.cache_stats()
        assert stats["entries"] == 3  # No new entries
        assert stats["hits"] == 2  # Two hits from duplicates
        assert stats["misses"] == 3  # Still 3, no new misses

    def test_single_string_cache(self, model):
        """Test caching with a single string input."""
        # First call
        emb1 = model.encode("Single sentence.", use_cache=True)

        # Second call
        emb2 = model.encode("Single sentence.", use_cache=True)

        # Should be identical
        np.testing.assert_array_almost_equal(emb1, emb2)

        stats = model.cache_stats()
        assert stats["entries"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_cache_does_not_affect_output_without_cache(self, model):
        """Test that use_cache=False produces same output as before."""
        sentences = ["Test sentence one.", "Test sentence two."]

        # Encode without cache
        emb_no_cache1 = model.encode(sentences, use_cache=False)
        emb_no_cache2 = model.encode(sentences, use_cache=False)

        # Should be identical
        np.testing.assert_array_almost_equal(emb_no_cache1, emb_no_cache2)

        # Stats should be empty
        stats = model.cache_stats()
        assert stats == {"entries": 0, "hits": 0, "misses": 0}

    def test_clear_cache_during_usage(self, model):
        """Test clearing cache in the middle of usage."""
        # First batch
        model.encode(["Sentence A.", "Sentence B."], use_cache=True)
        stats1 = model.cache_stats()
        assert stats1["entries"] == 2

        # Clear cache
        model.clear_cache()

        # Second batch - should start fresh
        model.encode(["Sentence A.", "Sentence C."], use_cache=True)
        stats2 = model.cache_stats()
        assert stats2["entries"] == 2  # Sentence A and C
        assert stats2["hits"] == 0  # No hits because cache was cleared
        assert stats2["misses"] == 2

    def test_semantic_search_basic(self, model):
        """Test semantic_search returns ranked results with text."""
        query = "Weather is nice today"
        corpus = [
            "The weather is lovely today.",
            "I like pizza with cheese.",
            "It is sunny outside.",
            "Driving to the stadium.",
        ]

        results = model.semantic_search(query, corpus, top_k=2)

        assert len(results) == 2
        assert {"corpus_id", "score", "text"} <= results[0].keys()
        assert results[0]["text"] in corpus
        assert results[1]["text"] in corpus
        assert results[0]["score"] >= results[1]["score"]

    def test_semantic_search_top_k_bounds(self, model):
        """Test semantic_search handles top_k larger than corpus size."""
        query = "Test query"
        corpus = ["Sentence one.", "Sentence two."]

        results = model.semantic_search(query, corpus, top_k=10)

        assert len(results) == len(corpus)
        assert all(result["text"] in corpus for result in results)

    def test_semantic_search_empty_corpus(self, model):
        """Test semantic_search returns empty list for empty corpus."""
        results = model.semantic_search("Query", [], top_k=5)
        assert results == []

    def test_semantic_search_with_cache_stats(self, model):
        """Test semantic_search uses cache when enabled."""
        query = "Semantic search query"
        corpus = [
            "Search result one.",
            "Search result two.",
            "Search result three.",
        ]

        # First call populates cache (1 query + 3 corpus = 4 misses)
        model.semantic_search(query, corpus, top_k=2, use_cache=True)
        stats_after_first = model.cache_stats()
        assert stats_after_first["entries"] == 4
        assert stats_after_first["hits"] == 0
        assert stats_after_first["misses"] == 4

        # Second call should have cache hits
        model.semantic_search(query, corpus, top_k=2, use_cache=True)
        stats_after_second = model.cache_stats()
        assert stats_after_second["hits"] == 4
        assert stats_after_second["misses"] == 4
        assert stats_after_second["entries"] == 4

    def test_semantic_search_precomputed_embeddings(self, model):
        """Test semantic_search with precomputed corpus embeddings."""
        query = "What is the weather today?"
        corpus = [
            "The weather is lovely today.",
            "It's so sunny outside!",
            "He drove to the stadium.",
            "Pizza is delicious.",
        ]

        corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

        results_precomputed = model.semantic_search(
            query,
            corpus,
            top_k=3,
            corpus_embeddings=corpus_embeddings,
        )
        results_encoded = model.semantic_search(query, corpus, top_k=3)

        assert [result["corpus_id"] for result in results_precomputed] == [
            result["corpus_id"] for result in results_encoded
        ]
        assert [result["text"] for result in results_precomputed] == [
            result["text"] for result in results_encoded
        ]
        assert [pytest.approx(result["score"], rel=1e-6) for result in results_precomputed] == [
            pytest.approx(result["score"], rel=1e-6) for result in results_encoded
        ]

    def test_semantic_search_precomputed_top_k(self, model):
        """Test semantic_search top_k with precomputed embeddings."""
        query = "Tell me about pizza"
        corpus = [
            "The weather is nice.",
            "Pizza is delicious.",
            "I like pizza with cheese.",
            "Driving to the stadium.",
        ]

        corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

        results_top_k = model.semantic_search(
            query,
            corpus,
            top_k=2,
            corpus_embeddings=corpus_embeddings,
        )

        assert len(results_top_k) == 2
        assert results_top_k[0]["score"] >= results_top_k[1]["score"]
        assert all(result["text"] in corpus for result in results_top_k)

    def test_semantic_search_precomputed_cache_ignored(self, model):
        """Test semantic_search with precomputed embeddings doesn't change cache entries for corpus."""
        query = "Sample query"
        corpus = [
            "Sentence one.",
            "Sentence two.",
            "Sentence three.",
        ]

        corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

        model.semantic_search(
            query,
            corpus,
            top_k=2,
            use_cache=True,
            corpus_embeddings=corpus_embeddings,
        )

        stats = model.cache_stats()
        # Only the query should be cached
        assert stats["entries"] == 1
        assert stats["misses"] == 1

    def test_semantic_search_batch_basic(self, model):
        """Test semantic_search_batch returns results per query."""
        queries = ["Weather today", "Tell me about pizza"]
        corpus = [
            "The weather is lovely today.",
            "It's sunny outside.",
            "Pizza is delicious.",
            "Driving to the stadium.",
        ]

        results = model.semantic_search_batch(queries, corpus, top_k=2)

        assert len(results) == len(queries)
        assert all(len(query_results) == 2 for query_results in results)
        assert all({"corpus_id", "score", "text"} <= query_results[0].keys() for query_results in results)

    def test_semantic_search_batch_sorted(self, model):
        """Test semantic_search_batch results are sorted by score."""
        queries = ["Weather today", "Tell me about pizza"]
        corpus = [
            "The weather is lovely today.",
            "It's sunny outside.",
            "Pizza is delicious.",
            "Driving to the stadium.",
        ]

        results = model.semantic_search_batch(queries, corpus, top_k=3)

        for query_results in results:
            scores = [result["score"] for result in query_results]
            assert scores == sorted(scores, reverse=True)

    def test_semantic_search_batch_top_k(self, model):
        """Test semantic_search_batch respects top_k."""
        queries = ["Weather today", "Tell me about pizza"]
        corpus = [
            "The weather is lovely today.",
            "It's sunny outside.",
            "Pizza is delicious.",
            "Driving to the stadium.",
        ]

        results = model.semantic_search_batch(queries, corpus, top_k=1)

        assert all(len(query_results) == 1 for query_results in results)

    def test_semantic_search_batch_precomputed_embeddings(self, model):
        """Test semantic_search_batch with precomputed corpus embeddings."""
        queries = ["Weather today", "Tell me about pizza"]
        corpus = [
            "The weather is lovely today.",
            "It's sunny outside.",
            "Pizza is delicious.",
            "Driving to the stadium.",
        ]

        corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

        results_precomputed = model.semantic_search_batch(
            queries,
            corpus,
            top_k=2,
            corpus_embeddings=corpus_embeddings,
        )
        results_encoded = model.semantic_search_batch(queries, corpus, top_k=2)

        assert [result["corpus_id"] for result in results_precomputed[0]] == [
            result["corpus_id"] for result in results_encoded[0]
        ]
        assert [result["corpus_id"] for result in results_precomputed[1]] == [
            result["corpus_id"] for result in results_encoded[1]
        ]
        assert [pytest.approx(result["score"], rel=1e-6) for result in results_precomputed[0]] == [
            pytest.approx(result["score"], rel=1e-6) for result in results_encoded[0]
        ]
        assert [pytest.approx(result["score"], rel=1e-6) for result in results_precomputed[1]] == [
            pytest.approx(result["score"], rel=1e-6) for result in results_encoded[1]
        ]

    def test_search_index_stores_corpus(self, model):
        """Test that build_search_index stores corpus and embeddings."""
        corpus = ["Sentence one.", "Sentence two."]
        corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

        index = model.build_search_index(corpus, corpus_embeddings=corpus_embeddings)

        assert index.corpus == corpus
        assert index.corpus_embeddings is corpus_embeddings
        assert index.model is model

    def test_search_index_search_matches_semantic_search(self, model):
        """Test index.search matches semantic_search with precomputed embeddings."""
        query = "Sentence one"
        corpus = ["Sentence one.", "Sentence two.", "Sentence three."]
        corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

        index = model.build_search_index(corpus, corpus_embeddings=corpus_embeddings)

        index_results = index.search(query, top_k=2)
        direct_results = model.semantic_search(
            query,
            corpus,
            top_k=2,
            corpus_embeddings=corpus_embeddings,
        )

        assert [result["corpus_id"] for result in index_results] == [
            result["corpus_id"] for result in direct_results
        ]
        assert [pytest.approx(result["score"], rel=1e-6) for result in index_results] == [
            pytest.approx(result["score"], rel=1e-6) for result in direct_results
        ]

    def test_search_index_search_batch_matches_semantic_search_batch(self, model):
        """Test index.search_batch matches semantic_search_batch with precomputed embeddings."""
        queries = ["Sentence one", "Sentence two"]
        corpus = ["Sentence one.", "Sentence two.", "Sentence three."]
        corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

        index = model.build_search_index(corpus, corpus_embeddings=corpus_embeddings)

        index_results = index.search_batch(queries, top_k=2)
        direct_results = model.semantic_search_batch(
            queries,
            corpus,
            top_k=2,
            corpus_embeddings=corpus_embeddings,
        )

        for index_hits, direct_hits in zip(index_results, direct_results):
            assert [result["corpus_id"] for result in index_hits] == [
                result["corpus_id"] for result in direct_hits
            ]
            assert [pytest.approx(result["score"], rel=1e-6) for result in index_hits] == [
                pytest.approx(result["score"], rel=1e-6) for result in direct_hits
            ]

    def test_search_index_empty_corpus(self, model):
        """Test build_search_index with empty corpus behaves consistently."""
        index = model.build_search_index([])
        assert index.corpus == []
        assert index.search("Query", top_k=2) == []
        assert index.search_batch(["Query"], top_k=2) == []
