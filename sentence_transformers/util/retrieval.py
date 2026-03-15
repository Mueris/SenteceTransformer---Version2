from __future__ import annotations

import heapq
import logging
import math
import queue
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor
from tqdm.autonotebook import tqdm

from .similarity import cos_sim
from .tensor import normalize_embeddings

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sentence_transformers.SentenceTransformer import SentenceTransformer


@dataclass(frozen=True)
class Bm25Indexer:
    idf: dict[str, float]
    avg_doc_len: float
    doc_len: list[int]
    term_frequencies: list[Counter[str]]


def paraphrase_mining(
    model: SentenceTransformer,
    sentences: list[str],
    show_progress_bar: bool = False,
    batch_size: int = 32,
    query_chunk_size: int = 5000,
    corpus_chunk_size: int = 100000,
    max_pairs: int = 500000,
    top_k: int = 100,
    score_function: Callable[[Tensor, Tensor], Tensor] = cos_sim,
    truncate_dim: int | None = None,
    prompt_name: str | None = None,
    prompt: str | None = None,
) -> list[list[float | int]]:
    """
    Given a list of sentences / texts, this function performs paraphrase mining. It compares all sentences against all
    other sentences and returns a list with the pairs that have the highest cosine similarity score.

    Args:
        model (SentenceTransformer): SentenceTransformer model for embedding computation
        sentences (List[str]): A list of strings (texts or sentences)
        show_progress_bar (bool, optional): Plotting of a progress bar. Defaults to False.
        batch_size (int, optional): Number of texts that are encoded simultaneously by the model. Defaults to 32.
        query_chunk_size (int, optional): Search for most similar pairs for #query_chunk_size at the same time. Decrease, to lower memory footprint (increases run-time). Defaults to 5000.
        corpus_chunk_size (int, optional): Compare a sentence simultaneously against #corpus_chunk_size other sentences. Decrease, to lower memory footprint (increases run-time). Defaults to 100000.
        max_pairs (int, optional): Maximal number of text pairs returned. Defaults to 500000.
        top_k (int, optional): For each sentence, we retrieve up to top_k other sentences. Defaults to 100.
        score_function (Callable[[Tensor, Tensor], Tensor], optional): Function for computing scores. By default, cosine similarity. Defaults to cos_sim.
        truncate_dim (int, optional): The dimension to truncate sentence embeddings to. If None, uses the model's ones. Defaults to None.
        prompt_name (Optional[str], optional): The name of a predefined prompt to use when encoding the sentence.
            It must match a key in the model `prompts` dictionary, which can be set during model initialization
            or loaded from the model configuration.

            Ignored if `prompt` is provided. Defaults to None.

        prompt (Optional[str], optional): A raw prompt string to prepend directly to the input sentence during encoding.

            For instance, `prompt="query: "` transforms the sentence "What is the capital of France?" into:
            "query: What is the capital of France?". Use this to override the prompt logic entirely and supply your own prefix.
            This takes precedence over `prompt_name`. Defaults to None.

    Returns:
        List[List[Union[float, int]]]: Returns a list of triplets with the format [score, id1, id2]
    """

    # Compute embedding for the sentences
    embeddings = model.encode(
        sentences,
        show_progress_bar=show_progress_bar,
        batch_size=batch_size,
        convert_to_tensor=True,
        truncate_dim=truncate_dim,
        prompt_name=prompt_name,
        prompt=prompt,
    )

    return paraphrase_mining_embeddings(
        embeddings,
        query_chunk_size=query_chunk_size,
        corpus_chunk_size=corpus_chunk_size,
        max_pairs=max_pairs,
        top_k=top_k,
        score_function=score_function,
    )


def paraphrase_mining_embeddings(
    embeddings: Tensor,
    query_chunk_size: int = 5000,
    corpus_chunk_size: int = 100000,
    max_pairs: int = 500000,
    top_k: int = 100,
    score_function: Callable[[Tensor, Tensor], Tensor] = cos_sim,
) -> list[list[float | int]]:
    """
    Given a list of sentences / texts, this function performs paraphrase mining. It compares all sentences against all
    other sentences and returns a list with the pairs that have the highest cosine similarity score.

    Args:
        embeddings (Tensor): A tensor with the embeddings
        query_chunk_size (int): Search for most similar pairs for #query_chunk_size at the same time. Decrease, to lower memory footprint (increases run-time).
        corpus_chunk_size (int): Compare a sentence simultaneously against #corpus_chunk_size other sentences. Decrease, to lower memory footprint (increases run-time).
        max_pairs (int): Maximal number of text pairs returned.
        top_k (int): For each sentence, we retrieve up to top_k other sentences
        score_function (Callable[[Tensor, Tensor], Tensor]): Function for computing scores. By default, cosine similarity.

    Returns:
        List[List[Union[float, int]]]: Returns a list of triplets with the format [score, id1, id2]
    """

    top_k += 1  # A sentence has the highest similarity to itself. Increase +1 as we are interest in distinct pairs

    # Mine for duplicates
    pairs = queue.PriorityQueue()
    min_score = -1
    num_added = 0

    for corpus_start_idx in range(0, len(embeddings), corpus_chunk_size):
        for query_start_idx in range(0, len(embeddings), query_chunk_size):
            scores = score_function(
                embeddings[query_start_idx : query_start_idx + query_chunk_size],
                embeddings[corpus_start_idx : corpus_start_idx + corpus_chunk_size],
            )

            scores_top_k_values, scores_top_k_idx = torch.topk(
                scores, min(top_k, len(scores[0])), dim=1, largest=True, sorted=False
            )
            scores_top_k_values = scores_top_k_values.cpu().tolist()
            scores_top_k_idx = scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(scores)):
                for top_k_idx, corpus_itr in enumerate(scores_top_k_idx[query_itr]):
                    i = query_start_idx + query_itr
                    j = corpus_start_idx + corpus_itr

                    if i != j and scores_top_k_values[query_itr][top_k_idx] > min_score:
                        pairs.put((scores_top_k_values[query_itr][top_k_idx], i, j))
                        num_added += 1

                        if num_added >= max_pairs:
                            entry = pairs.get()
                            min_score = entry[0]

    # Get the pairs
    added_pairs = set()  # Used for duplicate detection
    pairs_list = []
    while not pairs.empty():
        score, i, j = pairs.get()
        sorted_i, sorted_j = sorted([i, j])

        if sorted_i != sorted_j and (sorted_i, sorted_j) not in added_pairs:
            added_pairs.add((sorted_i, sorted_j))
            pairs_list.append([score, sorted_i, sorted_j])

    # Highest scores first
    pairs_list = sorted(pairs_list, key=lambda x: x[0], reverse=True)
    return pairs_list


def information_retrieval(*args, **kwargs) -> list[list[dict[str, int | float]]]:
    """This function is deprecated. Use semantic_search instead"""
    return semantic_search(*args, **kwargs)


def _tokenize_bm25(text: str) -> list[str]:
    """Tokenize text for BM25 scoring using a simple lowercase split."""
    return [token for token in text.lower().split() if token]


def build_bm25_index(corpus: list[str]) -> Bm25Indexer:
    """Build a lightweight BM25 index for a corpus.

    Args:
        corpus (List[str]): Corpus of documents.

    Returns:
        Bm25Indexer: Precomputed BM25 statistics.
    """
    if not corpus:
        return Bm25Indexer(idf={}, avg_doc_len=0.0, doc_len=[], term_frequencies=[])

    term_frequencies: list[Counter[str]] = []
    doc_len: list[int] = []
    document_frequency: Counter[str] = Counter()

    for document in corpus:
        tokens = _tokenize_bm25(document)
        counts = Counter(tokens)
        term_frequencies.append(counts)
        doc_len.append(sum(counts.values()))
        document_frequency.update(counts.keys())

    corpus_size = len(corpus)
    avg_doc_len = sum(doc_len) / corpus_size if corpus_size else 0.0
    idf = {
        term: math.log(1 + (corpus_size - freq + 0.5) / (freq + 0.5))
        for term, freq in document_frequency.items()
    }

    return Bm25Indexer(
        idf=idf,
        avg_doc_len=avg_doc_len,
        doc_len=doc_len,
        term_frequencies=term_frequencies,
    )


def _bm25_scores(
    index: Bm25Indexer,
    query: str,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Compute BM25 scores for a single query against the indexed corpus."""
    if not index.term_frequencies:
        return []

    tokens = _tokenize_bm25(query)
    if not tokens:
        return [0.0 for _ in index.term_frequencies]

    scores = [0.0 for _ in index.term_frequencies]
    for token in tokens:
        if token not in index.idf:
            continue
        idf = index.idf[token]
        for doc_idx, tf in enumerate(index.term_frequencies):
            freq = tf.get(token, 0)
            if freq == 0:
                continue
            doc_length = index.doc_len[doc_idx]
            denom = freq + k1 * (1 - b + b * doc_length / index.avg_doc_len)
            scores[doc_idx] += idf * (freq * (k1 + 1)) / denom

    return scores


def bm25_search(
    query: str,
    corpus: list[str],
    top_k: int = 10,
    index: Bm25Indexer | None = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[dict[str, int | float]]:
    """Run BM25 search against a corpus.

    Args:
        query (str): Query string.
        corpus (List[str]): Corpus of documents.
        top_k (int, optional): Number of hits to return. Defaults to 10.
        index (Optional[Bm25Indexer], optional): Precomputed BM25 index.
        k1 (float, optional): BM25 k1 parameter. Defaults to 1.5.
        b (float, optional): BM25 b parameter. Defaults to 0.75.

    Returns:
        List[Dict[str, Union[int, float]]]: List of hits containing corpus_id and score.
    """
    if not corpus:
        return []

    if index is None:
        index = build_bm25_index(corpus)

    scores = _bm25_scores(index, query, k1=k1, b=b)
    if not scores:
        return []

    top_k = min(top_k, len(scores))
    ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
    return [{"corpus_id": idx, "score": float(scores[idx])} for idx in ranked]


def hybrid_search(
    query: str,
    corpus: list[str],
    query_embedding: Tensor,
    corpus_embeddings: Tensor,
    top_k: int = 10,
    bm25_index: Bm25Indexer | None = None,
    bm25_weight: float = 0.35,
    dense_weight: float = 0.65,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[dict[str, int | float]]:
    """Hybrid search that combines BM25 and dense similarity scores.

    Args:
        query (str): Query string.
        corpus (List[str]): Corpus of documents.
        query_embedding (Tensor): Query embedding tensor.
        corpus_embeddings (Tensor): Corpus embedding tensor.
        top_k (int, optional): Number of hits to return. Defaults to 10.
        bm25_index (Optional[Bm25Indexer], optional): Precomputed BM25 index.
        bm25_weight (float, optional): Weight for BM25 scores. Defaults to 0.35.
        dense_weight (float, optional): Weight for dense scores. Defaults to 0.65.
        k1 (float, optional): BM25 k1 parameter. Defaults to 1.5.
        b (float, optional): BM25 b parameter. Defaults to 0.75.

    Returns:
        List[Dict[str, Union[int, float]]]: List of hits containing corpus_id and score.
    """
    if not corpus:
        return []

    if bm25_index is None:
        bm25_index = build_bm25_index(corpus)

    bm25_scores = _bm25_scores(bm25_index, query, k1=k1, b=b)
    dense_scores = semantic_search(query_embedding, corpus_embeddings, top_k=len(corpus))

    if not bm25_scores:
        return []

    dense_scores_map = {hit["corpus_id"]: float(hit["score"]) for hit in dense_scores[0]}
    max_bm25 = max(bm25_scores) if bm25_scores else 0.0
    max_dense = max(dense_scores_map.values()) if dense_scores_map else 0.0
    max_bm25 = max_bm25 if max_bm25 > 0 else 1.0
    max_dense = max_dense if max_dense > 0 else 1.0

    combined_scores = []
    for idx, bm25_score in enumerate(bm25_scores):
        dense_score = dense_scores_map.get(idx, 0.0)
        combined = bm25_weight * (bm25_score / max_bm25) + dense_weight * (dense_score / max_dense)
        combined_scores.append(combined)

    top_k = min(top_k, len(combined_scores))
    ranked = sorted(range(len(combined_scores)), key=lambda idx: combined_scores[idx], reverse=True)[:top_k]
    return [{"corpus_id": idx, "score": float(combined_scores[idx])} for idx in ranked]


def hybrid_search_batch(
    queries: list[str],
    corpus: list[str],
    query_embeddings: Tensor,
    corpus_embeddings: Tensor,
    top_k: int = 10,
    bm25_index: Bm25Indexer | None = None,
    bm25_weight: float = 0.35,
    dense_weight: float = 0.65,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[list[dict[str, int | float]]]:
    """Hybrid search for multiple queries."""
    if not corpus or not queries:
        return []

    if bm25_index is None:
        bm25_index = build_bm25_index(corpus)

    if isinstance(query_embeddings, (np.ndarray, np.generic)):
        query_embeddings = torch.from_numpy(query_embeddings)
    elif isinstance(query_embeddings, list):
        query_embeddings = torch.stack(query_embeddings)

    if len(query_embeddings.shape) == 1:
        query_embeddings = query_embeddings.unsqueeze(0)

    if isinstance(corpus_embeddings, (np.ndarray, np.generic)):
        corpus_embeddings = torch.from_numpy(corpus_embeddings)
    elif isinstance(corpus_embeddings, list):
        corpus_embeddings = torch.stack(corpus_embeddings)

    if corpus_embeddings.device != query_embeddings.device:
        query_embeddings = query_embeddings.to(corpus_embeddings.device)

    dense_scores = semantic_search(query_embeddings, corpus_embeddings, top_k=len(corpus))

    results = []
    for query, dense_hits in zip(queries, dense_scores):
        bm25_scores = _bm25_scores(bm25_index, query, k1=k1, b=b)
        if not bm25_scores:
            results.append([])
            continue
        dense_scores_map = {hit["corpus_id"]: float(hit["score"]) for hit in dense_hits}
        max_bm25 = max(bm25_scores) if bm25_scores else 0.0
        max_dense = max(dense_scores_map.values()) if dense_scores_map else 0.0
        max_bm25 = max_bm25 if max_bm25 > 0 else 1.0
        max_dense = max_dense if max_dense > 0 else 1.0

        combined_scores = []
        for idx, bm25_score in enumerate(bm25_scores):
            dense_score = dense_scores_map.get(idx, 0.0)
            combined = bm25_weight * (bm25_score / max_bm25) + dense_weight * (dense_score / max_dense)
            combined_scores.append(combined)

        top_k_for_query = min(top_k, len(combined_scores))
        ranked = sorted(range(len(combined_scores)), key=lambda idx: combined_scores[idx], reverse=True)[:top_k_for_query]
        results.append([{"corpus_id": idx, "score": float(combined_scores[idx])} for idx in ranked])

    return results


def expand_query(
    model: "SentenceTransformer",
    query: str,
    corpus: list[str] | None = None,
    top_k: int = 5,
    threshold: float = 0.8,
) -> list[str]:
    """Generate expanded queries based on semantic similarity using embeddings.

    This function generates additional queries that are semantically similar to the
    original query. It uses the model's embedding capabilities to find or generate
    semantically similar text, avoiding rule-based approaches.

    Args:
        model (SentenceTransformer): The SentenceTransformer model for encoding.
        query (str): The original query string.
        corpus (Optional[List[str]], optional): Optional corpus of sentences to
            find similar ones. If provided, similar sentences are found from the
            corpus using semantic similarity. Defaults to None.
        top_k (int, optional): Maximum number of expanded queries to return.
            Defaults to 5.
        threshold (float, optional): Minimum cosine similarity threshold for
            considering a query as expanded. Defaults to 0.8.

    Returns:
        List[str]: List of expanded queries that are semantically similar to
            the original query. The original query is included if it meets the
            threshold.

    Example:
        ::

            from sentence_transformers import SentenceTransformer
            from sentence_transformers.util import expand_query

            model = SentenceTransformer("all-MiniLM-L6-v2")
            corpus = ["Machine learning is AI.", "Deep learning uses neural nets."]
            expanded = expand_query(model, "What is ML?", corpus=corpus, top_k=3)
    """
    if not query:
        return []

    # Build candidates list
    candidates = [query]
    if corpus:
        candidates.extend(corpus)

    # Use paraphrase mining to find semantically similar sentences
    pairs = paraphrase_mining(
        model,
        candidates,
        max_pairs=top_k * len(candidates),
        top_k=top_k,
    )

    # Find pairs involving the original query (index 0)
    expanded = set()
    for score, idx1, idx2 in pairs:
        if score < threshold:
            continue
        if idx1 == 0 and idx2 < len(candidates):
            expanded.add(candidates[idx2])
        elif idx2 == 0 and idx1 < len(candidates):
            expanded.add(candidates[idx1])

    # Always include the original query if not already present and we have space
    result = list(expanded)
    if query not in result:
        result.insert(0, query)

    return result[:top_k]


def semantic_search(
    query_embeddings: Tensor,
    corpus_embeddings: Tensor,
    query_chunk_size: int = 100,
    corpus_chunk_size: int = 500000,
    top_k: int = 10,
    score_function: Callable[[Tensor, Tensor], Tensor] = cos_sim,
) -> list[list[dict[str, int | float]]]:
    """
    This function performs by default a cosine similarity search between a list of query embeddings  and a list of corpus embeddings.
    It can be used for Information Retrieval / Semantic Search for corpora up to about 1 Million entries.

    Args:
        query_embeddings (:class:`~torch.Tensor`): A 2 dimensional tensor with the query embeddings. Can be a sparse tensor.
        corpus_embeddings (:class:`~torch.Tensor`): A 2 dimensional tensor with the corpus embeddings. Can be a sparse tensor.
        query_chunk_size (int, optional): Process 100 queries simultaneously. Increasing that value increases the speed, but requires more memory. Defaults to 100.
        corpus_chunk_size (int, optional): Scans the corpus 100k entries at a time. Increasing that value increases the speed, but requires more memory. Defaults to 500000.
        top_k (int, optional): Retrieve top k matching entries. Defaults to 10.
        score_function (Callable[[:class:`~torch.Tensor`, :class:`~torch.Tensor`], :class:`~torch.Tensor`], optional): Function for computing scores. By default, cosine similarity.

    Returns:
        List[List[Dict[str, Union[int, float]]]]: A list with one entry for each query. Each entry is a list of dictionaries with the keys 'corpus_id' and 'score', sorted by decreasing cosine similarity scores.
    """

    if isinstance(query_embeddings, (np.ndarray, np.generic)):
        query_embeddings = torch.from_numpy(query_embeddings)
    elif isinstance(query_embeddings, list):
        query_embeddings = torch.stack(query_embeddings)

    if len(query_embeddings.shape) == 1:
        query_embeddings = query_embeddings.unsqueeze(0)

    if isinstance(corpus_embeddings, (np.ndarray, np.generic)):
        corpus_embeddings = torch.from_numpy(corpus_embeddings)
    elif isinstance(corpus_embeddings, list):
        corpus_embeddings = torch.stack(corpus_embeddings)

    # Check that corpus and queries are on the same device
    if corpus_embeddings.device != query_embeddings.device:
        query_embeddings = query_embeddings.to(corpus_embeddings.device)

    queries_result_list = [[] for _ in range(len(query_embeddings))]

    for query_start_idx in range(0, len(query_embeddings), query_chunk_size):
        query_end_idx = min(query_start_idx + query_chunk_size, len(query_embeddings))
        if query_embeddings.is_sparse:
            indices = torch.arange(query_start_idx, query_end_idx, device=query_embeddings.device)
            query_chunk = query_embeddings.index_select(0, indices)
        else:
            query_chunk = query_embeddings[query_start_idx:query_end_idx]

        # Iterate over chunks of the corpus
        for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
            corpus_end_idx = min(corpus_start_idx + corpus_chunk_size, len(corpus_embeddings))
            if corpus_embeddings.is_sparse:
                indices = torch.arange(corpus_start_idx, corpus_end_idx, device=corpus_embeddings.device)
                corpus_chunk = corpus_embeddings.index_select(0, indices)
            else:
                corpus_chunk = corpus_embeddings[corpus_start_idx:corpus_end_idx]

            # Compute cosine similarities
            cos_scores = score_function(query_chunk, corpus_chunk)

            # Get top-k scores
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                cos_scores, min(top_k, len(cos_scores[0])), dim=1, largest=True, sorted=False
            )
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(cos_scores)):
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                    corpus_id = corpus_start_idx + sub_corpus_id
                    query_id = query_start_idx + query_itr
                    if len(queries_result_list[query_id]) < top_k:
                        heapq.heappush(
                            queries_result_list[query_id], (score, corpus_id)
                        )  # heaqp tracks the quantity of the first element in the tuple
                    else:
                        heapq.heappushpop(queries_result_list[query_id], (score, corpus_id))

    # change the data format and sort
    for query_id in range(len(queries_result_list)):
        for doc_itr in range(len(queries_result_list[query_id])):
            score, corpus_id = queries_result_list[query_id][doc_itr]
            queries_result_list[query_id][doc_itr] = {"corpus_id": corpus_id, "score": score}
        queries_result_list[query_id] = sorted(queries_result_list[query_id], key=lambda x: x["score"], reverse=True)

    return queries_result_list


def community_detection(
    embeddings: torch.Tensor | np.ndarray,
    threshold: float = 0.75,
    min_community_size: int = 10,
    batch_size: int = 1024,
    show_progress_bar: bool = False,
) -> list[list[int]]:
    """
    Function for Fast Community Detection.

    Finds in the embeddings all communities, i.e. embeddings that are close (closer than threshold).
    Returns only communities that are larger than min_community_size. The communities are returned
    in decreasing order. The first element in each list is the central point in the community.

    Args:
        embeddings (torch.Tensor or numpy.ndarray): The input embeddings.
        threshold (float): The threshold for determining if two embeddings are close. Defaults to 0.75.
        min_community_size (int): The minimum size of a community to be considered. Defaults to 10.
        batch_size (int): The batch size for computing cosine similarity scores. Defaults to 1024.
        show_progress_bar (bool): Whether to show a progress bar during computation. Defaults to False.

    Returns:
        List[List[int]]: A list of communities, where each community is represented as a list of indices.
    """
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)

    threshold = torch.tensor(threshold, device=embeddings.device)
    embeddings = normalize_embeddings(embeddings)

    extracted_communities = []

    # Maximum size for community
    min_community_size = min(min_community_size, len(embeddings))
    sort_max_size = min(max(2 * min_community_size, 50), len(embeddings))

    for start_idx in tqdm(
        range(0, len(embeddings), batch_size), desc="Finding clusters", disable=not show_progress_bar
    ):
        # Compute cosine similarity scores
        cos_scores = embeddings[start_idx : start_idx + batch_size] @ embeddings.T

        # Use a torch-heavy approach if the embeddings are on CUDA, otherwise a loop-heavy one
        if embeddings.device.type in ["cuda", "npu"]:
            # Threshold the cos scores and determine how many close embeddings exist per embedding
            threshold_mask = cos_scores >= threshold
            row_wise_count = threshold_mask.sum(1)

            # Only consider embeddings with enough close other embeddings
            large_enough_mask = row_wise_count >= min_community_size
            if not large_enough_mask.any():
                continue

            row_wise_count = row_wise_count[large_enough_mask]
            cos_scores = cos_scores[large_enough_mask]

            # The max is the largest potential community, so we use that in topk
            k = row_wise_count.max()
            _, top_k_indices = cos_scores.topk(k=k, largest=True)

            # Use the row-wise count to slice the indices
            for count, indices in zip(row_wise_count, top_k_indices):
                extracted_communities.append(indices[:count].tolist())
        else:
            # Minimum size for a community
            top_k_values, _ = cos_scores.topk(k=min_community_size, largest=True)

            # Filter for rows >= min_threshold
            for i in range(len(top_k_values)):
                if top_k_values[i][-1] >= threshold:
                    # Only check top k most similar entries
                    top_val_large, top_idx_large = cos_scores[i].topk(k=sort_max_size, largest=True)

                    # Check if we need to increase sort_max_size
                    while top_val_large[-1] > threshold and sort_max_size < len(embeddings):
                        sort_max_size = min(2 * sort_max_size, len(embeddings))
                        top_val_large, top_idx_large = cos_scores[i].topk(k=sort_max_size, largest=True)

                    extracted_communities.append(top_idx_large[top_val_large >= threshold].tolist())

    # Largest cluster first
    extracted_communities = sorted(extracted_communities, key=lambda x: len(x), reverse=True)

    # Step 2) Remove overlapping communities
    unique_communities = []
    extracted_ids = set()

    for cluster_id, community in enumerate(extracted_communities):
        non_overlapped_community = []
        for idx in community:
            if idx not in extracted_ids:
                non_overlapped_community.append(idx)

        if len(non_overlapped_community) >= min_community_size:
            unique_communities.append(non_overlapped_community)
            extracted_ids.update(non_overlapped_community)

    unique_communities = sorted(unique_communities, key=lambda x: len(x), reverse=True)

    return unique_communities
