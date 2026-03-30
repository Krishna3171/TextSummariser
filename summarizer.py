"""
summarizer.py
-------------
High-level summarization pipeline:
  1. Preprocess document.
  2. Rank sentences with TF-IDF.
  3. Filter redundant sentences via cosine similarity.
  4. Return top-N sentences in original document order.
"""

import math

from preprocess import preprocess_document
from tfidf_ranker import (
    apply_position_weights,
    build_idf,
    build_tfidf_vector,
    compute_tfidf_scores,
    rank_sentences,
)


# ---------------------------------------------------------------------------
# Cosine similarity helpers
# ---------------------------------------------------------------------------

def _tfidf_vector(tokens: list[str], idf: dict[int, dict[str, float]]) -> dict[str, float]:
    """TF-IDF vector for a single sentence token list."""
    if not tokens:
        return {}
    return build_tfidf_vector(tokens, idf)


def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Cosine similarity between two sparse TF-IDF vectors."""
    dot = sum(vec_a.get(t, 0.0) * v for t, v in vec_b.items())
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Main summarizer
# ---------------------------------------------------------------------------

def summarize(
    text: str,
    n_sentences: int = 3,
    use_stemming: bool = False,
    use_spacy=None,
    length_norm: bool = True,
    similarity_threshold: float = 0.7,
    position_aware: bool = True,
    edge_boost: float = 1.25,
    middle_penalty: float = 0.85,
) -> dict:
    """
    Produce an extractive summary of *text*.

    Parameters
    ----------
    text                 : input document string
    n_sentences          : number of sentences to include in the summary
    use_stemming         : use stemming instead of lemmatisation
    use_spacy            : loaded spaCy model or None
    length_norm          : apply sqrt(L) length normalisation to TF-IDF scores
    similarity_threshold : cosine similarity above which a sentence is
                           considered redundant and skipped (0–1)
    position_aware       : if True, boost start/end sentence scores and
                           downweight middle sentences
    edge_boost           : multiplicative weight near document edges
    middle_penalty       : multiplicative weight near document middle

    Returns
    -------
    dict with keys:
      'summary'          : summary string
      'selected_indices' : original sentence indices chosen
      'scores'           : {index: score} for selected sentences
      'all_sentences'    : all split sentences
      'all_scores'       : TF-IDF score for every sentence
    """
    sentences, token_lists = preprocess_document(
        text, use_stemming=use_stemming, use_spacy=use_spacy
    )

    if not sentences:
        return {
            "summary": "",
            "selected_indices": [],
            "scores": {},
            "all_sentences": [],
            "all_scores": [],
        }

    scores = compute_tfidf_scores(token_lists, length_norm=length_norm)
    if position_aware:
        scores = apply_position_weights(
            scores,
            edge_boost=edge_boost,
            middle_penalty=middle_penalty,
        )
    ranked = rank_sentences(scores)

    idf = build_idf(token_lists)
    vectors = [_tfidf_vector(tl, idf) for tl in token_lists]

    # Greedily select sentences that are not too similar to already-chosen ones
    selected_indices: list[int] = []
    selected_vectors: list[dict[str, float]] = []

    for idx, _score in ranked:
        if len(selected_indices) >= n_sentences:
            break

        candidate_vec = vectors[idx]
        redundant = any(
            _cosine_similarity(candidate_vec, sv) >= similarity_threshold
            for sv in selected_vectors
        )
        if not redundant:
            selected_indices.append(idx)
            selected_vectors.append(candidate_vec)

    # Restore original document order
    selected_indices.sort()

    summary = " ".join(sentences[i] for i in selected_indices)

    return {
        "summary": summary,
        "selected_indices": selected_indices,
        "scores": {i: scores[i] for i in selected_indices},
        "all_sentences": sentences,
        "all_scores": scores,
    }
