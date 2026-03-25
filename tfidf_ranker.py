"""
tfidf_ranker.py
---------------
Compute TF-IDF scores for each sentence in a document and rank them.

Score formula
-------------
  score(s) = sum_t( tf(t,s) * idf(t) ) / sqrt(L)

where L is the number of tokens in sentence s (length normalisation).
Length normalisation can be disabled via ``length_norm=False``.

Position-aware weighting
------------------------
Use ``apply_position_weights`` to boost opening/closing sentences and
downweight middle sentences with a smooth U-shaped weighting curve.
"""

import math
from collections import Counter


def _term_frequency(token_list: list[str]) -> dict[str, float]:
    """Raw TF: count of each term divided by sentence length."""
    if not token_list:
        return {}
    total = len(token_list)
    counts = Counter(token_list)
    return {t: c / total for t, c in counts.items()}


def _inverse_document_frequency(token_lists: list[list[str]], n_docs: int) -> dict[str, float]:
    """
    Smooth IDF: log( (N + 1) / (df + 1) ) + 1

    This avoids zero-division and prevents complete suppression of terms
    that appear in every sentence.
    """
    df: dict[str, int] = {}
    for tokens in token_lists:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1

    idf: dict[str, float] = {}
    for term, freq in df.items():
        idf[term] = math.log((n_docs + 1) / (freq + 1)) + 1
    return idf


def compute_tfidf_scores(
    token_lists: list[list[str]],
    length_norm: bool = True,
) -> list[float]:
    """
    Compute a TF-IDF importance score for each sentence.

    Parameters
    ----------
    token_lists : preprocessed token lists (one per sentence)
    length_norm : if True, divide score by sqrt(sentence_length)

    Returns
    -------
    List of float scores, one per sentence.
    """
    n = len(token_lists)
    idf = _inverse_document_frequency(token_lists, n)

    scores: list[float] = []
    for tokens in token_lists:
        if not tokens:
            scores.append(0.0)
            continue

        tf = _term_frequency(tokens)
        raw_score = sum(tf[t] * idf.get(t, 0.0) for t in tf)

        if length_norm:
            raw_score /= math.sqrt(len(tokens))

        scores.append(raw_score)

    return scores


def rank_sentences(scores: list[float]) -> list[tuple[int, float]]:
    """
    Return (index, score) pairs sorted by descending score.
    """
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return indexed


def apply_position_weights(
    scores: list[float],
    edge_boost: float = 1.25,
    middle_penalty: float = 0.85,
) -> list[float]:
    """
    Apply sentence-position weighting to a score list.

    Sentences near the beginning and end receive higher weights, while
    middle sentences receive lower weights. The weighting is symmetric
    and smooth across the document.
    """
    n = len(scores)
    if n <= 1:
        return scores[:]

    weighted_scores: list[float] = []
    for i, score in enumerate(scores):
        x = i / (n - 1)
        edge_strength = (2 * x - 1) ** 2
        weight = middle_penalty + (edge_boost - middle_penalty) * edge_strength
        weighted_scores.append(score * weight)

    return weighted_scores
