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


def _ngram_tokens(tokens: list[str], n: int) -> list[str]:
    """Return a list of n-gram strings joined with underscores."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return ["_".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def build_idf(token_lists: list[list[str]], max_n: int = 3) -> dict[int, dict[str, float]]:
    """Build IDF dictionaries for 1-grams through max_n-grams."""
    return {
        n: _inverse_document_frequency(
            [_ngram_tokens(tokens, n) for tokens in token_lists],
            len(token_lists),
        )
        for n in range(1, max_n + 1)
    }


def build_tfidf_vector(
    tokens: list[str],
    idf_by_n: dict[int, dict[str, float]],
    weights: tuple[float, float, float] = (0.6, 0.3, 0.1),
) -> dict[str, float]:
    """Build a weighted TF-IDF vector using unigram/bigram/trigram interpolation."""
    vec: dict[str, float] = {}
    for n, weight in zip((1, 2, 3), weights):
        if weight <= 0:
            continue
        ngrams = _ngram_tokens(tokens, n)
        tf = _term_frequency(ngrams)
        for term, value in tf.items():
            vec[term] = vec.get(term, 0.0) + value * idf_by_n[n].get(term, 0.0) * weight
    return vec


def compute_tfidf_scores(
    token_lists: list[list[str]],
    length_norm: bool = True,
    weights: tuple[float, float, float] = (0.6, 0.3, 0.1),
) -> list[float]:
    """
    Compute a TF-IDF importance score for each sentence using linear
    interpolation over unigrams, bigrams, and trigrams.

    Parameters
    ----------
    token_lists : preprocessed token lists (one per sentence)
    length_norm : if True, divide score by sqrt(sentence_length)
    weights     : interpolation weights for (uni, bi, tri)

    Returns
    -------
    List of float scores, one per sentence.
    """
    n = len(token_lists)
    idf_by_n = build_idf(token_lists)

    scores: list[float] = []
    for tokens in token_lists:
        if not tokens:
            scores.append(0.0)
            continue

        raw_score = 0.0
        for ngram_size, weight in zip((1, 2, 3), weights):
            if weight <= 0:
                continue
            ngrams = _ngram_tokens(tokens, ngram_size)
            tf = _term_frequency(ngrams)
            raw_score += weight * sum(tf[t] * idf_by_n[ngram_size].get(t, 0.0) for t in tf)

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
    edge_boost: float = 1.05,
    middle_penalty: float = 0.95,
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
