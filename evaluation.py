"""
evaluation.py
-------------
Lightweight intrinsic evaluation metrics for extractive summarisation.

Metrics reported
----------------
- compression_ratio : len(summary) / len(source)  (character-level)
- sentence_coverage : selected / total sentences
- avg_score         : mean TF-IDF score of selected sentences
- density           : avg tokens per selected sentence
"""


def evaluate(result: dict) -> dict:
    """
    Compute basic quality metrics from a summarizer result dict.

    Parameters
    ----------
    result : dict returned by summarizer.summarize()

    Returns
    -------
    dict of metric_name -> value (floats, 2 d.p.)
    """
    summary = result["summary"]
    all_sentences = result["all_sentences"]
    selected_indices = result["selected_indices"]
    scores = result["scores"]

    source_text = " ".join(all_sentences)
    n_total = len(all_sentences)
    n_selected = len(selected_indices)

    compression_ratio = (
        round(len(summary) / len(source_text), 4) if source_text else 0.0
    )
    sentence_coverage = (
        round(n_selected / n_total, 4) if n_total else 0.0
    )
    avg_score = (
        round(sum(scores.values()) / n_selected, 4) if n_selected else 0.0
    )

    # Average token count of selected sentences
    token_lengths = [
        len(all_sentences[i].split()) for i in selected_indices
    ]
    density = round(sum(token_lengths) / n_selected, 2) if n_selected else 0.0

    return {
        "compression_ratio": compression_ratio,
        "sentence_coverage": sentence_coverage,
        "avg_tfidf_score": avg_score,
        "avg_tokens_per_sentence": density,
    }


def print_evaluation(metrics: dict) -> None:
    """Pretty-print evaluation metrics to stdout."""
    print("\n--- Evaluation Metrics ---")
    for k, v in metrics.items():
        label = k.replace("_", " ").title()
        print(f"  {label:<30} {v}")
    print("--------------------------")
