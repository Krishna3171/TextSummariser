"""
main.py
-------
CLI entry point for the TF-IDF extractive text summarizer.

Usage examples
--------------
Summarise a file:
    python main.py --file datasets/sample.txt --sentences 3 --show-details --show-eval

Summarise inline text:
    python main.py --text "Natural language processing helps computers understand text.
    Summarization reduces long text." --sentences 2
"""

import argparse
import sys

from summarizer import summarize
from evaluation import evaluate, print_evaluation


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extractive text summarizer using TF-IDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input source (mutually exclusive but not enforced here for better UX)
    p.add_argument("--text", type=str, default=None,
                   help="Input text string to summarize.")
    p.add_argument("--file", type=str, default="datasets/sample.txt",
                   help="Path to a plain-text file to summarize (default: datasets/sample.txt).")

    # Summarization settings
    p.add_argument("--sentences", type=int, default=3,
                   help="Number of sentences in the output summary (default: 3).")
    p.add_argument("--similarity-threshold", type=float, default=0.7,
                   help="Cosine similarity threshold for redundancy filtering (default: 0.7).")
    p.add_argument("--disable-length-normalization", action="store_true",
                   help="Turn off sqrt(L) sentence-length normalization.")

    # Preprocessing settings
    p.add_argument("--stemming", action="store_true",
                   help="Use Porter stemming instead of lemmatization.")
    p.add_argument("--use-spacy", action="store_true",
                   help="Use spaCy tokenizer/lemmatizer (requires en_core_web_sm).")

    # Output settings
    p.add_argument("--show-details", action="store_true",
                   help="Print each selected sentence with its TF-IDF score.")
    p.add_argument("--show-eval", action="store_true",
                   help="Print basic evaluation metrics.")

    return p


def load_text(args) -> str:
    if args.text:
        return args.text

    try:
        with open(args.file, "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        print(f"Error: file not found — '{args.file}'", file=sys.stderr)
        sys.exit(1)


def load_spacy_model():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception:
        print(
            "Warning: spaCy model 'en_core_web_sm' not available. "
            "Falling back to NLTK.",
            file=sys.stderr,
        )
        return None


def main():
    parser = build_parser()
    args = parser.parse_args()

    text = load_text(args)

    spacy_model = load_spacy_model() if args.use_spacy else None

    result = summarize(
        text=text,
        n_sentences=args.sentences,
        use_stemming=args.stemming,
        use_spacy=spacy_model,
        length_norm=not args.disable_length_normalization,
        similarity_threshold=args.similarity_threshold,
        use_ner_boost=False
    )

    print("\n=== Summary ===")
    print(result["summary"] or "(no sentences selected)")

    if args.show_details and result["selected_indices"]:
        print("\n--- Selected Sentences & Scores ---")
        for i in result["selected_indices"]:
            score = result["scores"][i]
            sentence = result["all_sentences"][i]
            print(f"  [{i}] (score={score:.4f}) {sentence}")

    if args.show_eval:
        metrics = evaluate(result)
        print_evaluation(metrics)


if __name__ == "__main__":
    main()
