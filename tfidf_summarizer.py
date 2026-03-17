import argparse
import math
import re
from collections import Counter, defaultdict
from typing import List, Tuple


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will",
    "with", "this", "these", "those", "or", "but", "if", "then", "than", "so", "such",
    "into", "about", "over", "after", "before", "under", "between", "while", "during",
    "do", "does", "did", "doing", "have", "had", "having", "i", "you", "we", "they",
    "them", "their", "our", "your", "my", "me", "him", "her", "she", "who", "whom",
    "which", "what", "when", "where", "why", "how", "can", "could", "should", "would",
    "may", "might", "must", "not", "no", "yes", "also", "very", "just", "only", "up",
    "down", "out", "off", "again", "once"
}


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def compute_idf(tokenized_sentences: List[List[str]]) -> dict:
    num_sentences = len(tokenized_sentences)
    doc_freq = defaultdict(int)

    for sent_tokens in tokenized_sentences:
        for token in set(sent_tokens):
            doc_freq[token] += 1

    # Smooth IDF to avoid division by zero and reduce extreme weights.
    return {
        token: math.log((1 + num_sentences) / (1 + freq)) + 1
        for token, freq in doc_freq.items()
    }


def sentence_tfidf_score(sent_tokens: List[str], idf: dict) -> float:
    if not sent_tokens:
        return 0.0

    tf = Counter(sent_tokens)
    length = len(sent_tokens)
    score = 0.0

    for token, count in tf.items():
        term_tf = count / length
        score += term_tf * idf.get(token, 0.0)

    return score


def summarize(text: str, num_sentences: int = 3) -> Tuple[str, List[Tuple[int, float, str]]]:
    sentences = split_sentences(text)
    if not sentences:
        return "", []

    tokenized = [tokenize(sentence) for sentence in sentences]
    idf = compute_idf(tokenized)

    scored = []
    for idx, sent_tokens in enumerate(tokenized):
        score = sentence_tfidf_score(sent_tokens, idf)
        scored.append((idx, score, sentences[idx]))

    num_sentences = max(1, min(num_sentences, len(sentences)))

    top = sorted(scored, key=lambda x: x[1], reverse=True)[:num_sentences]
    top_sorted_by_position = sorted(top, key=lambda x: x[0])

    summary = " ".join(sentence for _, _, sentence in top_sorted_by_position)
    return summary, top_sorted_by_position


def main() -> None:
    parser = argparse.ArgumentParser(description="Extractive text summarizer using TF-IDF sentence scoring")
    parser.add_argument("--text", type=str, help="Input text to summarize")
    parser.add_argument("--file", type=str, default="D:\\computer science\\NLP\\TextSummariser\\sample.txt", help="Path to a text file to summarize")
    parser.add_argument("--sentences", type=int, default=3, help="Number of sentences in summary")
    parser.add_argument("--show-scores", action="store_true", help="Print top selected sentence scores")
    args = parser.parse_args()

    if not args.text and not args.file:
        parser.error("Provide either --text or --file")

    if args.text:
        source_text = args.text
    else:
        with open(args.file, "r", encoding="utf-8") as f:
            source_text = f.read()

    summary, selected = summarize(source_text, num_sentences=args.sentences)

    print("\nSUMMARY:\n")
    print(summary)

    if args.show_scores:
        print("\nSELECTED SENTENCES (index, score):\n")
        for idx, score, sentence in selected:
            print(f"[{idx}] {score:.4f} -> {sentence}")


if __name__ == "__main__":
    main()
