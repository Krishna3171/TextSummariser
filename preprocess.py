"""
preprocess.py
-------------
Sentence splitting and token-level preprocessing pipeline.
Supports NLTK (default) and optional spaCy backend.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required NLTK data on first use
_NLTK_PACKAGES = {
    "punkt":     "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet":   "corpora/wordnet",
}
for _pkg, _path in _NLTK_PACKAGES.items():
    try:
        nltk.data.find(_path)
    except LookupError:
        try:
            nltk.download(_pkg, quiet=True)
        except Exception:
            pass  # network unavailable — will surface as a runtime error if actually needed


def split_sentences(text: str) -> list[str]:
    """Return a list of raw sentences from *text*."""
    return sent_tokenize(text.strip())


def preprocess_tokens(
    sentence: str,
    use_stemming: bool = False,
    use_spacy=None,
) -> list[str]:
    """
    Tokenise and clean a single sentence.

    Parameters
    ----------
    sentence   : raw sentence string
    use_stemming: use PorterStemmer instead of WordNetLemmatizer
    use_spacy  : a loaded spaCy language model, or None to use NLTK

    Returns
    -------
    List of lowercase, stemmed/lemmatised, non-stop tokens.
    """
    stop_words = set(stopwords.words("english"))

    if use_spacy is not None:
        doc = use_spacy(sentence)
        tokens = [
            tok.lemma_.lower()
            for tok in doc
            if tok.is_alpha and tok.lemma_.lower() not in stop_words
        ]
        return tokens

    # NLTK path
    raw_tokens = word_tokenize(sentence.lower())
    tokens = [t for t in raw_tokens if re.fullmatch(r"[a-z]+", t) and t not in stop_words]

    if use_stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    else:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens


def preprocess_document(
    text: str,
    use_stemming: bool = False,
    use_spacy=None,
) -> tuple[list[str], list[list[str]]]:
    """
    Split *text* into sentences and preprocess each.

    Returns
    -------
    sentences      : list of original sentence strings
    token_lists    : list of token lists (one per sentence)
    """
    sentences = split_sentences(text)
    token_lists = [
        preprocess_tokens(s, use_stemming=use_stemming, use_spacy=use_spacy)
        for s in sentences
    ]
    return sentences, token_lists
