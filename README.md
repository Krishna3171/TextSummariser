# TF-IDF Extractive Text Summarizer

This project implements a modular extractive summarizer with:

1. Preprocessing with stopword removal and stemming/lemmatization.
2. Sentence length normalization during TF-IDF scoring.
3. Cosine similarity filtering to reduce redundant output sentences.

## Project Structure

```
.
├── preprocess.py
├── tfidf_ranker.py
├── summarizer.py
├── evaluation.py
├── main.py
├── datasets/
│   └── sample.txt
└── README.md
```

## Features

### 1) Preprocessing

- Stopword removal using NLTK stopwords.
- Lemmatization with NLTK WordNet lemmatizer by default.
- Optional stemming with Porter stemmer.
- Optional spaCy pipeline support (`--use-spacy`) when installed.

### 2) Sentence Length Normalization

- Sentence score uses TF-IDF with normalization by $\sqrt{L}$, where $L$ is sentence token length.
- This prevents long sentences from dominating purely due to more terms.

### 3) Cosine Similarity Filtering

- After ranking, candidate sentences are filtered by cosine similarity.
- Highly similar sentences (above threshold) are skipped to reduce redundancy.

### 4) Keyword Boosting (with NER)
- Supports keyword boosting using Named Entity Recognition (NER).
- To enable this feature:
- Set use_ner_boost = True in the configuration/code.
- Run the CLI with --use-spacy to activate spaCy-based NER.

## Dependencies

Install required libraries:

```bash
pip install nltk spacy
```

Optional spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

## Run

Using sample file:

```bash
python main.py --file datasets/sample.txt --sentences 3 --show-details --show-eval
```

Using direct text:

```bash
python main.py --text "Natural language processing helps computers understand text. Summarization reduces long text into concise information. Extractive methods choose important sentences directly. TF-IDF highlights informative terms. This makes summaries simple and effective." --sentences 2
```

## CLI Options

- `--text`: input text to summarize
- `--file`: path to input file (default: `datasets/sample.txt`)
- `--sentences`: number of output sentences
- `--use-spacy`: use spaCy tokenizer/lemmatizer when available
- `--stemming`: use stemming instead of lemmatization
- `--disable-length-normalization`: turn off sentence length normalization
- `--similarity-threshold`: cosine threshold for redundancy filtering
- `--show-details`: show selected sentence scores
- `--show-eval`: print basic summary metrics
