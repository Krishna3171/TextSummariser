# TF-IDF Extractive Text Summarizer

This project implements an **extractive text summarizer** using **TF-IDF sentence scoring** in pure Python.

## How it works

1. Split input text into sentences.
2. Tokenize each sentence and remove common stopwords.
3. Compute IDF for each token across all sentences.
4. Compute each sentence score as the sum of token TF-IDF values.
5. Select top-N scoring sentences and keep their original order.

## Run

```bash
python tfidf_summarizer.py --text "Natural language processing helps computers understand text. Summarization reduces long text into concise information. Extractive methods choose important sentences directly. TF-IDF highlights informative terms. This makes summaries simple and effective." --sentences 2 --show-scores
```

Or summarize from a file:

```bash
python tfidf_summarizer.py --file sample.txt --sentences 3
```

## Arguments

- `--text`: input text string
- `--file`: path to input text file
- `--sentences`: number of summary sentences (default: 3)
- `--show-scores`: print selected sentence scores
