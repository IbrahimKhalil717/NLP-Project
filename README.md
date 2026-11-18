# News Topic Classification — Group 5

## Member 1: Ibrahim Aljurays
**Course:** ARTI-404  
**Project:** News Topic Classification for Short News Headlines

## Project Overview
This project classifies English news headlines into four categories:
1. World
2. Sports
3. Business
4. Science and Technology

We compare two NLP approaches:
- TF-IDF + Logistic Regression
- all-MiniLM-L6-v2 + Logistic Regression

Our research question is:
Which method performs better on headlines containing six words or fewer?

## Member 1 Contributions
- Downloaded and prepared the AG News dataset.
- Extracted news headlines for classification.
- Cleaned text and handled duplicate headlines.
- Created training, validation, and testing splits.
- Identified short headlines (6 words or fewer).
- Verified no exact headline overlap between splits.
- Prepared evaluation and error analysis code.

## Dataset Statistics

| Split | Total | Short Headlines |
|---|---:|---:|
| Train | 101,043 | 47,011 |
| Validation | 11,227 | 5,198 |
| Test | 7,464 | 3,554 |

The original AG News dataset contains 120,000 training
and 7,600 testing examples. The prepared dataset has fewer
rows due to cleaning, deduplication, and overlap handling.

## Dataset Columns
- text: News headline
- label: Category ID (1 to 4)
- word_count: Number of words
- is_short: True if word_count <= 6

## Installation
```bash
pip install -r requirements.txt
```

## Data Preparation
```bash
python preprocessing.py
```

Generated files:
- data/train.csv
- data/val.csv
- data/test.csv
- data/preprocessing_stats.json

Both models must use the same prepared datasets.
The test split must not be used for training or tuning.

## Evaluation
Once both models have generated predictions, provide:

- results/tfidf_predictions.csv
- results/minilm_predictions.csv

Each file must contain a prediction column with integer
labels 1 to 4 and a text column matching test.csv order.

Run:
```bash
python evaluation_member1.py
```

The evaluation script calculates:
- Accuracy
- Macro F1-score
- Short-headline performance
- Confusion matrices
- Classification reports
- Incorrect predictions for error analysis

## Current Status
Data preparation and overlap verification are complete.
Evaluation code is prepared, but final model results are
pending the two teammates' predictions.

## Dataset Source
https://huggingface.co/datasets/sh0416/ag_news

## Acknowledgment
AI assistance was used to support code development,
documentation, and debugging. The team is responsible
for reviewing, executing, and explaining the work.
