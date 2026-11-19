# TFIDF — Report-ready contribution (ARTI-404, Group 5)

## Model development and training

Our traditional baseline represents each English news **headline** using term frequency–inverse document frequency (TF-IDF), followed by a one-vs-rest Logistic Regression classifier. We used the team-approved AG News headline dataset prepared by Ibrahim. After preprocessing and duplicate handling, the shared splits contained **101,043 training**, **11,227 validation**, and **7,464 test** examples with four classes: World, Sports, Business, and Sci/Tech. We used the `text` field alone and preserved the original split assignments.

The feature extractor used lowercasing, Unicode accent stripping, sublinear term frequencies, and a maximum of 40,000 features. A scikit-learn `Pipeline` ensures TF-IDF fitting happens on training texts only. The classifier used one-vs-rest Logistic Regression with the `liblinear` solver (`max_iter=300`, `tol=1e-3`, `random_state=42`). We compared unigram and unigram–bigram features, with regularization strengths `C` = 0.5, 2.0, and 8.0. **Validation macro F1**, not test performance, determined the chosen configuration: **unigrams, C=8.0**. The chosen model was fitted on the training split and saved with the TF-IDF vectorizer in one joblib pipeline.

## Evaluation

We evaluated the frozen model on the held-out **7,464** test headlines, of which **3,554** contained six words or fewer. Test results were: **accuracy 0.8829 (88.29%)** and **macro F1 0.8823 (88.23%)** for all headlines. On the short-headline subset, accuracy was **0.8672 (86.72%)** and macro F1 **0.8656 (86.56%)**. For reference, validation accuracy was **87.85%** and validation macro F1 was **87.77%**. These are separate from the final test results.

## Error observations (examples from the final test)

1. *“Loosing the War on Terrorism”*: labelled **Sci/Tech**, predicted **World**. The title uses political vocabulary without explicit technology cues. This may be an ambiguous or context-dependent example.
2. *“Card fraud unit nets 36,000 cards”*: labelled **Sci/Tech**, predicted **Business**. Words such as “card” and “fraud” can point to financial news, while a technology/security angle is not explicit in the headline.
3. *“Super ant colony hits Australia”*: labelled **Sci/Tech**, predicted **World**. A very short headline does not provide enough topic context to distinguish a science story from a general world-news event.

These are interpretations of headlines, **not claims about the full articles**. The final team analysis must compare these with the pretrained model’s errors on the **same test rows** before answering the project research question.

## Limitations and next steps

The TF-IDF approach is based on surface word features and can struggle with unusual vocabulary, overlapping topics, and headlines that give little context. The model was not tuned using the test split. A comparison with the pretrained sentence-embedding method is still pending; neither method should be declared the winner before those results are available. The text-input prediction function and saved model allow new headlines to be classified without retraining.

## References

- [AG News dataset](https://huggingface.co/datasets/sh0416/ag_news)
- [scikit-learn TF-IDF vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [scikit-learn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)

*Credit libraries, reused code, models and AI assistance in the final team report, as required by the assignment. Review and rewrite this section to accurately reflect what you personally understand and carried out.*
