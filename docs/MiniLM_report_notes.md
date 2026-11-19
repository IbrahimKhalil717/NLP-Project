# MiniLM — Supporting Report Notes (Integrated Comparison)
**Mohammed Alshahrani  (ID: 2230007324)**  —  ARTI-404 Group 5
*News Topic Classification for Short News Headlines*

> This is a model-specific supporting draft retained for transparency. The up-to-date joint comparison and fully integrated report are in `deliverables/News_Topic_Classification_Report_DRAFT.docx` and `docs/error_analysis.md`.

---

## III. Related Works

AG News has been used as a test bed for text classification for years now, and most of the work around it gave us a clear starting point for the two methods we wanted to compare.

Zhang, Zhao, and LeCun [1] were the ones who first put together the four-way AG News split as part of their work on character-level convolutional networks, and it has been a standard short-text benchmark ever since. The setup they used — 120,000 training headlines plus short descriptions and 7,600 test items split evenly across *World*, *Sports*, *Business*, and *Sci/Tech* — is the same one we use here. We stick with the same labels and the same macro-averaged scoring, so our numbers sit on the same scale as the ones people usually report.

For the bag-of-words side, a simple linear model on top of TF-IDF features is still a very solid baseline on AG News, and most neural results get compared against it. That is why we picked it for Model 1 (TF-IDF + Logistic Regression, implemented in `traditional_model.py`). It gives us a fair point of reference to measure the pretrained embedding model against.

On the pretrained side, Reimers and Gurevych [2] introduced Sentence-BERT, a Siamese BERT setup that is trained so sentence vectors can be compared directly with cosine similarity. The family of models that came out of that work — in particular the small *all-MiniLM-L6-v2* checkpoint we use — gives us 384-dimensional embeddings that transfer pretty well to downstream classification with no task-specific fine-tuning. That is exactly what we needed here: we turn each headline into a fixed vector once, then train a cheap Logistic Regression on top. MiniLM-style models have been shown to keep most of the quality of bigger BERT-base encoders while being around five times smaller and a few times faster, which makes them a realistic pick for a headline classifier that has to run inside a single notebook session.

The reason the short-headline angle is interesting at all is that most AG News work reports one single accuracy number over the whole test set. Headlines in the headline-only field, though, are often very short (we define *very short* as six words or fewer, the way the proposal does), and a six-word phrase like "Sales slow at major retailers" does not really give a bag-of-words model enough to go on. The whole point of trying pretrained sentence embeddings here is that the knowledge sitting inside the encoder should help the most in exactly the spots where the bag-of-words view runs out of signal. That is the gap our paired comparison is trying to measure, and it is why we also report numbers for the short-headline subset of the test split on its own.

---

## IV. Method (Model 2 — Pretrained embeddings + Logistic Regression)

**Representation.** Every headline goes through `sentence-transformers/all-MiniLM-L6-v2`, which hands back a 384-dimensional embedding. We call `encode(..., normalize_embeddings=True)`, so each vector ends up with unit L2 norm. That puts all the examples on the unit sphere and makes Logistic Regression's decision boundaries behave the same way as a cosine-margin classifier. MiniLM is used as a frozen feature extractor — we do not fine-tune it — so the whole pipeline fits in a notebook without too much fuss.

**Classifier.** On top of those embeddings we train a multinomial Logistic Regression (`solver=lbfgs`, `max_iter=2000`, `random_state=42`). The inverse regularisation strength `C` is picked on the validation split only, never on test, over the grid `{0.5, 1, 2, 4, 8}`, going with the value that gives the best macro-F1. The best one turned out to be **C = 0.5**, which is the strongest regularisation in the grid. That makes sense once you remember the 384-d embedding space is already tight and dense, so a stronger prior helps generalisation.

**Inputs.** We feed the model the approved prepared splits (`train.csv`, `val.csv`, `test.csv`). Each row has `text`, `label`, `word_count`, and an `is_short` flag (True when `word_count ≤ 6`). After his cleaning and de-duplication the sizes work out to train = 101,043 · val = 11,227 · test = 7,464, with short-headline subsets of 47,011 / 5,198 / 3,554 respectively.

**Light pre-processing.** The encoder handles casing and punctuation on its own, so we only strip leftover HTML tags (`<b>`, `&lt;`, and the like that are still sitting in AG News) and squash whitespace before encoding. We do not lowercase or stem anything.

**Hand-off.** The script writes:
- `models/pretrained_lr.joblib` — the fitted classifier,
- `results/predictions_B.csv` — row-aligned with `test.csv` so the shared evaluation code can line it up directly,
- `results/predictions_B_val.csv` — same thing for the validation split,
- `results/metrics_B.json` — accuracy and macro-F1 on val, full test, and the short-only subset of test,
- `results/try_it_examples_B.csv` — two draft qualitative examples (not from the test set; each student must personally review and label their own originals) run through the frozen pipeline.

---

## V. Results (Model 2)

**Headline numbers.** On the full test split, the pretrained embedding model gets to **accuracy 0.8382** and **macro-F1 0.8375**. On the short-headline subset (`is_short == True`, n = 3,554) it drops to **accuracy 0.8106** and **macro-F1 0.8095**. On validation at the chosen `C = 0.5` it lands at accuracy 0.8391 and macro-F1 0.8378 — basically the same as test, which is a good sign that the hyper-parameter choice did not overfit.

| Split              |     n | Accuracy | Macro-F1 |
|--------------------|------:|---------:|---------:|
| Validation         | 11,227 |  0.8391  |  0.8378 |
| Test (all)         |  7,464 |  0.8382  |  0.8375 |
| Test (short ≤ 6 w) |  3,554 |  0.8106  |  0.8095 |

The gap between full test and short-only is **−2.76 macro-F1 points** (0.8375 → 0.8095), so the pretrained model does take a hit on very short headlines, but it does not fall apart — it holds on to around 97% of its full-test macro-F1 on the hardest subset.

**Per-class behaviour (full test).** Reading off the confusion matrix (order: World, Sports, Business, Sci/Tech):

```
             pred: World  Sports  Business  Sci/Tech   support
True World:        1608      97      110        67       1882
True Sports:         71    1701       71        43       1886
True Business:       92      88     1429       235       1844
True Sci/Tech:       85      51      198      1518       1852
```

- **Sports** is the easiest class to catch (recall 90.2%). Its vocabulary — team names, "wins", "beat", score lines — is pretty well separated from the other three.
- The biggest confusion by far is **Business ↔ Sci/Tech**: 235 *Business* headlines come back as *Sci/Tech*, and 198 *Sci/Tech* come back as *Business*. That one is kind of expected — a lot of headlines about tech company earnings, launches, or deals genuinely sit on the line between the two classes.
- **World ↔ Business** is the second biggest confusion (110 + 92), which lines up with the fact that headlines about trade, markets, or world finance are honestly dual-topic a lot of the time.
- **Sci/Tech** ends up with the lowest per-class recall (82.0%), mostly because of the Business confusion above.

**Per-class behaviour (short test).** The same story shows up on short headlines — every class loses a few points of recall, and the Business ↔ Sci/Tech confusion stays the main thing going wrong (134 Business → Sci/Tech, 110 Sci/Tech → Business). Short *Sci/Tech* headlines are the single hardest cell: they are often one- or two-word product or company names that look a lot like *Business* headlines on the surface.

**Comparison to TF-IDF + LR.** On the full test split, MiniLM macro F1 is **0.8375** versus TF-IDF **0.8823** (TF-IDF higher by **0.0448**). On the very-short subset, MiniLM is **0.8095** versus TF-IDF **0.8656** (TF-IDF higher by **0.0561**). These results answer the approved question for the two frozen models. See `results/metrics.csv`.

**Try-it examples.** The two draft qualitative headlines, subject to Mohammed’s personal review are not from AG News. Both get classified by the frozen pipeline with no retraining, and the function throws a readable error on empty input instead of crashing. Actual predictions and confidences live in `results/try_it_examples_B.csv`.

---

## VII. Conclusion

We went in with one clear question: on short English news headlines of six words or fewer, does classic TF-IDF + Logistic Regression do better, or do pretrained sentence embeddings + Logistic Regression do better? We worked on the AG News headline field, carved out a validation split for picking hyper-parameters, kept the original test split aside for the final comparison, trained both methods under matched conditions, and scored them with accuracy and macro-F1 on the same examples.

The pretrained *all-MiniLM-L6-v2* representation ends up at macro-F1 **0.838** on the full test split and **0.810** on the short-headline subset — a drop of only about 2.8 F1 points once the context shrinks to six words or fewer. The TF-IDF baseline is higher by **0.0448** macro F1 overall and **0.0561** on the short-headline subset. The joint error analysis is provided in `docs/error_analysis.md`. The error analysis there also shows that most of what the pretrained model still gets wrong sits on the *Business ↔ Sci/Tech* boundary, which is kind of a fuzzy line to begin with for news about tech companies and their money moves — probably no model working off the headline alone is going to clean that one up perfectly.

**Limitations.** We used MiniLM as a frozen feature extractor. Task-specific fine-tuning would almost certainly push it further, but it would also make the pipeline a lot heavier to train. Our cut-off for "very short" (≤ 6 words) is a project choice, not a property of AG News itself. There are also dataset-specific things to keep in mind — for instance, AG News leans heavily on U.S. English and pre-2015 news content — which means the numbers here should be read as a within-split comparison, not a general claim about headline classification everywhere. One more thing: we paired both models with the same Logistic Regression head on purpose, to isolate the effect of the representation; a stronger non-linear classifier on top of either one could easily change the ranking.

**Future work.** The cleanest next step would be to put a fine-tuned small transformer up against the frozen-MiniLM baseline on the exact same splits, just to see how much of the leftover short-headline error is really about missing task adaptation and how much is just genuinely ambiguous input. Another useful direction is to grow the short-headline subset on purpose by paraphrasing: if you take long headlines and cut them down (or rewrite them) and you get the same pattern of errors as the naturally short ones, that would back up the idea that it is *length itself* driving the drop, not some side effect of how short headlines are chosen.

---

## References

[1] X. Zhang, J. Zhao, and Y. LeCun, "Character-level Convolutional Networks for Text Classification," *NeurIPS*, 2015. https://arxiv.org/abs/1509.01626

[2] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," *EMNLP-IJCNLP*, 2019, pp. 3982–3992. https://aclanthology.org/D19-1410/

[3] W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and M. Zhou, "MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers," *NeurIPS*, 2020. https://arxiv.org/abs/2002.10957
