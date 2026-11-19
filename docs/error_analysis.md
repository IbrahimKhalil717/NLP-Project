# Shared error analysis — actual test headlines

**Evaluation rule:** An "error" means the model disagrees with the AG News **reference label**, not necessarily that a human judge would agree with that reference. All examples below are from the untouched test split; they are not used to tune models.

| Short headline | Dataset label | TF-IDF prediction | MiniLM prediction | Plausible interpretation |
|---|---|---|---|---|
| `Live: Olympics day four` | World (1) | Sports (2) | Sports (2) | Both models follow the obvious Olympics sports cue, whereas the reference label says World. This is a likely boundary or annotation ambiguity, not strong evidence of poor semantic understanding. |
| `Google Lowers Its IPO Price Range` | World (1) | Business (3) | Business (3) | The wording suggests a financial offering, so both models select Business. The mismatch points to overlapping topic definitions or reference noise. |
| `Small computers can have multiple personalities, too` | Sci/Tech (4) | Business (3) | Sci/Tech (4) | Sparse word cues may be weak or ambiguous for the traditional classifier; the embedding model gets the reference label right on this short-context example. This is a hypothesis, not proof of the model's decision process. |

**Answer to approved research question:** TF-IDF + Logistic Regression achieved macro F1 **0.8656** on the 3,554 very-short test headlines; pretrained MiniLM embeddings + Logistic Regression achieved **0.8095** on the *same* cases. In this experiment, TF-IDF is higher by **0.0561 absolute macro-F1**, approximately **5.61 percentage points**. Both methods perform worse on very-short than on all headlines.

**Other patterns:** 584 examples are wrong for both models, 290 only for TF-IDF, and 624 only for MiniLM (out of 7,464 test headlines). Exact source row details are saved in `results/errors_to_analyze.csv`, and the four class-specific confusion matrices are under `results/`.

**Limitations:** Historical headlines, imperfect reference categories, and context omitted from the headline may constrain results. No test examples were used for model selection. An alternative sentence embedding model, domain-specific tuning, or careful review of ambiguous labels is reasonable future work; it was not tested in this study.
