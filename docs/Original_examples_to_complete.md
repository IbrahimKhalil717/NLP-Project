# TFIDF — Two Short News Headline Examples

These two example headlines are AI-assisted drafts for Abdulaziz to review, personalize, and approve before final submission. Neither is copied from the supplied AG News train, validation, or test CSV files (exact normalized-text comparison).

| News headline | Word count | Expected label | TFIDF model prediction | Correct? |
|---|---:|---|---|---|
| Underdog team clinches regional football title | 6 | 2 — Sports | 2 — Sports | Yes |
| Scientists unveil faster battery charging method | 6 | 4 — Sci/Tech | 4 — Sci/Tech | Yes |

Class mapping: 1 = World, 2 = Sports, 3 = Business, 4 = Sci/Tech.

The predictions above were obtained by loading `outputs/tfidf/tfidf_logreg.joblib` and calling the saved model's `predict()` method on these two headlines, with no retraining. Both examples meet the team's very-short-headline condition (6 words or fewer). They are qualitative examples and are not included in official test accuracy or macro F1.

Before submission, Abdulaziz should review and if needed rewrite these headlines, personally confirm the expected labels, and disclose AI assistance in accordance with course requirements. If a headline is changed, rerun prediction and update the prediction column.
