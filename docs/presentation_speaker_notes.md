# 12-minute team presentation and live-demo rehearsal

*Suggested division — confirm with the three students before submitting.*

| Slide | Speaker | Approx. time | Focus |
|---|---|---:|---|
| 1–3 | Ibrahim | 4:00 | Motivation, research question, data source, cleaning, splitting, no leakage |
| 4–6 | Abdulaziz & Mohammed | 4:00 | Data flow, traditional model settings, pretrained model settings, unbiased result table |
| 7–9 | Entire group | 4:00 | Three authentic errors, live Try it demo, limitations and conclusion |

Ensure **each student speaks** and can explain their model code, one result, one error case, and an individual contribution. The instructor may give around two minutes of individual questions after the group presentation.

## Reliable live demo

1. Open the folder containing `News_Topic_Classification.ipynb` in Jupyter.
2. Before the talk, install requirements and execute **Kernel → Restart & Run All** to confirm there are no exceptions and output cells remain visible.
3. During the talk, go to **Try it**, enter a new English headline (for instance "Local football team wins regional championship"), run only that cell and identify its predicted category.
4. Demonstrate empty-input warning by setting the headline to spaces.
5. If you choose to demo MiniLM, **pre-download all-MiniLM-L6-v2 weights** on the same device beforehand, and verify the optional path before the presentation.
6. Make sure all shared accuracy/F1 values match `results/metrics.csv` and the report.

## Possible individual questions

- Why **macro F1**? It gives all four categories equal weight instead of letting frequent categories dominate.
- Why **validation**? To choose `C` and TF-IDF ngram settings without peeking at the held-out test.
- Why one model did better? TF-IDF may exploit strong headline keywords and is competitive with a frozen general embedding on this historical dataset. This is an informed hypothesis, not proof.
- Why do some obvious "Sports" headlines count wrong? Some AG News reference labels are ambiguous or inconsistent with the headline's apparent content; we report agreement with the reference labels, not absolute semantic truth.
- What is an **embedding**? A 384-dimensional vector encoding a text representation from the pretrained MiniLM encoder.
- What protects against leakage? All three data splits have no normalized identical headlines in common, and neither model tunes on test.

## Submission warnings

Do not describe AI-assisted examples as uniquely human-written without students personally reviewing them. Credit reused datasets, pretrained weights, package code and AI assistance under course rules. Ensure individually documented work is correct rather than assuming comments alone are sufficient.
