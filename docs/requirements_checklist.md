# Course requirement audit — Group 5

Checked against the NLP Project Student Brief (pages 3–7) and the approved proposal. Results are generated from the *provided* frozen models and original data; no test tuning was performed in the integration work.

## Verified through code or included artifacts

- [x] AG News public labelled dataset, source, class labels and split counts documented.
- [x] Separate train, validation and test headline rows, with no normalized overlaps in supplied splits.
- [x] Traditional TF-IDF + Logistic Regression and pretrained MiniLM embeddings + Logistic Regression implemented.
- [x] Frozen candidate model settings selected on validation, not test.
- [x] Accuracy and macro F1 reported on *identical test examples*, plus very-short (≤6 words) subset.
- [x] Two frozen predictions files with matching row order, text and 1–4 label scheme.
- [x] Comparison evaluation runs successfully, produces classification reports and confusion matrices.
- [x] At least 3 real mistakes described in `docs/error_analysis.md`, including reference-label ambiguity.
- [x] A main `News_Topic_Classification.ipynb` with Setup, Data, Method 1, Method 2, Results, Try it, Sources.
- [x] `Restart & Run All` executed successfully on provided artifact files and local environment.
- [x] New-text prediction without retraining, empty-input guard verified for TF-IDF; MiniLM optional requires pretrained download.
- [x] Source packages, environment, references and credit for libraries/weights/AI assistance documented.
- [x] Final English report contains the seven required sections and is within the eight-page limit.
- [x] Final PowerPoint presentation contains nine slides.
- [x] Team contribution appendix completed and approved by all three members.
- [x] Final documents are included in `deliverables/`.

## Team must still complete/verify before Blackboard submission

- [ ] Team reviews speaking roles and practices **12-minute presentation + live demonstration** with each person speaking.
- [ ] Team runs `Run All` on their own intended Colab/Jupyter environment and verifies **MiniLM live mode** can download/load its pretrained weights if they plan to demonstrate it. The default TF-IDF Try it does work offline.
- [ ] Team checks its GitHub repository includes `News_Topic_Classification.ipynb`, source scripts, `models/minilm_logreg_coefficients.npz`, frozen TF-IDF model and canonical `results/*.csv`, and verifies the tracked `data/*.csv` against the split manifest.
- [ ] Team bundles final report (Word/PDF), final appendix, presentation, notebook, supporting code and GitHub link in one `Group5_Project.zip` on Blackboard before precise posted deadline.
- [ ] Team confirms exact due time and presentation slot on Blackboard (course guideline only gives date ranges).

**Important:** Included `models/pretrained_lr.joblib` was originally saved with scikit-learn 1.9.1 while the TF-IDF model was saved with 1.8.0. The project uses the exported NumPy weights for MiniLM inference to avoid cross-version pickle loading. A complete *from-scratch MiniLM training* run requires downloading pretrained all-MiniLM-L6-v2; this was not repeated inside this local environment. Frozen original model results were validated thoroughly.
