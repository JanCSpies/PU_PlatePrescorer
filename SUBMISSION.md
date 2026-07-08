# Submission summary (~180 words)

**Predicting Buchwald–Hartwig C–N coupling feasibility from literature data alone.**

Exploratory high-throughput and direct-to-biology campaigns waste plate capacity on reactions that
simply fail. We ask whether a model trained *only* on biased, positive-heavy literature data can
pre-score a substrate matrix to design higher-hit-rate plates — framed as Positive-Unlabeled (PU)
learning and evaluated on a held-out J&J HTE plate (Neves 2024), under a frozen, substrate-disjoint,
anti-leakage protocol (no test labels ever seen in training).

Naive literature→test transfer is random (ROC-AUC ≈0.52): a PU classifier learns "was this reaction
*published*?" (domain-discriminator AUC = 1.00), not "does it work?". We fix this with a
**covariate-shift, importance-weighted PU model** ("a better PAYN") and a **35,929-pair C–N corpus
mined from the Open Reaction Database** (provably free of the test set). Result: **ROC-AUC 0.72**,
beating the spy-PU/SCAR baseline by ~0.15, and a **2.3× plate hit-rate uplift**. Honestly, the gain
is transductive (ranking a *known* substrate library — the plate-design goal); generalizing to novel
substrates, and site-specific QM features, remain open. Fully open-sourced.
