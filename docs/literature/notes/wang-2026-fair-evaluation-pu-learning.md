> **Provenance.** Agent summary, not yet checked by Luka. Written 2026-10-28 from the PDF, which was read in that session. Locators are pages given by Luka's session (27 pages in the arXiv PDF); where none was given they are `[n]`. Read the paper before citing a detail.
>
> **Reference.** Wei Wang, Dong-Dong Wu, Ming Li, Jingxiong Zhang, Gang Niu, Masashi Sugiyama. *Accessible, Realistic, and Fair Evaluation of Positive-Unlabeled Learning Algorithms*. ICLR 2026. arXiv:2509.24228. DOI: not stated. Licence of data: `[n]`. Date read: 2026-10-28.

# Wang et al. 2026: fair evaluation of PU learning

## What they did

They address how PU algorithms are validated and compared when no negative data exist.

- **Definition 2 (proxy AUC score, PAUC)**, p. 4, and **Proposition 2**, p. 5; the proof is in Appendix B.2 (read through the arXiv HTML, no page). PAUC is the ROC-AUC of ranking the positive validation set above the unlabeled one, ties counted as one half. It restates the quantity in Wei et al. 2022 (Theorem 1), which is the older source.
- **Justification.** The unlabeled validation data are treated "to be corrupted negative data". AUC is robust to corrupted labels (Charoenphakdee, Lee, Sugiyama, ICML 2019).
- **Recommendation:** "We recommend using PA and PAUC in future PU learning experiments, especially in real-world applications where negative data cannot be obtained." p. 5.

## What the data is

Benchmark PU datasets and a set of PU algorithms compared with validation by PAUC and by an oracle with negatives. Names and sizes: `[n]`.

## What carries over to this project

- Confirmation and recommendation for the information axis (ADR 019 items 1 and 2). The paper is cited second, after Wei et al.
- The corrupted-negatives reading is a short way to say why the unlabeled set can stand in for negatives in a ranking statistic.

## What does not carry over

1. The assumptions stay in the algebra and are not stated as limits. Our U does not follow the true marginal (matched on purpose, ADR 013 item 1), and P is not a SCAR sample (reporting bias). See the note on Wei et al.
2. **Table 1 is mixed** (page `[n]`):

   | method | test AUC, selected by PAUC | test AUC, selected by oracle |
   |---|---|---|
   | PUbN | 93.44 | 94.33 |
   | VPU-c | 93.51 | 91.79 |

   PAUC selection beat the oracle for one method and lost for the other.
3. No rank-agreement or correlation statistic between PAUC selection and oracle selection is reported.
4. They select hyperparameters and checkpoints. They do not choose a feature representation. Our use is an extension beyond what they validated.
5. They do not mention Wei et al.'s finding that IPM was more robust than AUC.

## Numbers we may cite

- The recommendation quotation, p. 5. Definition 2, p. 4. Proposition 2, p. 5.
- Table 1 values above, page `[n]`: PUbN 93.44 against 94.33, VPU-c 93.51 against 91.79.
- Nothing else. Dataset sizes and other results: `[n]`.
