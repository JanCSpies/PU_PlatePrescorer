> **Provenance.** Agent summary, **not yet checked by Luka**. The PDF was read in the working session of 2026-10-29 from the open-access copy; Luka supplied the file but has not read it himself. Page locators come from that reading, and where none was taken they are `[n]`. Check a detail against the paper before citing it.
>
> **Reference.** Lieyang Chen, Anthony Cruz, Steven Ramsey, Callum J. Dickson, Jose S. Duca, Viktor Hornak, David R. Koes, Tom Kurtzman. *Hidden bias in the DUD-E dataset leads to misleading performance of deep learning in structure-based virtual screening*. PLoS ONE 2019, 14(8): e0220113. DOI: 10.1371/journal.pone.0220113. Licence: open access. Date read: 2026-10-29.

# Chen et al. 2019: hidden bias in DUD-E

## What they did

They ask whether deep-learning scores on DUD-E come from the receptor-ligand interaction or from how the benchmark was built.

- **Actives-as-Decoys (AAD) control**, built on p. 3, result on p. 11: the decoys are rebuilt without the suspect step, so the construction can no longer be what the model reads.
- A ligand-only model matched a receptor-ligand model to within 0.001 AUC.
- A 2-D fingerprint K-NN reached AUC above 0.82 on all 102 targets, with K = 1 best on 79.

## What the data is

DUD-E: actives and property-matched decoys for 102 targets. Their decoys were matched to the actives on one-dimensional properties but differ from them topologically. Sizes: `[n]`.

## What carries over to this project

- The worry: a model may read our construction of U and not the chemistry. Their remedy, a control that rebuilds the negatives without the suspect step, became the anchor ablation (journal 2026-10-29): a second rule, `size_stratified_both_sides`, replacing both sides of a pair.
- K-NN on a 2-D fingerprint as a probe, and K = 1 winning as the signature of two separable clusters. In our ablation K = 1 loses everywhere, so the signature is absent.

## What does not carry over

- Their decoys came from a different source than the actives. Our replacements are real published substrates. Our construction is closer to their control than to their failure.
- Their task is receptor-ligand; ours is a pair of reaction partners. Matching on one-dimensional properties is their bias; ours is matched marginals per role (ADR 013, ADR 015).
- The control finding nothing here does not prove our sampler is clean. It says these probes did not find a leak.

## Numbers we may cite

- Ligand-only against receptor-ligand model: within 0.001 AUC. Page `[n]`.
- 2-D fingerprint K-NN: AUC above 0.82 on all 102 targets; K = 1 best on 79. Page `[n]`.
- AAD control: built p. 3, result p. 11.
