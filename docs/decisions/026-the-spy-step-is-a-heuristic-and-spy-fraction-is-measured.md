# 026 — The spy step is a heuristic, because both estimators out of SCAR need the separability I removed

- **Status:** Accepted, 2026-10-30
- **Supersedes:** — · **Superseded by:** 027 in part (item 2)

This record changes no code. It supersedes nothing. ADR 013 item 3 (the spy technique, chosen over
estimating the class prior) stands as written. What changes is the reason it stands, and what the step is
called. It also supplies the citation ADR 019 lacked for `e(x)`, SAR and SNAR.

I asked for the literature to be read before `spy_fraction` was chosen. It did not return a value for it.
It returned two papers that point the same way, and that bears on whether the spy step is the right
instrument at all.

Both notes below are agent summaries I have **not yet checked** against the PDFs. Locators are the notes':
Bekker et al. by printed page, Kumar and Lambert by section heading (the PMC copy has no page numbers).
Every figure taken from a note says so.

## Context

**1. Both principled routes out of SCAR need P and U to be separable.**

- PULSNAR (Kumar and Lambert, *PeerJ Computer Science* 2024, 10, e2451) estimates α, the positive
  fraction **among the unlabeled only**. It is not the class prior; the conversion is
  `p(y=1) = (|P| + α|U|)/(|P| + |U|)` (their Eq. 1). The estimate is `α_max = min(D_u_i / D_p_i)`, the
  largest α that keeps `D_u − α D_p` non-negative (Eq. 2, Algorithm 1). It needs the two score densities
  to be distinguishable, and the paper says the SCAR estimator works "under the assumption of SCAR and that
  the classes are separable" (§ Discussion and conclusion). Note `kumar-2024-pulsnar-class-proportion-without-scar.md`.
- Bekker, Robberechts and Davis (ECML PKDD 2019, arXiv:1809.03207v4) fit a classifier and a propensity
  model `e(x)` jointly by EM. The paper says the method "will work best if the classes are separable"
  (p. 9). Note `bekker-2019-beyond-scar-propensity.md`.

My sampler removes separability on purpose. It matches the per-side marginals of U to P and swaps within
the same heavy-atom decile (ADR 013 item 1, ADR 015). Stage U's gate confirms it worked:

| | rank-AUC of one side's literature frequency |
|---|---|
| acids | 0.5049 |
| amines | 0.5095 |

Command: `uv run python scripts/unlabeled_literature.py amide_coupling ~/University/Phd/Dataset/Datasets/derived --config configs/models/unlabeled_literature_k10.yml`
(journal, stage U entry; also ADR 018 and ADR 020).

So the property that makes U a good unlabeled set is the one that makes both estimators degenerate. For
PULSNAR: if `D_u` and `D_p` nearly coincide, `α_max` is driven toward 1. A near-1 reading cannot be told
apart from "my matching simply worked". The note says this too (what does not carry over, second bullet).

**2. My U is not the object either framework describes.** Bekker et al. work in the single-training-set
scenario (p. 2): one sample of triples `(x, y, s)` exists, and labelling acts on it with per-instance
probability `y_i e_i`. Mine is the reverse. Publication happened in the world; then I built 102,200
unlabeled pairs around the published ones (stage U, same command, `pairs kept` 102,200). My U rows had
probability **zero** of ever being labelled.

The precise version, because it is easy to get wrong in both directions:

- The weighted estimator **is** unbiased for the empirical risk on whatever finite set I hold. The
  derivation (p. 5) fixes `x_i` and `y_i` and averages only over `s_i`. The arithmetic is portable.
- What breaks is the inferential claim. The risk it is unbiased for is the risk under **my sampler's**
  marginal. It is not the plate's marginal and not nature's.
- An EM fitted on my tables would absorb the sampling design (the decile matching, the ratio) and report it
  back as publication propensity.

**3. The mechanism of SCAR-under-SAR failure is the spy threshold's.** Bekker et al., p. 11: SCAR methods
pursue "the correct ratio of the instance space as positive. When it picks the wrong subspace to get to
this ratio, it results in both false positives and false negatives, where the more conservative naive
approach would only give the false negatives." That is a mis-shaped decision region, not a mis-calibration.
A spy threshold is a device for choosing how much of the space to call reliably negative. It could cut the
wrong part **while looking calibrated**. This is the same defect as ADR 014 item 1 and ADR 019 item 3 (the
threshold is set on reported positives, the hidden ones are unreported), with a measured size for it.

Their worst cases, ROC-AUC, SCAR-KM2 against SAR-EM (note, Fig. 1b, p. 12; no command, authors' figures):

| dataset | SCAR-KM2 | SAR-EM |
|---|---|---|
| diabetes | 0.500 | 0.659 |
| 20 Newsgroups | 0.603 | 0.891 |

Friedman test over the eight datasets, p < 0.001 (p. 11). The bias there is **simulated**: four artificial
propensity attributes, `e` between 0.2 and 0.8 (pp. 10–11). None of it is chemistry.

**4. Correcting the bias is often not worth much, and this has equal weight.** In the same paper (note,
Fig. 1b, p. 12):

- The naive approach, U treated as negative, beat SCAR-KM2 on **6 of 8** datasets.
- It beat SAR-EM itself on breast cancer (0.992 against 0.983) and tied it on mushroom (1.000 against 1.000).
- On four of the eight datasets the whole question is worth under 0.05 AUC (the note's reading of the same
  figure; adult +0.016, breast cancer +0.012, splice +0.031 for SAR-EM over KM2). The large wins sit in two
  datasets, diabetes and 20 Newsgroups.

I do not read finding 3 without finding 4.

**5. SCAR-based α estimators underestimate when α is high.** Kumar and Lambert (§ SNAR data, Results): "For
fractions ≥ 15%, KM1, KM2, TIcE, DEDPUL, and PULSCAR usually underestimated α across all datasets." Their
assumption 2 (§ PU data assumptions) says that without support of the unlabeled positives in the labelled
set the methods "may estimate an inaccurate α, often resulting in underestimates". Mine may well be high:

| | value | source |
|---|---|---|
| positive rate, Chemlex **validation** split, one condition | 29.14 % of 2,615 pairs | ADR 008; `docs/design/modeling.md` line 18 |
| PAYN unlabeled sets, three of four | 58–77 % positive | `boser-2026-payn.md`, SI pp. S4–S6 (agent summary) |
| PAYN reliable negatives that are false, one dataset (Stevens) | about 50 % | ADR 024 Context, from the PAYN note (agent summary) |

A SCAR-based device would then think U cleaner than it is, in the direction that makes it look fine.
PULSNAR was validated at true α of 1, 5, 10, 20, 30, 40 and 50 % (note, § Synthetic data), so 0.3 is
inside its tested band. The paper's stated gap is the **low** end, α ≤ 5 % (Introduction). This is not an
argument that PULSNAR would be miscalibrated at 0.3. It is an argument about the SCAR-based device and
about separability, in finding 1.

## Decision

1. **The spy step stays, and is named a heuristic, not an estimator.** ADR 013 item 3 chose it over
   estimating the class prior and that choice stands. ADR 013 item 3 already said π "rests on a separability
   assumption which reporting bias is precisely what casts doubt on" and is not honestly estimable. What this
   record adds is the evidence and the sharper reason: the two methods built for the non-SCAR case both need
   the separability my sampler removes (Context 1), so the estimate is unavailable and not only doubtful.
   In the paper the spy threshold is a rule that picks reliable negatives, and its output is
   not described as an estimate of anything.
2. **The gating measurement is the 1,121 real literature negatives, as ADR 024 set out.** Its standing
   changes. In ADR 024 it was prudence. It is now the **only available check** on the reliable-negative
   set. Nothing else can tell me whether the threshold cut the right part of the space. Roles 2 and 3 of
   ADR 024 remain intended uses with nothing measured yet.
3. **The naive P-against-U control of ADR 016 item 2 is retained, and is now better motivated.** Reason:
   the measurement in Context 4. Naive beat the SCAR method on 6 of 8 datasets, so "what did the spy step
   buy?" is a live question whose answer may be "little". It stays secondary, as ADR 014 item 1 and ADR 016
   item 2 set.
4. **No new dependency.** Bekker's code is at `dtai.cs.kuleuven.be/software/sar` with no code licence
   stated (note, footnote 7, p. 11). PULSNAR's is at `github.com/unmtransinfo/PULSNAR` and Zenodo
   10.5281/zenodo.13126647; the paper names no software licence (note, § Data Availability). Under
   CLAUDE.md §5 neither can be adopted until a licence is read at source and I agree to it. Reading a
   licence is not the same as agreeing to the dependency.
5. **`spy_fraction` cannot be inherited and must be measured.** PAYN's 0.2 and 0.05 were tuned at
   P:U = 1:1 (ADR 024 Context, SI pp. S11, S16). Mine is about 1:9.7 (`docs/design/modeling.md`, as cited
   by ADR 024). The research found nothing on how the spy threshold responds to the P:U ratio or to the size
   of U. That is a statement about the research session; the search is not recorded in the repository. The
   one relevant paper it returned treats spy rate and spy tolerance as **hyperparameters to search over**
   (arXiv:2401.06452, AutoML for PU learning; not read by me, not in the repository, id from the research
   brief). So the value is a measurement I owe, not a number to copy. It is fixed **before any Chemlex
   read**, by the same discipline as ADR 023: chosen on label-free evidence and the 1,121 negatives, then
   frozen.

I have not earned two things in this record, and I mark them. First, that the near-1 reading of PULSNAR is
the *only* thing it would return on my U. That is the note's reasoning from the formula, not a run. Second,
that finding 3 transfers to a spy threshold on chemistry data. Bekker et al. measured it on simulated bias in
eight non-chemical datasets.

## Consequences

- No code changes. The spy step is built as ADR 013 item 3, ADR 014 and ADR 016 describe it.
- The paper calls the spy step a heuristic for choosing reliable negatives. It does not call the
  threshold an estimate of π or of α.
- Stage S2 cannot choose `spy_fraction` and the tolerance by copying PAYN. A sweep over both, checked
  against the 1,121 negatives, is owed before Chemlex is read. Not designed here; it needs a count first
  (CLAUDE.md §8).
- The limitation ADR 024 states stands, and is sharper: the negatives are biased, so the precision
  estimated from them is biased in a direction I cannot name.
- This is a third consequence of the one SCAR defect of ADR 019 item 3, after ADR 024. It is not a new
  defect.
- ADR 019 named `e(x)`, SAR and SNAR without a source. The source is Bekker et al. 2019, Definition 1,
  p. 4, with SCAR, SAR and SNAR defined on pp. 3–4. One caution from the note: if publication depends on the
  true success probability beyond what the features capture, I am in SNAR, which that paper sets aside
  and offers nothing for.

### Open, not decided here

- **α on the inventory grid.** A thought that needs a check, not a plan. The inventory U (284,600 lab
  pairs, `docs/design/modeling.md` line 356) is not marginal-matched to P. Separability is measured
  there: the literature-against-lab adversarial AUC on `morgan` is 0.859 for acids and 0.879 for amines
  (`uv run python scripts/counts/count_adversarial.py`, as quoted in ADR 018 and ADR 020; that task is on
  substrates, not pairs). The plate is the inventory grid, so α there is the operationally relevant
  quantity. The trap: PULSNAR's assumption 2 needs the unlabeled positives to have support in P, and
  inventory substrates have poor support, which by the paper's own words gives underestimates. A different
  failure mode, not a way out.
- **He et al., arXiv:1808.02180.** Bekker's p. 13 notes that propensity falls with the probabilistic gap,
  so harder-to-label positives are less likely to be labelled (note, "what carries over"). That is close to
  my story that clean, publishable couplings get reported. It may be the right propensity shape. Nobody has
  read it; the id comes from the research brief, not from the note.
- **Whether SCAR can be tested rather than asserted.** Teisseyre, Furmanczyk and Mielniczuk,
  arXiv:2404.00145, already pointed to in ADR 019 Consequences. I assert SCAR fails and have never tested
  it. That paper assumes a single iid sample, so it does not apply off the shelf.

## Alternatives rejected

- **Estimating α with PULSNAR on the literature U.** Rejected. The separability it needs is what my
  sampler removes, and a near-1 reading would be uninterpretable (Context 1). Running PULSCAR alone as a
  cheap first reading (3.19 min at P 5,000 / U 100,000, note, Table 2) is the note's suggestion. I do not
  adopt it: the reading would not tell me anything I could act on.
- **Fitting a SAR propensity model by EM on my tables.** Rejected. It would absorb my sampling design and
  return it as publication propensity (Context 2).
- **Adopting either codebase now.** Rejected on licence (Decision 4).
- **Dropping the spy step for the naive control.** Not rejected on evidence. Context 4 says it may be
  competitive. It stays a control (Decision 3) because the paper needs the comparison, and because I
  cannot yet say the naive result carries over from simulated bias to mine.
- **Copying PAYN's 0.2 and 0.05.** Rejected: tuned at a P:U ratio I am not at (Decision 5).

## References

- `docs/literature/notes/bekker-2019-beyond-scar-propensity.md` (agent summary, not yet checked by me).
- `docs/literature/notes/kumar-2024-pulsnar-class-proportion-without-scar.md` (agent summary, not yet
  checked by me).
- `docs/literature/notes/boser-2026-payn.md` (agent summary, not yet checked by me).
- ADR 008; ADR 013 items 1 and 3; ADR 014 item 1; ADR 015; ADR 016 item 2; ADR 018; ADR 019 items 2–3
  and Consequences; ADR 020; ADR 023; ADR 024.
- `docs/design/modeling.md` (line 18, validation rate; line 356, inventory U; P:U 1:9.7).
