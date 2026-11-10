# Independent PU literature memo, 2026-11-02

**This is a third-party summary, not a paper. Do not cite it as one.** Cite the underlying papers, after
checking them.

- **What it is:** a literature memo produced by an independent AI literature search, run from my brief on
  2026-11-02. It is not peer reviewed and I did not write it.
- **Location, as of 2026-11-02:** `/Users/lukapraprotnik/Downloads/2026-11-02-pu-literature-memo.pdf`
  (380 KB, 8 pages, received 2026-11-02 at 14:21). `~/Downloads` is not an archive. The file needs moving
  somewhere durable beside my other literature, and this note updated when it moves.
- **The note is the only record the repository will hold.** `.gitignore` line 38 excludes every PDF at any
  depth ("Nothing this project tracks is a PDF", CLAUDE.md §4), so the memo cannot be committed. That makes
  this note load-bearing.
- **Filename:** the deviation from `<firstauthor>-<year>-<slug>` is deliberate. That convention presumes a
  paper with an author; this has none. Do not rename it.
- **Licence of data:** none; no data in it.
- **Date read:** 2026-11-02.
- Agent summary, not yet checked by Luka. Nothing below has been checked against a PDF.

## What they did

The memo searched for work on positive-unlabeled learning, literature-trained reaction models tested on new
plates, and reaction fingerprints, and gives bibliographic details. This section is thin on purpose: nobody
here has read any of the three papers below, so a fuller summary would be a summary of a summary. What this
note carries is the claim, the number, the access level the memo had, and that nobody here has checked it.

## Access levels, as the memo states them (memo p. 1)

| level | papers |
|---|---|
| full text read | Fitzner 2023 only |
| abstract only | ScopeBO, Liu/Isayev, Beker, DRFP, Jang 2020, Kariofillis 2022, Zhang 2025 |
| search-engine summary only | Weiss & Provost, Zhao 2019, Strieth-Kalthoff 2022, Jia 2019, Raccuglia 2016, Rana 2024, Kovács 2021, Wallach & Heifets, McSherry & Najork, Cabanac 2010, Huang & von Lilienfeld, Mordelet & Vert, Elkan & Noto, Nippa 2024, King-Smith 2024, UniReact |

The memo says Crossref was rate-limited, so some details were confirmed only through Europe PMC. Ben-David
2007 is a bibliographic check, not a reading.

## Findings that matter to this project

Page numbers below are the **memo's own pages**, not pages of the underlying papers, which nobody here has
opened (the way ADR 026 cites Bekker by printed page).

1. **Fitzner et al., ACS Omega 2023** (memo p. 2, "Prior art (Q7)"; the memo gives no volume or page, not
   filled in here). About 62k Buchwald-Hartwig reactions from Reaxys, CAS and USPTO, tested on 768 new plate
   reactions. Ranking correlation about zero, sometimes negative. The closest direct test of my thesis, and
   it is negative. The memo adds that this is a search result and not proof that no such paper exists.
2. **Liu, Moroz and Isayev, Chem. Sci. 2023, 14, 10835** (PMC10566507; memo p. 4, "Yield regression
   target"). 41,239 Reaxys amide couplings. Best R² 0.395 ± 0.020, against R² about 0.9 on an HTE set.
   Removing "reactivity cliffs" and uncertain yields raised it only to 0.457, so the gap is not mostly noise.
   My exact chemistry. Abstract only. Beside it, Saebi et al., Chem. Sci. 2023, 14, 4997: a graph neural
   network matching the state of the art on HTE gave no predictive model on electronic-lab-notebook data.
3. **Beker et al., JACS 2022, 144, 4819** (PMC8949728; memo p. 6, "Q4, reaction-specific evidence"). More
   than 10,000 literature heteroaryl Suzuki couplings. No model or representation beat a frequency baseline
   for conditions. The memo attributes this to chemist preference, reagent availability and missing
   negatives. Abstract only.
4. **Same page, memo p. 6:** Kovács, McCorkindale and Lee, Nat. Commun. 2021, doi 10.1038/s41467-021-21895-w,
   on "Clever Hans" scaffold bias in the Molecular Transformer; and the domain-shift frame with Ben-David et
   al., NIPS 2007 and Zhao et al., ICML 2019. This is where the unchecked citation in ADR 030 Context 7
   comes from.

## Plate design: the memo's power arithmetic (memo p. 8, Q6)

The memo reckons about 57 wells per arm to tell 29 % from 55-57 % at α = 0.05 with 80 % power. It warns that
29 % is the held-out HTE rate under one condition, not the inventory's base rate, so the concurrent random
arm has to measure that. This is the only figure in the memo that bears on the plate design rather than the
model. It says my 96 wells split evenly may be underpowered. This is the memo's arithmetic, unchecked by me.
[n: I have not recomputed it; a command is needed before it enters a design document.]

## What carries over to this project

- Findings 1 to 3 are the prior I must beat or explain. They say literature-trained models transfer poorly to
  plates, so the frequency baseline and the tiers in my protocol are not optional.
- Claim used in ADR 030 Context 7: lowest provenance AUC as a selection criterion lacks theoretical support
  (Ben-David 2007; Zhao 2019). It rests on a Crossref check and a search-engine summary.
- DRFP cancels unchanged substructure by design (ADR 030, Alternatives). Abstract only; licence MIT per the
  memo's author, verified at source by them.

## What does not carry over

- The memo's reading of any paper beyond Fitzner 2023 is unverified.
- Nothing in it is about a virtual unlabeled substrate space, which is what I build.

## Numbers we may cite

Each figure is a claim from the memo, located by the memo's page above (p. 2, 4, 6, 8). None is located in
the underlying paper. Before citing one in anything published, I read the paper and add its page or table
here. [n]
