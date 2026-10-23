> **Provenance.** Summary written by a Claude agent on 2026-10-09 from the full PDF (Supporting Information), focused on this project's questions. Page references point to that PDF. Checked only where `docs/design/modeling.md` or ADR 003 relies on it; read the paper before citing a detail.
>
> **Reference.** Molga, Beker, Roszak, Czerwiński, Grzybowski, "Hierarchical reaction logic enables computational design of complex peptide syntheses", *J. Am. Chem. Soc.*, doi:10.1021/jacs.4c17057 (Supporting Information only). Mostly irrelevant to this project.

# Molga et al., Allchemy peptide synthesis (SI): reading notes

## Paper 2: `ja4c17057_si_001.pdf`

**Identity** (p. 1): Supporting Information for **"Hierarchical reaction logic enables computational design of complex peptide syntheses"** by Karol Molga, Wiktor Beker, Rafał Roszak, Andrzej Czerwiński and Bartosz A. Grzybowski (Allchemy Inc.; IOC PAS Warsaw; IBS CARS / UNIST). JACS, DOI 10.1021/jacs.4c17057, 29 pages.

**Topic**: a user manual and implementation notes for Allchemy's automated *retrosynthesis* module for peptides. It covers the control panel (pp. 1–13), the hierarchy of peptide disconnections in a config.xlsx file (pp. 18–21), and a tutorial on hand-coding SMARTS reaction rules (pp. 22–29). It is **not** about reaction outcome prediction, HTE or PU learning, and has no yield or success data. It is mostly **not relevant** to us.

There are minor tangential points, which I skimmed but did not read in depth:
- Amide couplings are ranked by steric hindrance. "Coupling of unhindered amines with unhindered acids [is] prioritized" (p. 21). Lactamizations at hindered or epimerization-prone residues get lower priority (p. 20). This is an expert prior that hindrance hurts couplings.
- Each rule lists SMARTS for **incompatible functional groups** under coupling conditions (HOBt/DIC in DMF). Examples are primary alkyl iodides, anhydrides, acyl chlorides, terminal Michael acceptors, free primary amines, phenols, thiols and activated methylenes (pp. 28–29). These could seed chemoselectivity or "reliable negative" filter features for our HATU/DIPEA/DMF couplings.
- The buyable catalogue is MCULE, about 2.5 M compounds with prices (p. 8). It could be one source for a commercial-availability propensity attribute.
