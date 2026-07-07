# Registration — Built with Claude: Life Sciences

**Event:** Built with Claude: Life Sciences (Anthropic × Gladstone Institutes × Cerebral Valley)
**Apply at:** https://cerebralvalley.ai/e/built-with-claude-life-sciences
**Deadline:** Sunday **July 5, 2026, 11:00 PM EST** (≈ 5:00 AM Mon, July 6, CEST)
**Event dates:** July 7–13, 2026 · fully virtual · runs on Discord
**Track chosen:** ✅ **Researcher** ("Build from the bench" — Claude Science)
**Team:** Florian Boser + Prof. Frank Glorius (max team size 2)

> ⚠️ **Both team members must submit their own application.** This file is Florian's; a parallel draft for Prof. Glorius is at the bottom.
> ⚠️ **Claude Science requires Mac / Linux / WSL 2** — Florian is on Linux ✓. Confirm Prof. Glorius is too, or his form should say Builder.
> ⚠️ For the **API org UUID**, use a **personal claude.ai account** — an account under the university's Anthropic agreement is *not* eligible for the credits.

---

## Florian Boser — application answers

**0. Where do you study or work?**
> Organisch-Chemisches Institut, University of Münster, Germany — group of Prof. Frank Glorius.

**1. Where do you live (City, State, Country)?**
> ⬜ *[FILL IN — e.g., "Münster, North Rhine-Westphalia, Germany"]*

**2. Which track are you applying for: Researcher or Builder?**
> **Researcher**

**3. Describe your background in life sciences and/or building things with software. Tell us what you actually do day to day.**
> PhD researcher in Prof. Frank Glorius's group (University of Münster), working at the interface of organic synthesis, machine learning, and lab automation. Day to day I run high-throughput experimentation (HTE) campaigns, build ML models for reaction-yield and reactivity prediction, and develop agentic/automation workflows for the lab. I am co-first author of PAYN (J. Am. Chem. Soc. 2026), an open-source Python framework that uses positive-unlabeled learning to recover negative data from biased reaction datasets, and a contributor to TACIT (reaction-outcome prediction from positive-only data, under revision at Nature Machine Intelligence). My teammate is Prof. Frank Glorius, who leads the group.

**4. What do you want to build or investigate at this hackathon? Be specific about the problem, not just the topic area.**
> Exploratory HTE and direct-to-biology plates suffer from low synthetic hit rates — many wells simply fail, wasting compounds and screening capacity. We will build a model, trained only on biased literature data, that pre-scores a large substrate × substrate matrix under a fixed set of *general* reaction conditions (generalist Buchwald–Hartwig C–N and/or heteroaryl Suzuki couplings) to predict, per pairing, (i) whether it reacts and forms the intended product, and (ii) its expected yield. We fuse two methods from our group: TACIT (implicit-negative bond-ranking) as the feasibility gate, and PAYN, our spy-based positive-unlabeled framework, to recover reliable negatives and rank yield. From these scores we design a high-hit-rate, diverse plate. During the week we will (1) assemble a literature C–N / Suzuki dataset (Reaxys / USPTO / ORD), (2) train and validate against held-out and out-of-distribution HTE ground truth (Neves, Ahneman, Perera), reporting the hit-rate uplift of a designed plate vs. a naive plate with no new experiments, and (3) deliver it as a reproducible, open-source extension of PAYN with a plate-design demo. The payoff: plates whose crude mixtures go straight to biological assay (direct-to-biology) yield more real, testable small molecules per plate — accelerating the discovery of better small-molecule drugs.

**5. What's the work you're most proud of? Share a link if you can (repo, paper, preprint, demo, product).**
> PAYN — "Positivity is All You Need": Yield Prediction of Organic Reactions in Biased Data Sets via Positive-Unlabeled Learning. *J. Am. Chem. Soc.* 2026, 148, 15066–15075 (co-first author). DOI: 10.1021/jacs.6c00127. Open-source framework: https://github.com/GloriusGroup/PAYN. I'm also a contributor to TACIT (reaction-outcome prediction from positive-only data, under revision at *Nature Machine Intelligence*).

**6. What is your experience level with Claude Code?**
> **Experienced**

**7. What's your LinkedIn?**
> ⬜ *[FILL IN — LinkedIn URL]*

**8. What's your GitHub?**
> https://github.com/FlorianBoser

**9. Google Scholar, ORCID, or publications page?**
> https://orcid.org/0000-0003-0550-3822  ⬜ *[optionally add Google Scholar URL]*

**10. What's your GitHub? (optional)**
> https://github.com/FlorianBoser

**11. Personal site, portfolio, or company website (optional)**
> ⬜ *[optional — leave blank or add]*

**12. What's your Twitter / X? (optional)**
> ⬜ *[optional — leave blank or add]*

**13. What is your Anthropic API organization UUID?**
> ⬜ *[FILL IN — from https://platform.claude.com/settings/organization. Use a **personal** claude.ai account, not an institutional one.]*

**14. What's your Discord handle?**
> ⬜ *[FILL IN — create one if needed; the event runs on Discord]*

**15. How did you hear about this hackathon?**
> ⬜ *[FILL IN — e.g., "Anthropic / Cerebral Valley announcement" or a colleague]*

---

## Prof. Frank Glorius — parallel application (submit separately)

Mirror the answers above; only these differ:

**0. Where do you study or work?**
> Full Professor of Organic Chemistry, Organisch-Chemisches Institut, University of Münster, Germany (group leader).

**2. Track:** Researcher *(confirm Mac/Linux/WSL 2 for Claude Science; else Builder)*

**3. Background / day to day:**
> Full Professor of Organic Chemistry at the University of Münster. My group's research spans photocatalysis, C–H activation, asymmetric and heterogeneous catalysis, arene hydrogenation, and organocatalysis, alongside data-science and molecular machine-learning tools with direct practical relevance to synthetic chemistry (e.g., PAYN and TACIT). Teammate: Florian Boser.

**4. What to build/investigate:** *(same project as Florian's #4 — reuse verbatim.)*

**5. Proudest work:**
> ⬜ *[Prof. Glorius to choose a representative paper]* — ORCID: https://orcid.org/0000-0002-0648-956X

**6. Claude Code experience:** ⬜ *[Prof. Glorius to self-assess]*
**7–15.** ⬜ *[Prof. Glorius's own LinkedIn / GitHub (if any) / ORCID above / API org UUID (personal account) / Discord / how heard]*

---

## Pre-submit checklist
- [ ] Florian: fill ⬜ fields (location, LinkedIn, API org UUID, Discord, how-heard)
- [ ] Prof. Glorius: submit his own application (⬜ fields + confirm track/OS)
- [ ] Both: personal claude.ai account ready for the API org UUID
- [ ] Both: Discord handle created
- [ ] Submit before **July 5, 11:00 PM EST** (≈ 5:00 AM Mon CEST)
