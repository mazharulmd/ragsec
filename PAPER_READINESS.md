# Manuscript readiness

Updated 14 November 2026. This is a revision assessment, not a prediction of editorial outcome.

## Target venue

**IEEE Transactions on Information Forensics and Security (TIFS)**, Regular Paper.

Why this venue, after the *Expert Systems with Applications* scope rejection:

- The paper's contribution is an information-security one: a threat-model extension
  (compromised authorized signer), an authentication/provenance trust analysis, and an
  aggregation rule keyed on verified signer identity. ESWA is an applied-AI venue, and the
  paper does not present an expert system or an application study.
- The two Elsevier security journals considered earlier are closed to it. *Computers &
  Security* states that security of AI/ML systems (LLMs, federated learning) is out of scope,
  and the publisher's listing for the *Journal of Information Security and Applications* says
  the same.
- TIFS's scope is "the sciences, technologies, and applications relating to information
  forensics, information security, biometrics, surveillance and systems applications that
  incorporate these features". Secondary sources list RAG-poisoning papers in TIFS (BadRAG,
  vol. 20, 2025) and in IEEE TDSC (2026).
- TIFS is Q1 in both SJR and JCR.

Fallback with the same format family: *IEEE Transactions on Dependable and Secure Computing*.

**Not verified from this environment.** The publisher sites (IEEE Xplore, Signal Processing
Society, ScienceDirect) and the reference-lookup services were unreachable, so the items below
come from search-result summaries and must be confirmed on the journal's own pages before
submission:

- the 13-page double-column limit for initial Regular Paper submissions (16 on revision), and
  overlength charges above 10 published pages;
- the current EDICS category list (choose the closest security/authentication category);
- that the TIFS version of BadRAG and the 2026 TIFS paper "Provenance-Anchored Retrieval
  Integrity" exist as described. The latter was found only through an IEEE DataPort dataset
  page, is **not cited**, and should be read and cited or distinguished if it is real, because
  it is the closest prior work in the target journal.

## What changed for this venue

- Class changed from `elsarticle` to `IEEEtran` (journal), numeric IEEE citations, IEEE
  keywords, author block in the manuscript (TIFS review is single-anonymous). 12 pages.
- Elsevier-only items removed: highlights, separate title page, declaration of interest and
  ORCID files, `elsarticle` class files. ORCIDs are entered in the submission system.
- The paper is reorganized around the security thesis. The compromised-publisher adversary,
  the per-principal cap, its bound (Property 1), and the binding attack are now the core; the
  layered ablation is condensed into one subsection as the unauthorized-writer baseline.
- Notation repaired: the principal is `pr(j)` throughout (the previous draft used both `pi`
  and `kappa`, and `kappa` also denoted the HMAC key).
- One text/table disagreement fixed: the previous draft said aggregation alone lowered
  adaptive attributable success on HotpotQA; Table II shows 0.97 -> 0.98.
- Related work adds RAGShield (arXiv:2604.00387v1), a provenance-based RAG defense that
  itself reports residual insider success, and the Sybil attack as the classical statement of
  the corroboration problem. The claim that prior provenance work ignores insiders was
  softened accordingly.
- Figure legends say "per-principal cap" instead of "independence discounting".
- AI-assistance disclosure moved to the Acknowledgment section, as IEEE requires.
- The 42-row served-model table, the dataset table, the heatmap and the ingestion-ladder
  figure are no longer in the main paper; their numbers are quoted in the text and the files
  remain as supplementary material.

## Review risks that formatting does not fix

These are the likely grounds for rejection at a venue of this standard. None could be
addressed here without new experiments.

1. **Deterministic generator.** The headline numbers come from a mock generator that is
   credulous by construction. The served-model confirmation is two 7-8B models, 100
   questions, one seed.
2. **The cap is unconfirmed on served models** (4 discordant pairs of 200, McNemar p = 0.125),
   and the compromised-publisher configuration has never been run on a served model.
3. **No reproduced baselines.** RobustRAG, TrustRAG and RAGShield are not reimplemented; the
   identifier-counting control arm is the paper's own rule.
4. **Scale.** Corpora of 30 to 1,999 documents; one retriever.
5. **Partial effect.** With the cap, 0.59 of attacks still succeed on SQuAD at b = 5.

Highest-value additions before submitting: run the compromised-publisher study on served
models with several seeds; add RobustRAG's keyword/decoding aggregation as an external
baseline arm; evaluate on a larger corpus (e.g. Natural Questions).

## Repository gaps

- `tables/tab_binding.tex` has no source file in `paper-submission/artifacts/`
  (`binding.jsonl` is absent), so Table IV cannot be regenerated from the public bundle.
- `ragsec figures` needs the `datasets` extra and network access to rebuild the dataset table;
  Figures 4 and 5 were regenerated directly from the artifacts for the legend change.

## Author actions before submission

- Confirm the items under "Not verified" above.
- Confirm funding, competing-interest and authorship statements, and that the AI-use
  disclosure is accurate.
- Archive the artifact with a DOI (e.g. Zenodo) and cite it in the paper.
- IEEE prefers institutional e-mail addresses; the second author's is a personal address.
