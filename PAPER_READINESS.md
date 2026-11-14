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

## Checked against the journal's Information for Authors (supplied by the author, 14 Nov 2026)

| Requirement | Status |
|---|---|
| Regular Paper, at most 13 double-column pages at 10 pt, everything included | Met: 12 pages |
| IEEE template | Met: `IEEEtran` journal class |
| Abstract 150-250 words, self-contained, no abbreviations, equations, footnotes or references | Met after revision: the abbreviation "RAG" and inline math were removed |
| Problem, contribution, significance, closest literature, what is new | Present in Sections I and II |
| Supplemental material, at most six pages, numbered consecutively with the manuscript | `Supplement.tex`, 4 pages, Tables V-X and Figs. 4-5 |
| Statement on previously rejected manuscript, with verbatim decisions and a scope motivation | `PriorSubmissions.tex` drafted; **the verbatim decision text must be pasted in by the authors** |
| ORCID for every author | Available; enter in the Author Portal |
| EDICS category | To choose at submission; see below |
| List of posted preprints with URLs | Authors to supply; none is known to this repository |
| Single-anonymized review | Author block is in the manuscript |

**Resubmission rule.** A manuscript rejected by any journal for a reason other than scope may
be resubmitted only once. The ESWA rejection was for scope and does not count. Whether the
MDPI desk rejection counts depends on its stated reason, which is not recorded in this
repository. If it was not a scope rejection, this submission is the single permitted
resubmission, so it should not be spent before the review risks below are reduced.

**Charges.** Published pages beyond ten are billed at USD 220 each and cannot be waived; at
12 pages that is USD 440 if the accepted version keeps this length. Voluntary charges
(USD 110 per page for the first ten) are optional. Revisions may grow to 16 pages, which
raises the mandatory charge.

**Immediate-rejection grounds named by the journal** are lack of novelty, insufficient
experimental data, poor English, and insufficient citations. The second is this manuscript's
main exposure; see "Review risks".

**EDICS.** The unified list could not be opened from this environment. A copy about three
years old lists IFS-CYB (cybersecurity) and IFS-APC (applied cryptography) among the
Information Forensics and Security categories; IFS-CYB looks closest. Confirm on the current
list.

**Still not verified.** The reference-lookup services and publisher sites were unreachable.
A 2026 TIFS paper titled "Provenance-Anchored Retrieval Integrity" was found only through an
IEEE DataPort dataset page. It is **not cited** and should be read and cited or distinguished
if it exists, because it would be the closest prior work in the target journal. The two added
references (RAGShield, the Sybil attack) were not checked against a bibliographic database.

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
- Two text/table disagreements fixed. The previous draft said aggregation alone lowered
  adaptive attributable success on HotpotQA; Table II shows 0.97 -> 0.98. It also said that
  dropping source spoofing changes little; the evasion ablation shows it removes the attack.
- Related work adds RAGShield (arXiv:2604.00387v1), a provenance-based RAG defense that
  itself reports residual insider success, and the Sybil attack as the classical statement of
  the corroboration problem. The claim that prior provenance work ignores insiders was
  softened accordingly.
- Figure legends say "per-principal cap" instead of "independence discounting".
- AI-assistance disclosure moved to the Acknowledgment section, as IEEE requires.
- The 42-row served-model table, the dataset table, the supporting ablation tables, the
  heatmap and the ingestion-ladder figure moved to `Supplement.tex`.

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
