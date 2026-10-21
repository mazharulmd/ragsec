# Manuscript readiness and reviewer assessment

Reviewed on 18 October 2026 against the manuscript, experiment implementation, pinned results,
and recent RAG-security literature. This is a revision assessment, not a prediction of editorial
outcome.

## Current assessment

The manuscript is more accurate and reproducible than the earlier draft, but the scientific
contribution still looks incremental for a competitive Q1 security journal. The paper now frames
itself as a controlled component-level diagnosis, rather than a new defense or the first adaptive
RAG evaluation. The remaining novelty problem is substantive: the adaptive threat, layered
defenses, and poisoning benchmarks overlap with published and recent work, while this study does
not reproduce those systems head to head or introduce a new defense, benchmark, or threat model.

*Computers & Security* was removed from consideration after checking its [current scope](https://shop.elsevier.com/journals/computers-and-security/0167-4048),
which excludes security of AI/ML/LLM systems. The *Journal of Information Security and
Applications* is a more plausible subject fit, but its [guidance](https://shop.elsevier.com/journals/journal-of-information-security-and-applications/2214-2126)
screens out incremental AI/ML applications.
JISA is Q1 under current SJR/Scopus quartiles and Q2 in Journal Citation Reports. Confirm which
quartile index is required before describing it as the target Q1 venue.

## What changed in this revision

- The title, abstract, introduction, and highlights state the measured scope and avoid a
  first-of-its-kind, general-defense, or enterprise-scale claim.
- Related work now discusses SafeRAG, the 2025 poisoning benchmark, RAGForensics, TriShieldRAG,
  and *Through the Stealth Lens*. Comparisons are explicitly qualitative because those systems
  were not reproduced.
- The HMAC mechanism is described as symmetric message authentication under a protected
  publisher/signing service. The paper no longer implies that document IDs establish independent
  evidence or publishers.
- Main results, supplementary figures, result macros, and the local-model table are generated
  from the experiment artifacts and included by the Elsevier manuscript. The local-model result
  remains explicitly exploratory.
- The main-study uncertainty is a 2,000-replicate bootstrap over question clusters, conditional
  on three seeds. The public artifact export keeps SHA-256 cluster IDs and omits query, gold, and
  generated-answer text, while retaining target-answer token-F1 scores for sensitivity analysis.
- The Elsevier source now uses space-fitted, generated tables for the main comparison,
  provenance controls, and clean-utility results. The complete 23-page PDF was built against
  `elsarticle.cls` 3.4 with the requested `5p,times` options.
- Highlights were rewritten and are each within Elsevier's 85-character limit.
- The class remains `\documentclass[5p,times]{elsarticle}`.

## Evidence and reproducibility limits

- The main factorial uses a deterministic mock generator, three small corpora, one sentence
  transformer retriever, exact inner-product search, and three seeds. It does not establish
  production-scale behavior.
- The pinned run completed all eight aggregate studies: 2,322 aggregate rows, plus 66,780
  query-level outcomes for the main uncertainty and target-F1 analysis. Dataset and embedding
  revisions are recorded in the artifact manifest. This is a reproducibility record, not evidence
  of independent replication.
- The local confirmation uses Qwen2.5 0.5B and Llama 3.2 1B in Q4_K_M, 100 SQuAD questions per
  cell, and one seed. It shows a strong clean-accuracy cost and is not a hosted-model substitute.
  No commercial hosted-model run was completed.
- The adaptive campaign is a manually specified, one-factor-at-a-time diagnostic, not an
  optimized or feedback-driven attacker. The attacker and defenses share one implementation.
- Binary success is scored with normalized substring matching, which can count incidental answer
  mentions. The main mock study now adds normalized target token-F1 as a graded sensitivity check;
  exact match is still absent. The local-model confirmation exports only aggregates, so it has no
  query-cluster intervals or target-F1 analysis.
- The main comparison does not reproduce SafeRAG, RobustRAG, TrustRAG, TriShieldRAG, or the
  attention-based detector from *Through the Stealth Lens* on a common dataset and threat model.
- HMAC results assume the adversary cannot compromise or invoke the authorized publisher. The
  experiment does not cover malicious authorized publishers, key rotation, freshness, replay,
  or public-key alternatives.

## Highest-priority work before submission

1. **Strengthen novelty.** Add a directly comparable baseline or a genuinely new mechanism or
   benchmark contribution. A broader discussion alone will not resolve the incremental-work
   concern.
2. **Broaden model evidence.** Add more than two small local models and more than one seed; if a
   hosted model is claimed, report the exact provider/model/version and a reproducible protocol.
3. **Strengthen the adaptive evaluation.** Add an independently designed or optimized attacker,
   limited-target campaigns, and an implementation-independent replication.
4. **Complete metric validation.** Add exact match and target-F1 results to the real-model
   confirmation, preferably with query-level outputs and cluster intervals. The current model
   follow-up is aggregate-only, and the main study's token-F1 metric remains a sensitivity
   measure alongside substring ASR.
5. **Confirm journal requirements and author declarations.** Authors must verify funding,
   conflicts of interest, CRediT roles, acknowledgements, the AI-assisted-writing declaration,
   and the requested quartile index. The current AI statement says all authors reviewed and
   edited the text; they must confirm that is accurate. These details cannot be inferred from
   the repository and have not been invented.
6. **Archive the final artifact.** A GitHub commit is useful for review; a DOI-backed release
   (for example, through Zenodo) is preferable for a journal submission.

## Literature overlap to cite and compare directly

- SafeRAG, ACL 2025: <https://aclanthology.org/2025.acl-long.230/>.
- “Benchmarking Poisoning Attacks against Retrieval-Augmented Generation,” arXiv 2505.18543:
  <https://arxiv.org/abs/2505.18543>.
- RAGForensics, WWW 2025: <https://doi.org/10.1145/3696410.3714756>.
- TriShieldRAG, 2026 preprint: <https://arxiv.org/abs/2607.23838>.
- “Through the Stealth Lens,” ICML 2026: <https://proceedings.mlr.press/v306/choudhary26b.html>.
