# Paper readiness review

Reviewed against the current experiment code, committed result files and the authoritative
Elsevier manuscript on 17 October 2026. This is a pre-writing assessment; the manuscript's
claims should be revisited after the evidence gaps below are addressed.

Since the initial review, I completed the missing toy-corpus cells in `backfill_strict`, bringing
the offline result set to 2,322 rows, regenerated the tables and figures, and resumed the full
offline suite successfully. I also corrected the manuscript's dataset-size statement and made
the confidence-interval and signing-key assumptions explicit. The Elsevier manuscript now
builds in a temporary TeX environment; a real-model run remains blocked because no provider key
or local model endpoint is configured. The hard-coded public HMAC key has since been removed;
the signed-ingestion harness now creates a fresh key in memory for each runner and does not write
it to configuration or result records.

## What is already in place

- The repository contains eight offline experiment result files with 2,322 JSONL records, plus
  generated tables and figures. The suite is deterministic and now pins its external datasets,
  but the archived records were produced before revision metadata was captured.
- The experiment suite includes a real-model confirmation study in code. It compares two
  models, four defense configurations and three attacks on a reduced SQuAD sample.
- The manuscript has a limitations section that identifies the mock generator, small corpora,
  one adaptive attacker, implementation coupling and excluded model/network latency.
- The main result is appropriately qualified in places: adaptive poisoning defeats the tested
  content filters, while the signature result depends on a protected signer that the upload
  adversary cannot invoke or compromise.

## Highest-priority gaps before paper submission

1. **No real-model confirmation evidence is committed.** There is no `confirmation.jsonl` in
   `results/paper/`. The manuscript still says replication on hosted and open-weight models is
   future work. Until the confirmation study runs, keep the paper's conclusions explicitly
   scoped to the deterministic mock and avoid presenting the study's existence in code as
   empirical validation. The default study uses two models and roughly 3,200 completion calls
   for one SQuAD seed; estimate provider cost and record exact model IDs, endpoint/provider,
   date, sample count and failures with any run.
2. **Query-level uncertainty remains unavailable.** The manuscript retains pooled rates as
   descriptive estimates and adds the observed minimum-to-maximum range across seeds, rather
   than treating repeated questions as independent Wilson trials. The stored records contain
   only per-seed aggregates, so they cannot support a query-clustered bootstrap. Persist
   per-question outcomes and rerun the study before making population-level uncertainty claims.
3. **The measured corpus scale is small.** The abstract and conclusion now state that the
   evaluated corpora contain 30 to 1,999 documents and do not establish enterprise-scale
   behavior. A larger-corpus or scale-sensitivity experiment is still needed for that claim.
4. **The signature result is a trust-boundary result.** The experiment now generates a fresh
   HMAC key in memory and assumes the adversary cannot invoke or compromise the trusted signer.
   It still does not prevent poisoning by a malicious authorized publisher. Signatures now bind
   the document ID, source ID, and text in a canonical envelope, but do not prevent replay of an
   old signed document. A deployment needs explicit signer authorization, freshness/replay
   controls, and key rotation/revocation.
5. **Independent replication is still needed.** Attack and defense implementations share one
   codebase, and the adaptive attack was designed against the evaluated defenses. The paper
   acknowledges this coupling; a second implementation or an independently designed adaptive
   attack would strengthen the central finding.
6. **The archived dataset snapshots are unknown.** The suite now pins SQuAD and HotpotQA to
   immutable Hub commits and records source, revision, configuration and split in new result
   rows. Existing `results/paper/` rows predate that metadata; rerun the study before claiming
   that its current numbers correspond to the newly pinned snapshots.
7. **Attack success uses a lenient scorer and full query access.** The paper now discloses that
   it uses normalized substring matching and gives the attacker every benchmark question as a
   target. A strict exact-match/token-F1 comparison and limited-target campaign remain untested.

## Paper and artifact cleanup

- Verify each manuscript number against the generated tables and raw records after any rerun;
  `paper-submission/MainPaper.tex` has been edited directly and is authoritative, while the
  manuscript flattener is only a cross-check. All 341 result macros in the current submission
  match the regenerated macro file exactly.
- Record exact corpus snapshots/splits, embedding model and package versions, run command, and
  random seeds in a compact reproducibility appendix. Dataset pins are now in the suite, but
  dependency versions and the archived dataset snapshots remain unresolved.
- Include clean utility and false-positive costs beside attack reductions wherever space allows;
  the defense's utility trade-off is central to interpreting the results.
- If confirmation results are added, update the limitations and abstract/conclusion together,
  and report the generator's model, decoding parameters and whether the result cache was used.
- The manuscript now uses Elsevier's `5p,times` two-column layout with numeric citations. Wide
  tables and plots span both columns, and the updated 23-page PDF has been rendered for visual
  review. There are no undefined citations or references; several small paragraph overfull boxes
  (largest about 11 pt) remain for a final typography pass.

## Immediate repository correction

The README said the suite runs eight studies while listing the opt-in confirmation study as a
ninth. It now distinguishes the eight offline studies from the paid, opt-in confirmation run.
