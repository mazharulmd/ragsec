# Paper readiness review

Reviewed against the current experiment code, committed result files and the authoritative
Elsevier manuscript on 17 October 2026. This is a pre-writing assessment; the manuscript's
claims should be revisited after the evidence gaps below are addressed.

Since the initial review, I completed the missing toy-corpus cells in `backfill_strict`, bringing
the offline result set to 2,322 rows, regenerated the tables and figures, and resumed the full
offline suite successfully. I also corrected the manuscript's dataset-size statement and made
the confidence-interval and signing-key assumptions explicit. The Elsevier manuscript now
builds in a temporary TeX environment; a real-model run remains blocked because no provider key
or local model endpoint is configured.

## What is already in place

- The repository contains eight offline experiment result files with 2,322 JSONL records, plus
  generated tables and figures. The default experiment suite is deterministic and intended to
  reproduce those records.
- The experiment suite includes a real-model confirmation study in code. It compares two
  models, four defense configurations and three attacks on a reduced SQuAD sample.
- The manuscript has a limitations section that identifies the mock generator, small corpora,
  one adaptive attacker, implementation coupling and excluded model/network latency.
- The main result is appropriately qualified in places: adaptive poisoning defeats the tested
  content filters, while the signature result depends on the modeled source signing key remaining
  unavailable to the attacker.

## Highest-priority gaps before paper submission

1. **No real-model confirmation evidence is committed.** There is no `confirmation.jsonl` in
   `results/paper/`. The manuscript still says replication on hosted and open-weight models is
   future work. Until the confirmation study runs, keep the paper's conclusions explicitly
   scoped to the deterministic mock and avoid presenting the study's existence in code as
   empirical validation. The default study uses two models and roughly 3,200 completion calls
   for one SQuAD seed; estimate provider cost and record exact model IDs, endpoint/provider,
   date, sample count and failures with any run.
2. **Confidence intervals pool repeated queries across seeds.** The manuscript reports Wilson
   intervals over pooled attacked queries. The same QA questions are reused across seeds, so
   those observations are not necessarily independent. Report per-seed variability and consider
   query-clustered bootstrap intervals (or another method that preserves query as the sampling
   unit) alongside pooled estimates.
3. **The measured corpus scale is small.** The evaluated corpora contain 30 to 1,999 documents,
   and the paper itself notes that this does not establish behavior at enterprise index scale.
   Either add a larger-corpus/scale sensitivity experiment or make the boundary prominent in the
   abstract and conclusion.
4. **The signature result is a trust-boundary result.** The experiment signs legitimate
   documents with a simulated shared HMAC key and assumes the adversary cannot obtain it. Clarify
   that signatures authenticate an ingestion path only when key custody and signer
   authorization are sound; they do not prevent poisoning by a compromised or malicious
   authorized publisher. Avoid framing HMAC itself as a complete deployment solution.
5. **Independent replication is still needed.** Attack and defense implementations share one
   codebase, and the adaptive attack was designed against the evaluated defenses. The paper
   acknowledges this coupling; a second implementation or an independently designed adaptive
   attack would strengthen the central finding.

## Paper and artifact cleanup

- Verify each manuscript number against the generated tables and raw records after any rerun;
  `paper-submission/MainPaper.tex` has been edited directly and is authoritative, while the
  manuscript flattener is only a cross-check. All 341 result macros in the current submission
  match the regenerated macro file exactly.
- Record exact corpus snapshots/splits, embedding model and package versions, run command, and
  random seeds in a compact reproducibility appendix. Dependency ranges and live dataset
  loaders alone do not pin a reproducible environment.
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
