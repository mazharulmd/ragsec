# Experimental Results

Narrative summary of the evaluation. The authoritative numbers are the generated tables in
`paper/tables/` and the raw records in `results/paper/*.jsonl`; regenerate both with:

```bash
ragsec suite
ragsec figures
```

## Setup

Three corpora (SQuAD, HotpotQA with distractors, a bundled toy corpus), a
`all-MiniLM-L6-v2` retriever over an exact inner-product index, `top_k = 3` with a candidate
multiplier of 10, and a deterministic mock generator that answers extractively and — unless
hardened — obeys instructions found in retrieved text. Seven configurations, five attacks,
three seeds. Point estimates pool query-seed trials descriptively; figure error bars show the
observed minimum-to-maximum range across seeds. Since the same questions recur across seeds,
these ranges measure seed variability rather than population-level uncertainty over questions.

Two metrics matter for reading what follows:

- **ASR** — the adversary's goal was met.
- **Attributable ASR** — ASR restricted to queries the *same configuration* answered correctly
  with no adversary present. For the suppression attack the raw rate absorbs the pipeline's own
  error rate; the attributable rate is the part the attack caused.

## 1. The undefended pipeline is vulnerable to everything

Pooled over the four static attacks, ASR is 0.95 on SQuAD and 0.95 on HotpotQA. The adaptive
attack reaches 0.82 and 0.96 respectively. Clean accuracy is 0.62 (SQuAD) and 0.25 (HotpotQA) —
the latter is low because the mock generator answers from a single passage and HotpotQA is
multi-hop, which is exactly why the attributable rate is needed.

## 2. Against static attacks, the layered defense works

| Configuration | Static ASR (attributable), SQuAD | Clean accuracy |
|---|---:|---:|
| No defense | 0.95 | 0.62 |
| L1 source verification | 0.00 | 0.62 |
| L2 retrieval filtering | 0.01 | 0.61 |
| L3 robust aggregation | 0.68 | 0.52 |
| L4 monitoring | 0.95 | 0.62 |
| Full defense | 0.00 | 0.51 |

L1 and L2 each independently neutralise the published attack constructions, by different
mechanisms: L1 rejects the poison before indexing, L2 removes it from the candidate set. L3
defeats prompt injection completely but not targeted poisoning, and carries the largest utility
cost. L4 changes nothing by construction — it alerted under every attack, which is what a
detective control does.

Context flooding under the full defense shows raw ASR 0.49 on SQuAD but **attributable ASR
0.00**: the defense stops it entirely, and the raw figure was measuring baseline errors.

## 3. Against an adaptive adversary, the same stack fails

The adaptive attack rewrites the payload — declarative rather than interrogative, assertive
rather than imperative, padded with neutral prose, uploaded under the source name the
deployment already trusts.

| Configuration | Adaptive ASR (attributable) | SQuAD | HotpotQA | Toy |
|---|---|---:|---:|---:|
| No defense | | 0.83 | 0.97 | 0.41 |
| L3 robust aggregation | | 0.88 | 0.98 | 1.00 |
| Full defense | | 0.85 | 0.91 | 0.95 |
| **Full defense + signed ingestion** | | **0.01** | **0.00** | **0.00** |

Two things stand out.

**The retrieval filter's true-positive rate against the adaptive attack is 0.00.** Not degraded
— zero. Its ROC lies on the chance diagonal, so no choice of operating point recovers it, even
though the same filter is near-separable against every static attack.

**Robust aggregation makes things worse.** On every corpus, enabling L3 *raised* adaptive attack
success relative to the undefended pipeline. The adversary uploads several textually distinct
documents making the same claim, so the redundancy bonus that rewards corroboration rewards the
adversary. On the toy corpus the effect is starkest (0.41 → 1.00), because the true answer is
supported by exactly one document and the adversary supplies three.

## 4. Only three of six evasion techniques are load-bearing

Leave-one-out on SQuAD under the full defense (3 poisoned documents per target):

| Adaptive variant | ASR | Poison retrieved | Filter TPR |
|---|---:|---:|---:|
| All six techniques | 0.83 | 1.00 | 0.00 |
| − query paraphrase | 0.00 | 1.00 | 0.15 |
| − assertive payload | 0.01 | 0.91 | 1.00 |
| − provenance spoofing | 0.01 | 0.00 | — |
| − natural padding | 0.83 | 1.00 | 0.00 |
| − similarity dilution | 0.83 | 1.00 | 0.00 |
| − corroboration | 0.83 | 1.00 | 0.00 |

The two necessary content-level techniques are single rewriting decisions that cost the
adversary nothing. The third is a credential problem, not a text problem. Note the difference in
mechanism: without paraphrasing the poison is still retrieved but gets filtered; without
provenance spoofing it is never indexed at all.

## 5. Ingestion controls: claims versus proofs

Adaptive attack success as the ingestion control is strengthened (quota = 1.5× the legitimate
corpus size, so a source may grow what it contributed by half):

| Ingestion control | SQuAD static | SQuAD adaptive | HotpotQA adaptive |
|---|---:|---:|---:|
| None | 0.99 | 0.82 | 0.96 |
| Trust threshold | 0.01 | 0.82 | 0.96 |
| Source allowlist | 0.01 | 0.82 | 0.96 |
| Allowlist + quota | 0.01 | 0.15 | 0.96 |
| **HMAC-signed ingestion** | **0.01** | **0.01** | **0.00** |

Against a static adversary, trust thresholds and allowlists work. Against an adversary who
spoofs the source name they do nothing — a source identifier and a trust score are attributes
the uploader *asserts*, not ones they must prove.

The quota rung repays a careful reading. It bites on SQuAD (900 uploads against a remaining
budget of 150) and does nothing on HotpotQA (600 uploads against a budget of nearly 1000). It
is a volume control, not an authenticity control: it prices a mass campaign and is blind to a
targeted one — and the budget sweep shows the adaptive attack works at one document per target.
A quota expressed as a fixed document count is worse than useless: set below the corpus size it
silently truncates the knowledge base, and the resulting inability to answer then reads as
security.

Requiring an HMAC signature the adversary cannot forge takes adaptive attack success to
approximately zero, at no cost to clean accuracy. This is the clearest result in the study, and
it is a supply-chain result rather than a machine-learning one.

## 6. The filter has no useful operating point

Sweeping the calibrated suspicion threshold τ on SQuAD (L2 only, attributable ASR):

| τ | TPR static | TPR adaptive | FPR clean | Clean accuracy | ASR adaptive |
|---:|---:|---:|---:|---:|---:|
| 1.0 (default) | 1.00 | 0.00 | 0.01 | 0.61 | 0.84 |
| 0.8 | 1.00 | 0.07 | 0.02 | 0.45 | 0.73 |
| 0.7 | 1.00 | 0.21 | 0.09 | 0.25 | 0.46 |
| 0.6 | 1.00 | 0.46 | 0.30 | 0.16 | 0.19 |
| 0.5 | 1.00 | 0.76 | 0.56 | 0.06 | 0.07 |

Detection of the adaptive attack *is* purchasable — by switching the system off. Static-attack
TPR is 1.00 across the entire sweep, so no single threshold serves both adversaries.
Threshold-free: AUROC 0.97–1.00 against static attacks, **0.32–0.65 against the adaptive one**
(at or below chance on two of three corpora).

## 7. Adversary budget: the attack costs two documents

Attributable ASR against the full defense, by documents per target:

| b | SQuAD | HotpotQA | Toy |
|---:|---:|---:|---:|
| 1 | 0.70 | 0.76 | 0.31 |
| 2 | 0.85 | 0.91 | 0.95 |
| 20 | 0.87 | 0.97 | 0.95 |

Against the *undefended* pipeline the adaptive curve is flat (one passage suffices). Against the
*defended* one it steps at b=2 — that is L3's redundancy bonus changing hands. With one poisoned
document the false claim is a minority of one; with two it carries the corroboration bonus and
the genuine passage does not. Everything above b=2 is wasted effort, which is the wrong shape
for a defender: the attack that works leaves the smallest footprint.

## 8. Over-retrieval is the wrong lever

The intuitive claim — "filtering needs backfill or it becomes a denial of service" — is true
only in a regime nobody should deploy in. SQuAD, sweeping multiplier m:

| | m=1 | m=20 |
|---|---:|---:|
| **τ=1.0** clean acc / abstain / static ASR / p95 | 0.51 / 0.31 / 0.00 / 5 ms | 0.51 / 0.31 / 0.00 / 63 ms |
| **τ=0.6** clean acc / abstain / static ASR / p95 | 0.11 / 0.67 / 0.25 / 6 ms | 0.16 / 0.04 / 0.03 / 45 ms |

At the default threshold the filter discards under 1% of candidates, so over-retrieval changes
*no outcome at all* and multiplies latency tenfold. Under aggressive filtering backfill does
repair availability (abstention 0.67 → 0.04) but recovers only a fraction of the lost utility
(0.11 → 0.16, against 0.51 at the default threshold). Backfill cannot manufacture evidence a
mistuned filter threw away. Tune the threshold; don't compensate with the multiplier.

## 9. Other findings

- **Latency.** Measured time is dominated by filtering (25.2 ms of a 26.5 ms defended query),
  because the layer scores mk = 30 candidates. This ranking is an artefact of the local mock
  generator: the component that would dominate a real deployment — L3's k-fold multiplication of
  generator calls — is exactly the one this harness makes invisible.
- **Generator hardening.** Disabling instruction-following eliminates prompt injection
  (0.95 → 0.01) and leaves the other four attacks unchanged. They exploit grounding, not
  obedience.

## Interpretation

Content-based heuristics belong in a RAG security architecture as detection and as incident
evidence, not as the preventive control. The preventive control has to be verifiable
provenance. A corroboration-weighted aggregator deployed without an enforced bound on how much
evidence the adversary controls is not a neutral addition — it is an amplifier pointed at
whichever majority exists.

## Limitations

- The generator is a deterministic mock, not a production language model. It is a controlled
  instrument for measuring the retrieval pipeline, and the generator study bounds only the
  instruction-following part of that assumption.
- Corpora are hundreds of documents, not an enterprise index.
- The adaptive adversary is one adversary, built by us against our own defense: a lower bound on
  what adaptive attackers achieve, not an upper bound. A stronger one would optimise passage
  embeddings against the retriever directly.
- Attacks and defenses live in one codebase, which risks benchmark coupling; per-layer
  attribution mitigates but does not remove this.
- Reported latencies exclude external model and network cost.
