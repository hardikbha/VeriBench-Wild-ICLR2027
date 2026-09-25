# VeriBench-Wild: A Benchmark and Measurement Testbed for Misinformation Evaluation

Anonymous repository accompanying a submission to ICLR 2027 (double-blind review).
Author information is withheld.

A higher benchmark score need not mean better misinformation detection: it also depends on the
information a model receives, the sources it is adapted and evaluated on, how it is adapted, and how
its outputs are scored. VeriBench-Wild is a benchmark and measurement testbed for studying these
choices. This repository contains the public Tier-1 data, the frozen prompts, the aggregate results
behind the paper's figures and tables, and the script that regenerates them.

## The resource

- **27,158 real-world records** from X Community Notes and from PolitiFact and GossipCop records
  distributed through UPFD, with frozen splits: 21,852 train / 2,651 validation / 2,655 test.
- **Primary task T1 (mechanism typology):** claim + fact-check note → six non-exclusive
  mechanisms (factual error, manipulated media, outdated information, missing context, unverified
  claim, satire). The labels are the misleading-reason tags chosen by the Community Notes writers,
  so a note and its labels share an author. T1 therefore measures evidence-conditioned mechanism
  recognition, not independent claim verification.
- **Official T1 setup:** train and select on Community Notes records only
  (17,728 / 2,133 / 2,150); UPFD records carry only one mechanism label, and missing labels are not
  negatives.
- Supporting task views (T2-HP, T-Prop, T-Author), a generation task (T0), and a construction
  diagnostic (T-Match) are described in the paper (Table 1, Appendix A). Not every record supports
  every task.
- The collection is naturally multilingual but English-dominant (English: 81.7% of training and
  78.9% of test records), centered on X/Community Notes, and a static snapshot (latest record
  timestamp 2025-02-27).

## The three audits (paper, Section 4)

The audits use named subsets of the resource and different targets; they are not an additive
decomposition of one error measure.

| Audit | Task and pool | What varies | Held fixed | Main result |
|---|---|---|---|---|
| Information access | T1, 2,150 Community Notes test records | claim only, note only, claim + note; random, same-mechanism, or masked note | test records and labels; prompted-model weights | Removing the note costs 12.7–21.3 macro-F1 points across five models; note-only inputs stay close to full input. |
| Source and adaptation | binary verdict, UPFD vs. Community Notes, unmatched and strict-matched pools | 16-shot in-context learning vs. QLoRA (which also uses far more supervision) | initial Qwen3-4B backbone; evaluation records within each pool | Home-source gap *G* = 35.56 [31.67, 39.45] points for QLoRA vs. 3.07 [−5.51, 11.66] for ICL; strict matching lowers the QLoRA gap to 13.83 [8.21, 19.45] while changing the pools. |
| Scoring | T0, 384 generated items, 476 mechanism judgments | eight judge-and-parser pipelines; strict vs. lenient parser for the five added judges | generated outputs, prompt, mechanism definitions | Acceptance of the same outputs ranges from 15.3% to 63.4%; human-majority acceptance is 16.4%; re-parsing Qwen3-4B's stored replies moves it from 0.4% to 31.9%. |

## Reporting results on VeriBench-Wild

- **T1:** state the permitted input channels, the CN-only training setup, and the data view
  (Tier-1 or full).
- **Transfer:** state the source pools, matching rule, adaptation regime, and supervision budget,
  and report all four source-to-target scores beside the gap.
- **Generation:** state the judge and its version, the prompt, the parser, and the denominator,
  and report agreement with human labels where available. Acceptance alone is neither correctness
  nor agreement.

## Contents

```
README.md
LICENSE
CITATION.cff
data/tier1/                 public Tier-1 root splits (train/val/test) and SHA256SUMS
prompts/                    frozen prompts, definitions, and masking vocabulary (verbatim from the paper)
results/final_results.json  aggregate results behind Figures 2-4 and Tables 3-4
scripts/build_figures.py    regenerates Figures 2-4 and Tables 3-4 from results/ (no model calls)
scripts/requirements.txt
figures/, tables/           outputs of scripts/build_figures.py
```

## What is and is not in this repository

| Asset | Status |
|---|---|
| Tier-1 root splits: claim and note text, task labels, anonymous author groups, aggregate cascade statistics; post/note identifiers removed, X-internal URLs replaced by a placeholder, timestamps coarsened to month | included (`data/tier1/`) |
| Frozen generator and evaluator prompts, evaluator mechanism definitions, T1 and binary-transfer prompts, external verdict prompt, masking vocabulary | included (`prompts/`) |
| Aggregate results and the figure/table script | included (`results/`, `scripts/`) |
| Task-specific files (T2-HP strict labels, author histories, pair files, generation-task files) | not included; described in the paper's Appendix A |
| Full propagation graphs | gated; not included |
| Exact author histories and timestamps | not public |
| Generation-audit sample, stored judge replies, individual human annotations | not included; the paper reports their prompts, parser rules, and aggregate results |
| Trained model checkpoints | not included |

Tier-1 supports the paper's public T1 references. Other views need task-specific inputs that are not
in Tier-1, and scores can differ across representations, so report which one you use.

## Data format

Each line of `data/tier1/*.tier1.jsonl` is one record with the fields `record_id`, `lang`,
`temporal_bucket`, `tweet_created_month`, `claim_text`, `author_id_anon`, `evidence`
(including `note_text`), `source`, `cascade_summary_stats`, `labels`, `metadata`, and `splits`.
T1 targets are `labels.sub_reasons` (six booleans). `metadata.tier` is one of `wild`,
`wild_control`, `wild_support` (Community Notes) or `upfd_legacy` (UPFD); the official T1 setup
excludes `upfd_legacy`.

Known text artifacts are left unchanged so that reported results stay reproducible: salted
16-character hexadecimal tokens replace some ordinary words in note text and post previews, some
`@USER` placeholders are merged with the next character, and language tags include codes for
undetermined or non-linguistic content (for example `und`, `zxx`).

Verify the files with:

```bash
cd data/tier1 && sha256sum -c SHA256SUMS
```

## Regenerating the figures and tables

```bash
pip install -r scripts/requirements.txt
python scripts/build_figures.py
```

This reads `results/final_results.json` and writes `figures/fig2_information.pdf`,
`figures/fig3_source.pdf`, `figures/fig4_scoring.pdf`, `tables/table3_judges.tex`, and
`tables/table4_human_agreement.tex`. No model calls or data downloads are needed.

## License

See `LICENSE`. Community Notes-derived records are CC-BY-NC-SA-4.0; UPFD-derived records
(`metadata.tier == "upfd_legacy"`) inherit CC-BY-SA-4.0 from UPFD. Scripts are MIT; prompts and
aggregate results are CC-BY-4.0. Do not attempt to re-identify posters or annotators.

## Citation

See `CITATION.cff`. The citation is anonymized for review.
