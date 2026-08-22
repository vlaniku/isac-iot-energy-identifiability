# When Communication Energy Is Sub-Percent

Code, result files and figures for *When Communication Energy Is Sub-Percent:
Evaluating Energy-Allocation Claims in Battery-Powered ISAC-IoT*.

Laniku, Krasniqi and Akyildiz. Submitted to IEEE Internet of Things Journal.

---

## What this is

Every quantitative claim in the paper is produced by a script here and written to
a file in `results/`. The figures in `figures/` are generated from those files;
none is drawn by hand. The intent is that a reader can check any number in the
paper by running the script that made it.

That includes the numbers we withdrew. The corrections log in the paper is not a
narrative device — the scripts that produced the withdrawn results are here too,
alongside the validation that killed them.

## Layout

```
code/       analysis scripts, one concern each
results/    every number in the paper, as JSON
figures/    the figures, PDF (vector) and PNG
```

## The corpus measurement

| script | what it does |
|---|---|
| `systematic_search.py` | the original recorded search, with strings and date filter |
| `corpus_iot_subset.py` | partitions the corpus; the IoT-facing subset is 7% of it |
| `corpus_iot_manifest.py` | freezes the 25-record IoT-facing population so the denominator cannot move |
| `corpus_iot_screen.py` | deduplication, inclusion criteria, full-text screen |
| `corpus_iot_classify.py` | wide context around every term hit, for hand adjudication |
| `corpus_multi_index.py` | the same screen across OpenAlex, Crossref and Semantic Scholar |

The full-text screen requires the papers themselves. **They are not redistributed
here** — they are copyrighted and were obtained through an institutional
subscription. `results/corpus_iot_manifest.json` carries every DOI, so the set
can be reassembled by anyone with access. Place the PDFs as `P01.pdf`, `P03.pdf`
and so on in a `docs/lit_corpus/` directory and re-run the screen.

## The bound and the regime map

| script | what it does |
|---|---|
| `comm_action_bracket.py` | shows the energy-identity bracket is a simulator artefact |
| `regime_map_v3.py` | the map, with common random numbers and two confounds removed |
| `regime_map_invariance.py` | 25 configurations; reproduces the published map to 4.2e-14 |
| `regime_map_invariance_readout.py` | the binding diagnosis and the re-read at the measured condition |
| `ceiling_band.py` | what is firm and what stays indeterminate |

## Deployment analyses, including the ones that failed

| script | what it does |
|---|---|
| `deploy_depletion_analysis.py` | depletion against workload |
| `deploy_workload_control.py` | the export capture problem, and the rank inversion it causes |
| `deploy_robustness.py` | power analysis, leave-one-out, rank tests |
| `channel_coherence.py` | link correlation against elapsed time |
| `uo_survival_v2.py` | Cox proportional hazards on the public archive; null, with its MDE |
| `hazard_scan_schedule.py` | state-conditioned scan cadence; null against its own floor |
| `coincidence_bound.py` | a bound we built |
| `coincidence_estimator_validation.py` | the Monte-Carlo that withdrew it |

## The proposed experiment

`experiment_protocol_power.py` — per-device noise calibration inverted from each
unit's own slope error, the closed-form paired-*t* expression, and a check that
Monte-Carlo agrees with it at run time.

## Data

Raw device telemetry is not included pending a decision on release scope.
`fetch_uo_archive.py` re-pulls the Newcastle Urban Observatory data, which is
public.

## Reproducing

Python 3.10+, with `numpy`, `scipy`, `matplotlib`, `pandas` and `pypdf`. Scripts
run from `code/` and write to `../results/`.

```bash
cd code
python regime_map_invariance.py        # ~45 min
python regime_map_invariance_readout.py
python make_figures.py
```

The network-facing scripts (`corpus_*`, `systematic_search.py`,
`resolve_references.py`) query OpenAlex, Crossref and Semantic Scholar. They
identify themselves with a contact address as those services ask, and they back
off on rate limits. Counts drift as indexes are updated; every result file
records its own retrieval date.

## Licence

Code under MIT. Result files and figures under CC BY 4.0.
