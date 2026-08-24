# When the Optimised Term Is Smaller Than the Instrument

Code, data, result files and figures for *When the Optimised Term Is Smaller
Than the Instrument: An Identifiability Criterion for Energy Claims in
Battery-Powered ISAC-IoT*.

Laniku, Krasniqi and Akyildiz. Submitted to IEEE Internet of Things Journal.

---

## What this is

The intent is that a reader can check a number in the paper by running the
script that made it. Every figure in `figures/` is generated from a file in
`results/`; none is drawn by hand.

Two things qualify that, and both are listed explicitly below rather than left
to be discovered: two result files were not produced by any script, and three
scripts read a third-party archive that is not mirrored here.

That includes the numbers we withdrew. The corrections log in the paper is not
a narrative device -- the scripts that produced the withdrawn results are here,
alongside the validation that killed them.

## Layout

```
code/       analysis scripts, and the simulator modules they import
data/       the deployment telemetry the analyses read
results/    the numbers, as JSON
figures/    the figures, PDF (vector) and PNG
```

## The data

`data/FIEK_parking_export_83day.xlsx` is the export the deployment analyses
read: 701 events from five LoRaWAN parking sensors, one sheet per device plus a
fleet summary and a combined `All Events` sheet. The columns that matter here
are `battery_v`, `f_cnt`, `spreading_factor`, `rssi`, `snr` and the timestamps.
Note `temperature_c`, which is present and empty in all 701 records -- that
absence is the reason Table XI of the paper gained a ninth row, because
conditioning on temperature is what would have recovered the independent noise
column the criterion originally assumed.

`data/chirpstack_12mo_metrics.json` is the twelve-month link-metric view read
from the network server; ChirpStack retains aggregates only, so monthly is the
finest granularity available over that span, and it carries no battery field.
`data/kadriu2024_public_events.xlsx` is the 2024 window released with Kadriu et
al., used only by `two_year_workload.py`.

There is no personal data here. The records are occupancy state, radio
statistics and battery readings for five parking bays; the device EUIs appear
in Fig. 1 of the paper.

`data/uo_archive/` holds only the two survival cohort definitions and a device
profile -- who entered the analysis, which is the part worth auditing. The
Newcastle Urban Observatory archive itself is not mirrored: it is public,
roughly 850 MB here, and `fetch_uo_archive.py` re-pulls it.

`build_release.py` is the script that assembles this repository from the
working tree. It selects by allow-list rather than by exclusion, scans
everything it selects for host addresses, tokens and local paths, and then
checks three structural properties: that every shipped script can import what
it needs, that every shipped result is written by a shipped script, and which
scripts depend on withheld data. It refuses to finish if the first two fail.
Those checks exist because the first build of this repository passed the leak
scan and still shipped four scripts that raised `ModuleNotFoundError`, two of
them the two the README told the reader to run first.

## The corpus measurement

| script | what it does |
|---|---|
| `systematic_search.py` | the original recorded search, with strings and date filter |
| `corpus_iot_subset.py` | partitions the corpus; the IoT-facing subset is 7% of it |
| `corpus_iot_manifest.py` | freezes the 25-record IoT-facing population so the denominator cannot move |
| `corpus_iot_screen.py` | deduplication, inclusion criteria, full-text screen |
| `corpus_iot_classify.py` | wide context around every term hit, for hand adjudication |
| `corpus_multi_index.py` | the same screen across OpenAlex, Crossref and Semantic Scholar |
| `novelty_check_restructured.py` | the lead claims put back through the search, with syntax controls |

The full-text screen requires the papers themselves. **They are not
redistributed here** -- they are copyrighted and were obtained through an
institutional subscription. `results/corpus_iot_manifest.json` carries every
DOI, so the set can be reassembled by anyone with access. Place the PDFs as
`P01.pdf`, `P03.pdf` and so on in a `docs/lit_corpus/` directory and re-run the
screen.

## The bound and the regime map

| script | what it does |
|---|---|
| `comm_action_bracket.py` | shows the energy-identity bracket is a simulator artefact |
| `regime_map.py` | the first map: the objective, the action table and the fixed-policy run |
| `regime_map_v3.py` | the map, with common random numbers and two confounds removed |
| `regime_map_invariance.py` | 25 configurations; reproduces the published map to 4.2e-14 |
| `regime_map_invariance_readout.py` | the binding diagnosis and the re-read at the measured condition |
| `ceiling_band.py` | what is firm and what stays indeterminate |
| `pareto_shell_size.py` | how much the non-dominated filter actually prunes (see below) |

Those scripts are built on a simulator that also ships:
`integrated_models.py`, `isac_physical_models.py`,
`energy_aware_isac_framework.py`, `closed_loop_simulator.py`,
`experiment_lifetime_budget.py`, `experiment_nonstationary.py`,
`experiment_temporal_ceiling.py`, `audit_binding_boundary.py` and
`adaptive_hybrid.py`. They are here because the regime map imports them, not
because this paper reports results from them.

## Deployment analyses, including the ones that failed

| script | what it does |
|---|---|
| `deploy_depletion_analysis.py` | depletion against workload |
| `deploy_workload_control.py` | the export capture problem, and the rank inversion it causes |
| `deploy_robustness.py` | power analysis, leave-one-out, rank tests |
| `two_year_workload.py` | the same devices two years apart. **The rate comparison is invalid** -- the 2026 side reads an export that captures 44-88% unevenly, so the decline it prints cannot be separated from ingestion loss. Only the rank correlation survives |
| `channel_coherence.py` | link correlation against elapsed time |
| `channel_coherence_control.py` | the same correlation under four constructions. The published one reuses each uplink in up to 22 overlapping pairs, which makes two long-gap correlations look significant; neither survives disjoint pairing or a device bootstrap |
| `uo_survival.py` | the first version. **Superseded** -- its covariate is the battery-reporting rate, a configuration setting rather than work done |
| `uo_survival_v2.py` | Cox proportional hazards on the public archive, covariate rebuilt from the payload streams; null, with its MDE |
| `hazard_scan_schedule.py` | state-conditioned scan cadence; null against its own floor |
| `coincidence_bound.py` | a bound we built. **Withdrawn** -- the script is published with its argument intact, because the validation is only legible next to what it refuted |
| `coincidence_estimator_validation.py` | the Monte-Carlo that withdrew it |

## The proposed experiment, and what it can resolve

`experiment_protocol_power.py` -- per-device noise calibration inverted from
each unit's own slope error, the closed-form paired-*t* expression, and a check
that Monte-Carlo agrees with it at run time.

`sf_energy_ratio.py` -- derives the SF7/SF12 amplification factor from the LoRa airtime
formula and a stated radio model, reproduces the constants the paper uses to within
1.9%, and prices what else a spreading-factor change moves. The receive windows are the
term that is not controlled, and they move against the design.

`noise_autocorrelation.py` -- checks the assumption the power expression rests on, and
finds it violated: lag-1 residual correlation of 0.46 at the analysis cadence, not
explained by the quantisation staircase and not removed by a quadratic trend. Inflates
every minimum detectable effect by 1.68x. The earlier Monte-Carlo validation could not
see this because it generated independent noise too.

`identifiability.py` -- the same expression inverted the other way. Instead of
asking what share a given design detects, it asks what campaign a claim of a
given size would need, and reports the smallest claim this fleet could check
at all. It recomputes every row of the published MDE table first and refuses
to print anything else if a row disagrees by more than 2%.

---

## What is not reproducible from what ships here

**Two result files have no producing script.**

| file | why |
|---|---|
| `literature_f_placement.json` | read by hand from the cited papers. Each entry names the paper and the table the value came from. There is nothing to automate; the check is to open the papers. |
| `rq4b_lorawan_battery_screen.json` | the archived pull, kept as the record of what was actually screened. The script that made it was not kept; `rq4b_deep_screen.py` restores the retrieval and reproduces it at 94% DOI overlap, the difference being index drift. The archived file is cp1252, not UTF-8. |

**Three scripts read the Newcastle archive, which is not mirrored here.**
`uo_survival.py` and `uo_survival_v2.py` read a 762 MB pickle cache built from
it, and `fetch_uo_archive.py` is what builds that cache. The archive is public
and the fetch script re-pulls it; mirroring 850 MB of someone else's open data
would add weight without adding access. Their outputs are in `results/`, so the
numbers can be read either way.

Everything else runs against the data in this repository. That was checked
rather than assumed: the ten result files those scripts write were regenerated
from a clean copy of this release and came back byte-identical to the ones
shipped here. The earlier version of this repository shipped the code for the
residual-dependence measurement and the injection validation while withholding
the telemetry they read, which made the paper's reproducibility claim false for
exactly the two analyses carrying its contribution.

## Known defects in the code released here

**The Pareto shell prunes far less than the superseded submission claimed.**
That submission's first contribution said the non-dominated filter reduces 27
actions to 5-12, with a median of six. `ParetoGridSelector` in
`integrated_models.py` enumerates 75 actions, not 27, and `pareto_shell_size.py`
measures what survives over 400 sampled states: **65.5 of 75 under TDMA and
62.0 of 75 under OFDMA**, so the filter removes 13% and 17% respectively. The
filter itself is a textbook non-dominated sort and it is correct -- the actions
in this grid genuinely trade off against one another, so few are dominated. The
defect was in the claim, not in the code, and the claim is not made in this
paper.

**The superseded submission's evaluation pipeline is not included.**
`run_full_evaluation.py` maps voltage to state of charge as
`clip(v / 4.2, 0.2, 1.0)`, which is the full-charge voltage of a 3.7 V Li-ion
cell applied to every device regardless of chemistry; the 12 V units in the
public archive are pinned at full charge for the whole run. It is not fixed,
because fixing it means asserting a chemistry and a full-to-empty voltage pair
for cells we have not identified, and an invented mapping is worse than a
labelled one. Neither that script nor its result files ship, and no number in
this paper comes from them.

## Reproducing

Python 3.10+, with numpy, scipy, matplotlib, pandas and pypdf. Scripts run from
`code/` and write to `../results/`.

    cd code
    python identifiability.py              # seconds
    python pareto_shell_size.py            # about 1 min
    python regime_map_invariance.py        # about 45 min
    python regime_map_invariance_readout.py

`make_figures.py` reads `data/` and now runs as shipped. The two analyses the
paper's criterion rests on are the quickest way to check that this release
works end to end:

    python noise_autocorrelation.py        # the residual dependence, and the 1.68x
    python identifiability_injection.py    # the criterion against real noise

`uo_survival.py` and `uo_survival_v2.py` need the Newcastle cache first; run
`fetch_uo_archive.py` to build it.

The network-facing scripts (`corpus_*`, `systematic_search.py`,
`resolve_references.py`) query OpenAlex, Crossref and Semantic Scholar. They
identify themselves with a contact address as those services ask, and they back
off on rate limits. Counts drift as indexes are updated; every result file
records its own retrieval date.

## Licence

Code under MIT. Result files, figures and the deployment data under CC BY 4.0.
The Newcastle Urban Observatory archive is not redistributed here and carries
its own terms.
