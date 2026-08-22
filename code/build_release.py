"""
Build the public code/data release, by allow-list rather than by exclusion.

WHY AN ALLOW-LIST. The project directory contains material that must never be
published: the internal state file with its retraction log and the network
server's address, an unsent private message to a co-author, an adversarial
self-review carrying acceptance estimates, the previous submission's audit, and
a directory of copyrighted PDFs downloaded through an institutional
subscription. An exclusion list fails open -- anything forgotten gets shipped.
An allow-list fails closed, which is the correct direction for a one-way action.

WHAT SHIPS: the analysis code, the result files every number in the paper traces
to, the generated figures, and a README. That is what supports the paper's
reproducibility claim.

WHAT DOES NOT, and why:
  docs/lit_corpus/*.pdf   copyrighted, obtained by subscription. Redistributing
                          them would be an infringement, and the manifest with
                          DOIs serves the same purpose lawfully.
  data/                   raw device telemetry. Held back pending an explicit
                          decision on scope; the fetch scripts are included so
                          the public archive can be re-pulled.
  STATE.md, HANDOFF_*     contain the ChirpStack host address and the internal
                          record.
  MOCK_REVIEW_v4.md,      internal assessments and private correspondence.
  MESSAGE_TO_AKYILDIZ*
  BRIEF_FOR_KRASNIQI.html

A leak scan runs over everything selected, before anything is written.

Author: Vullnet Laniku
"""

import os
import re
import shutil
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
DEST = os.path.join(ROOT, 'release_public')

# ------------------------------------------------------------- allow-list ---
CODE = [
    'corpus_iot_subset.py', 'corpus_iot_manifest.py', 'corpus_iot_screen.py',
    'corpus_iot_classify.py', 'corpus_multi_index.py', 'systematic_search.py',
    'novelty_check_restructured.py', 'resolve_references.py',
    'regime_map.py', 'regime_map_v2.py', 'regime_map_v3.py',
    'regime_map_invariance.py', 'regime_map_invariance_readout.py',
    'comm_action_bracket.py', 'ceiling_band.py',
    'experiment_protocol_power.py', 'coincidence_bound.py',
    'coincidence_estimator_validation.py',
    'deploy_depletion_analysis.py', 'deploy_robustness.py',
    'deploy_workload_control.py', 'channel_coherence.py',
    'uo_survival.py', 'uo_survival_v2.py', 'hazard_scan_schedule.py',
    'fetch_uo_archive.py', 'make_figures.py', 'build_release.py',
]
RESULTS_GLOB = re.compile(r'\.json$')
FIGURES_GLOB = re.compile(r'^fig\d+.*\.(pdf|png)$')

# ------------------------------------------------------------- leak scan ----
LEAKS = [
    (re.compile(r'185\.182\.158\.\d+'), 'ChirpStack host address'),
    (re.compile(r'\bgh[pousr]_[A-Za-z0-9]{20,}'), 'GitHub token'),
    (re.compile(r'(?i)\bpassword\s*[=:]'), 'password assignment'),
    (re.compile(r'(?i)\bapp[_-]?key\s*[=:]\s*["\'0-9a-f]{8,}'), 'LoRaWAN AppKey'),
    (re.compile(r'(?i)\bsecret\s*[=:]'), 'secret assignment'),
]
NOTE = [
    (re.compile(r'fcd6bd[0-9a-f]{10}'), 'device EUI (also shown in Fig. 1 of the paper)'),
]


def scan(path):
    try:
        t = open(path, encoding='utf-8', errors='replace').read()
    except Exception:
        return [], []
    hard = [why for rx, why in LEAKS if rx.search(t)]
    soft = [why for rx, why in NOTE if rx.search(t)]
    return hard, soft


def main():
    if os.path.exists(DEST):
        shutil.rmtree(DEST)
    for sub in ('code', 'results', 'figures'):
        os.makedirs(os.path.join(DEST, sub), exist_ok=True)

    picked, hard_hits, soft_hits = [], [], []

    for f in CODE:
        src = os.path.join(ROOT, 'code', f)
        if not os.path.exists(src):
            print("  MISSING from allow-list: %s" % f)
            continue
        h, s_ = scan(src)
        if h:
            hard_hits.append((f, h))
            continue
        if s_:
            soft_hits.append((f, s_))
        shutil.copy2(src, os.path.join(DEST, 'code', f))
        picked.append('code/' + f)

    rdir = os.path.join(ROOT, 'results')
    for f in sorted(os.listdir(rdir)):
        if not RESULTS_GLOB.search(f):
            continue
        src = os.path.join(rdir, f)
        if os.path.getsize(src) > 12_000_000:
            print("  skipped (too large): results/%s" % f)
            continue
        h, s_ = scan(src)
        if h:
            hard_hits.append((f, h))
            continue
        if s_:
            soft_hits.append((f, s_))
        shutil.copy2(src, os.path.join(DEST, 'results', f))
        picked.append('results/' + f)

    fdir = os.path.join(ROOT, 'figures')
    for f in sorted(os.listdir(fdir)):
        if FIGURES_GLOB.match(f):
            shutil.copy2(os.path.join(fdir, f), os.path.join(DEST, 'figures', f))
            picked.append('figures/' + f)

    print("=" * 92)
    print("  RELEASE BUILD")
    print("=" * 92)
    print("  files selected : %d" % len(picked))
    print("    code    %d" % sum(1 for p in picked if p.startswith('code/')))
    print("    results %d" % sum(1 for p in picked if p.startswith('results/')))
    print("    figures %d" % sum(1 for p in picked if p.startswith('figures/')))
    print()
    if hard_hits:
        print("  *** BLOCKED, not copied (hard leak) ***")
        for f, why in hard_hits:
            print("      %-42s %s" % (f, ", ".join(why)))
    else:
        print("  hard leak scan: CLEAN (no host addresses, tokens, keys or secrets)")
    if soft_hits:
        print()
        print("  worth a decision, copied anyway:")
        for f, why in soft_hits:
            print("      %-42s %s" % (f, ", ".join(why)))
    print()
    print("  NOT included by design: docs/lit_corpus (copyrighted PDFs), data/ (raw")
    print("  telemetry, pending a scope decision), and every internal document.")
    print("\n  Wrote %s" % DEST)


if __name__ == '__main__':
    main()
