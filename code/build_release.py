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
  data/uo_archive/*       the Newcastle Urban Observatory mirror: ~85 MB of
                          zips and a 762 MB pickle cache, none of it ours and
                          all of it publicly downloadable. fetch_uo_archive.py
                          re-pulls it, so mirroring it here would add weight
                          without adding access. The two small cohort
                          definitions DO ship, because they record which
                          devices entered the survival analysis and that is a
                          choice a reader should be able to audit.
  data/2025-9-Battery.csv the same archive, one month of it.

WHAT NOW DOES SHIP, and why the scope decision went that way: the deployment
telemetry. Fifteen shipped scripts read data/, and the two carrying the
paper's contribution -- noise_autocorrelation.py, which finds the residual
dependence, and identifiability_injection.py, which validates the corrected
criterion against it -- were among them. The Reproducibility section states
that both are in the release. Shipping the code without the data it reads
would have made that sentence false. The files are our own deployment's, hold
no personal data, and the device EUIs in them are already printed in Fig. 1.
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
    'two_year_workload.py',
    'pareto_shell_size.py',
    'identifiability.py',
    'channel_coherence_control.py',
    'sf_energy_ratio.py',
    'rq4b_deep_screen.py',
    'noise_autocorrelation.py',
    'verify_manuscript.py',
    'identifiability_injection.py',
    'venue_genre_sweep.py',
    'venue_genre_sample.py',
    # Imported by the regime-map, bracket and ceiling scripts above. The
    # first build listed only the scripts a reader would run and none of
    # what they import, so four of them raised ModuleNotFoundError -- two
    # being the two the README tells the reader to run first.
    'integrated_models.py', 'isac_physical_models.py',
    'energy_aware_isac_framework.py', 'closed_loop_simulator.py',
    'experiment_lifetime_budget.py', 'experiment_nonstationary.py',
    'experiment_temporal_ceiling.py', 'audit_binding_boundary.py',
    'adaptive_hybrid.py',
]
RESULTS_GLOB = re.compile(r'\.json$')
# Internal working files that happen to live in results/. The references
# local_papers.json produced are in the bibliography; the file itself is an
# inventory of a private collection and carries the absolute path it was read
# from, so it is not shipped.
RESULTS_EXCLUDE = {'local_papers.json'}
FIGURES_GLOB = re.compile(r'^fig\d+.*\.(pdf|png)$')

# Telemetry, by allow-list for the same reason the code is. Paths are relative
# to data/. Everything not named here stays behind, and check_data_dependencies
# reports which scripts that still costs.
DATA = [
    'FIEK_parking_export_83day.xlsx',
    'chirpstack_12mo_metrics.json',
    'chirpstack_daily_final_days.json',
    'kadriu2024_public_events.xlsx',
    'uo_archive/_survival_cohort.csv',
    'uo_archive/_survival_cohort_v2.csv',
    'uo_archive/_device_profile.csv',
]

# ------------------------------------------------------------- leak scan ----
LEAKS = [
    (re.compile(r'185\.182\.158\.\d+'), 'ChirpStack host address'),
    (re.compile(r'\bgh[pousr]_[A-Za-z0-9]{20,}'), 'GitHub token'),
    (re.compile(r'(?i)\bpassword\s*[=:]'), 'password assignment'),
    (re.compile(r'(?i)\bapp[_-]?key\s*[=:]\s*["\'0-9a-f]{8,}'), 'LoRaWAN AppKey'),
    (re.compile(r'(?i)\bsecret\s*[=:]'), 'secret assignment'),
    # A local filesystem path is not a credential, so the original scan let it
    # through. It still reveals a user account and a directory layout, and it is
    # of no use to a reader.
    (re.compile(r'[A-Za-z]:\\{1,2}Users\\{1,2}'), 'absolute Windows path'),
    (re.compile(r'/home/[a-z0-9_.-]+/'), 'absolute POSIX home path'),
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


# --------------------------------------------------- structural checks -----
# The leak scan asks whether the release says something it should not. These
# ask whether it IS something a reader can use. The first build passed the leak
# scan and still shipped four scripts that could not import and thirty result
# files no shipped script regenerates -- including the two scripts the README
# tells the reader to run first. A net that only looks for secrets will clear a
# release that does not work.

# Result files no script here produces, each with the reason. A reader is owed
# the reason rather than a silent gap, so the README carries the same table.
# Anything not listed must have a producer among the shipped scripts.
RESULTS_NO_PRODUCER = {
    'literature_f_placement.json':
        'read by hand from the cited papers -- controllable-energy shares, '
        'each entry naming its source and the table it came from. There is '
        'nothing here to automate; the check is to open the papers.',
    'rq4b_lorawan_battery_screen.json':
        'the archived pull, kept as the record of what was actually screened '
        'when the paper was written. The script that made it was not kept; '
        'rq4b_deep_screen.py restores the retrieval and reproduces it at 94% '
        'DOI overlap, the difference being index drift. Note the archived file '
        'is cp1252, not UTF-8.',
}


def check_readme_title(tex=None, readme=None):
    """Does the README still name the paper the manuscript is?

    verify_manuscript.py watches the manuscript against its result files. It
    does not watch the release, and the release is a second surface that
    drifts: this repository shipped under the pre-revision title for two
    revisions after the paper was renamed, so a reviewer following the link
    would have landed on what reads as a different paper. Reading did not catch
    it either time. Returns a reason string, or None when clean.
    """
    tex = tex or os.path.join(ROOT, 'paper_v4', 'main.tex')
    readme = readme or os.path.join(DEST, 'README.md')
    if not (os.path.exists(tex) and os.path.exists(readme)):
        return None
    s = open(tex, encoding='utf-8', errors='replace').read()
    m = re.search(r'\\title\{(.+?)\}\s*\n\s*\n', s, re.S)
    if not m:
        return 'could not read \\title{} from the manuscript'
    # Unwrap the LaTeX line breaks and normalise whitespace.
    title = re.sub(r'\s+', ' ', m.group(1).replace('\\\\', ' ')).strip()
    body = re.sub(r'\s+', ' ', open(readme, encoding='utf-8',
                                    errors='replace').read())
    if title not in body:
        return 'README does not carry the manuscript title: %r' % title
    return None


def local_modules():
    return {f[:-3] for f in os.listdir(os.path.join(ROOT, 'code'))
            if f.endswith('.py')}


def imports_of(path):
    import ast
    try:
        tree = ast.parse(open(path, encoding='utf-8', errors='replace').read())
    except SyntaxError:
        return set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                names.add(node.module.split('.')[0])
    return names


def check_imports(shipped_code):
    """Every module a shipped script imports from this project must ship too."""
    local = local_modules()
    shipped = {f[:-3] for f in shipped_code}
    bad = []
    for f in sorted(shipped_code):
        missing = sorted((imports_of(os.path.join(DEST, 'code', f)) & local)
                         - shipped)
        if missing:
            bad.append((f, missing))
    return bad


def check_result_producers(shipped_code, shipped_results):
    """Every shipped result must be named by a shipped script, or declared."""
    text = [open(os.path.join(DEST, 'code', f), encoding='utf-8',
                 errors='replace').read() for f in shipped_code]
    return [r for r in sorted(shipped_results)
            if r not in RESULTS_NO_PRODUCER
            and not any(r in t for t in text)]


def check_data_dependencies(shipped_code):
    """Which shipped scripts still need a data file that does NOT ship.

    Not a failure -- the withheld remainder is the Newcastle mirror, which is
    public and re-pullable. But it is a fact a reader hits on the first run, so
    it belongs in the build output and in the README rather than in a traceback.
    Only unsatisfied dependencies are reported: naming files that now ship would
    make the release look less reproducible than it is, which is the error in
    the opposite direction and just as misleading.
    """
    ddir = os.path.join(ROOT, 'data')
    names = set()
    for root, _dirs, files in os.walk(ddir):
        for f in files:
            names.add(f)
    shipped_data = {os.path.basename(p) for p in DATA}
    need = {}
    for f in sorted(shipped_code):
        if f == 'build_release.py':
            continue
        txt = open(os.path.join(DEST, 'code', f), encoding='utf-8',
                   errors='replace').read()
        hits = sorted(n for n in names if n in txt and n not in shipped_data)
        if hits:
            need[f] = hits
    return need


def self_test():
    """Check the checkers: each must fire on a case built to make it fire.

    Both nets here replace ones that reported clean over a broken release, so
    neither is trusted until it has been watched to fail."""
    ok = True
    probe = os.path.join(DEST, 'code', '_selftest_probe.py')
    with open(probe, 'w', encoding='utf-8') as fh:
        fh.write('import integrated_models\n')
    fired = check_imports(['_selftest_probe.py'])
    if not (fired and fired[0][1] == ['integrated_models']):
        print('  SELF-TEST FAILED: the import check did not fire')
        ok = False
    os.remove(probe)
    orphan = check_result_producers([], ['_selftest_probe.json'])
    if orphan != ['_selftest_probe.json']:
        print('  SELF-TEST FAILED: the producer check did not fire')
        ok = False

    # The title check, against a README carrying the title this paper had
    # before it was renamed -- the exact drift that shipped twice.
    stale = os.path.join(DEST, '_selftest_readme.md')
    with open(stale, 'w', encoding='utf-8') as fh:
        fh.write('# When Communication Energy Is Sub-Percent\n')
    if not check_readme_title(readme=stale):
        print('  SELF-TEST FAILED: the README title check did not fire')
        ok = False
    if check_readme_title(readme=os.path.join(DEST, 'README.md')):
        print('  SELF-TEST FAILED: the README title check fired on a good README')
        ok = False
    os.remove(stale)
    return ok


def main():
    # Clear only the content directories. DEST is a git working tree, and
    # rmtree'ing it would take .git with it.
    for sub in ('code', 'results', 'figures', 'data'):
        d = os.path.join(DEST, sub)
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

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

    # A result file earns its place by being reproducible from what ships with
    # it. The first build took every .json in results/, which shipped 23 files
    # from the superseded simulation study whose scripts are not here and whose
    # numbers this paper does not report.
    shipped_src = [open(os.path.join(DEST, 'code', f), encoding='utf-8',
                        errors='replace').read()
                   for f in os.listdir(os.path.join(DEST, 'code'))
                   if f.endswith('.py')]
    dropped = []

    rdir = os.path.join(ROOT, 'results')
    for f in sorted(os.listdir(rdir)):
        if not RESULTS_GLOB.search(f) or f in RESULTS_EXCLUDE:
            continue
        if f not in RESULTS_NO_PRODUCER and not any(f in t for t in shipped_src):
            dropped.append(f)
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

    # Telemetry. The leak scan runs over these too. It is a text scan and two of
    # them are xlsx, so it reads the container rather than the cells -- which is
    # why the columns were also read by hand before this list was written; the
    # scan is not what clears them.
    for rel in DATA:
        src = os.path.join(ROOT, 'data', rel.replace('/', os.sep))
        if not os.path.exists(src):
            print("  MISSING from data allow-list: %s" % rel)
            continue
        h, s_ = scan(src)
        if h:
            hard_hits.append((rel, h))
            continue
        if s_:
            soft_hits.append((rel, s_))
        dst = os.path.join(DEST, 'data', rel.replace('/', os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        picked.append('data/' + rel)

    print("=" * 92)
    print("  RELEASE BUILD")
    print("=" * 92)
    print("  files selected : %d" % len(picked))
    print("    code    %d" % sum(1 for p in picked if p.startswith('code/')))
    print("    results %d" % sum(1 for p in picked if p.startswith('results/')))
    print("    figures %d" % sum(1 for p in picked if p.startswith('figures/')))
    print("    data    %d" % sum(1 for p in picked if p.startswith('data/')))
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
    print("  NOT included by design: docs/lit_corpus (copyrighted PDFs), the")
    print("  Newcastle mirror under data/uo_archive (public, re-pullable, 850 MB),")
    print("  and every internal document.")
    if dropped:
        print("  not shipped, no script here regenerates them (%d):" % len(dropped))
        for f in dropped:
            print("      %s" % f)
        print()

    code_files = [q[5:] for q in picked if q.startswith('code/')]
    res_files = [q[8:] for q in picked if q.startswith('results/')]
    failed = False
    if not self_test():
        failed = True
    else:
        print("  check self-test: all three structural checks fire when they should")
    bad = check_imports(code_files)
    if bad:
        failed = True
        print("  *** IMPORT CHECK FAILED -- shipped scripts that cannot run ***")
        for f, missing in bad:
            print("      %-42s needs %s" % (f, ", ".join(missing)))
    else:
        print("  import check: every shipped script imports what it needs")
    stale_title = check_readme_title()
    if stale_title:
        failed = True
        print("  *** README CHECK FAILED -- %s" % stale_title)
    else:
        print("  README check: the release names the paper the manuscript is")
    orphans = check_result_producers(code_files, res_files)
    if orphans:
        failed = True
        print("  *** PRODUCER CHECK FAILED -- results no shipped script makes ***")
        for r in orphans:
            print("      %s" % r)
    else:
        print("  producer check: every shipped result traces to a shipped script")
    for r, why in sorted(RESULTS_NO_PRODUCER.items()):
        if r in res_files:
            print("  declared, no producer: %s" % r)
            print("      %s" % why)

    need = check_data_dependencies(code_files)
    if need:
        print()
        print("  still needs withheld data -- the Newcastle mirror, which "
              "fetch_uo_archive.py re-pulls (%d):" % len(need))
        for f, hits in sorted(need.items()):
            print("      %-38s %s" % (f, ", ".join(hits)))

    print("\n  Wrote %s" % DEST)
    if failed:
        print("\n  BUILD IS NOT RELEASABLE -- fix the failures above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
