"""
Screen the frozen IoT-facing corpus: deduplicate, apply inclusion criteria, and
- once PDFs are present - convert SALIENCE into PRESENCE by full-text search.

WHY THIS EXISTS. The paper's lead claim is that ISAC work naming IoT does not
model a standing charge. Measured on titles and abstracts that is a claim about
what authors foreground. A reviewer can answer it in one sentence: "of course
sleep is not in the abstract of a beamforming paper." The only way to close that
is to read the papers, and the only reason that is feasible is that the
IoT-facing population is 25 records rather than 360.

THREE STAGES, reported separately, because a systematic screen that reports only
its final number is not checkable.

  1. DEDUPLICATE. The retrieval contains preprint/journal pairs of the same work
     and at least one record duplicated across two indexing sources. Counting
     those twice would inflate the denominator.
  2. INCLUDE / EXCLUDE. Surveys, reviews and standardisation roadmaps do not
     carry a system model, so they cannot model a standing charge and cannot
     fail to. Counting them in the denominator would manufacture the result.
     They are excluded and counted, not silently dropped.
  3. FULL-TEXT SCREEN. For each included paper, search the whole text for any
     standing-charge term. Report per-paper hits with the matching context, so
     every classification can be checked by hand.

Stages 1 and 2 run today from the manifest. Stage 3 runs when PDFs are in
docs/lit_corpus/ as <slug>.pdf.

WHAT WOULD FALSIFY THE PAPER'S CLAIM: any meaningful number of included papers
carrying a circuit-power, sleep, idle or duty-cycle term in the system model. If
that happens it is the finding and the claim narrows again. This script is
written to find that, not to avoid it.

Author: Vullnet Laniku
"""

import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results')
LITDIR = os.path.join(HERE, '..', 'docs', 'lit_corpus')
MANIFEST = os.path.join(RESULTS, 'corpus_iot_manifest.json')

# Standing-charge terms. Deliberately generous: it is better to over-detect and
# inspect the context by hand than to under-detect and overstate the result.
SC_PATTERNS = [
    r'circuit\s+power', r'circuitry\s+power', r'static\s+power', r'hardware\s+power',
    r'constant\s+power', r'\bsleep\b', r'\bidle\b', r'quiescent', r'standby',
    r'duty[- ]cycle', r'battery\s+life', r'battery\s+lifetime', r'device\s+lifetime',
    r'\bP_?\{?c\}?\b', r'\bP_?\{?circ', r'\bP_?\{?static',
]
# Exclusion markers for stage 2, matched against title + abstract.
SURVEY_MARKERS = [r'\bsurvey\b', r'\breview\b', r'\boverview\b', r'\broadmap\b',
                  r'\bstandardi[sz]ation\b', r'\btutorial\b', r'\bvision\b',
                  # Non-Latin equivalents. P24's title is Ukrainian and opens with
                  # "Огляд" (overview); an English-only marker list carried it into
                  # the system-model denominator, which is a language bias in the
                  # screen and is reported as such in the paper's method.
                  r'огляд', r'обзор', r'\u7efc\u8ff0']


def norm_title(t):
    t = (t or '').lower()
    t = re.sub(r'[^a-z0-9 ]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


def load():
    with open(MANIFEST, encoding='utf-8') as fh:
        return json.load(fh)


def stage1_dedup(works):
    """Group by normalised title; keep the most citable record of each group."""
    groups = {}
    for w in works:
        groups.setdefault(norm_title(w['title'])[:90], []).append(w)
    kept, dups = [], []
    for key, g in groups.items():
        if len(g) == 1:
            kept.append(g[0])
            continue
        # prefer a record with a DOI and a real venue over a preprint
        g_sorted = sorted(g, key=lambda w: (bool(w['doi']),
                                            'arxiv' not in (w['venue'] or '').lower(),
                                            w['year'] or 0), reverse=True)
        kept.append(g_sorted[0])
        for d in g_sorted[1:]:
            dups.append((d['slug'], g_sorted[0]['slug'], d['venue']))
    kept.sort(key=lambda w: w['n'])
    return kept, dups


def stage2_include(works):
    inc, exc = [], []
    for w in works:
        hay = (w['title'] + ' ' + (w.get('abstract') or '')).lower()
        hit = [p for p in SURVEY_MARKERS if re.search(p, hay)]
        if hit:
            exc.append((w['slug'], w['title'][:64], hit[0].strip(r'\b')))
        else:
            inc.append(w)
    return inc, exc


def read_pdf(path):
    """Extract text. Returns None if no extractor is available or it fails."""
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            return None
    try:
        r = PdfReader(path)
        return "\n".join((p.extract_text() or '') for p in r.pages)
    except Exception as e:
        print("      ! extraction failed: %s" % e)
        return ''


def stage3_fulltext(works):
    have, missing, results = [], [], {}
    for w in works:
        p = os.path.join(LITDIR, w['slug'] + '.pdf')
        (have if os.path.exists(p) else missing).append(w)
    if not have:
        return results, missing

    print()
    print("=" * 100)
    print("  STAGE 3   full-text screen: does the system model carry a standing charge?")
    print("=" * 100)
    for w in have:
        txt = read_pdf(os.path.join(LITDIR, w['slug'] + '.pdf'))
        if txt is None:
            print("  no PDF text extractor available (pip install pypdf); stage 3 skipped")
            return results, missing + have
        low = txt.lower()
        hits = {}
        for pat in SC_PATTERNS:
            for m in re.finditer(pat, low):
                ctx = re.sub(r'\s+', ' ', low[max(0, m.start() - 70):m.end() + 70])
                hits.setdefault(pat, []).append(ctx)
        results[w['slug']] = {
            'title': w['title'], 'doi': w['doi'], 'chars': len(txt),
            'n_terms_hit': len(hits),
            'hits': {k: v[:3] for k, v in hits.items()},
            'carries_standing_charge': bool(hits),
        }
        print("\n  %-5s %-58s  %s" % (w['slug'], w['title'][:58],
                                      "STANDING CHARGE PRESENT" if hits else "none found"))
        if len(txt) < 4000:
            print("        ! only %d chars extracted - check the PDF is text, not a scan"
                  % len(txt))
        for k, v in list(hits.items())[:4]:
            print("        %-18s %s" % (k[:18], v[0][:96]))
    return results, missing


def main():
    d = load()
    works = d['works']
    print("=" * 100)
    print("  IoT-FACING CORPUS SCREEN")
    print("=" * 100)
    print("  manifest retrieved %s, %d records" % (d['_method']['retrieved'], len(works)))

    kept, dups = stage1_dedup(works)
    print()
    print("  STAGE 1  deduplication: %d records -> %d distinct works" % (len(works), len(kept)))
    for a, b, venue in dups:
        print("      %s is a duplicate of %s   (%s)" % (a, b, (venue or '')[:40]))

    inc, exc = stage2_include(kept)
    print()
    print("  STAGE 2  inclusion: %d distinct -> %d with a system model" % (len(kept), len(inc)))
    print("      excluded as survey / review / roadmap (no system model to carry a term):")
    for slug, title, why in exc:
        print("      %-5s %-62s [%s]" % (slug, title, why))

    print()
    print("  INCLUDED, and to be read in full:")
    for w in inc:
        print("      %-5s %-4s %-52s %s" % (w['slug'], w['year'], w['title'][:52],
                                            (w['venue'] or '')[:26]))

    res, missing = stage3_fulltext(inc)

    summary = {'retrieved': len(works), 'distinct': len(kept),
               'duplicates': [{'dup': a, 'of': b} for a, b, _ in dups],
               'excluded': [{'slug': s, 'title': t, 'reason': r} for s, t, r in exc],
               'included': [w['slug'] for w in inc],
               'n_included': len(inc),
               'fulltext_screened': list(res.keys()),
               'fulltext_missing': [w['slug'] for w in missing],
               'results': res}
    if res:
        with_sc = [k for k, v in res.items() if v['carries_standing_charge']]
        summary['n_with_standing_charge'] = len(with_sc)
        summary['with_standing_charge'] = with_sc
        print()
        print("=" * 100)
        print("  RESULT   %d of %d screened papers carry a standing-charge term in full text"
              % (len(with_sc), len(res)))
        print("=" * 100)
        print("  Every hit above must be inspected by hand before it is counted: a paper")
        print("  that says 'idle' about a channel is not modelling a standing charge.")
    else:
        print()
        print("  STAGE 3 not run: no PDFs found in docs/lit_corpus/.")
        print("  Save each as <slug>.pdf (P01.pdf, P03.pdf, ...) and re-run.")
        print("  %d papers to retrieve." % len(missing))

    dest = os.path.join(RESULTS, 'corpus_iot_screen.json')
    with open(dest, 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print("\nSaved %s" % dest)


if __name__ == '__main__':
    main()
