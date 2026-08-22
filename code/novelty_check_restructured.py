"""
Novelty audit for the RESTRUCTURED paper's claims.

WHY THIS AND NOT `systematic_search.py`. That script asked whether the
*deployment-led* paper was novel: temporal link predictability at a node's own
decision cadence (RQ1), predictability as a bound on allocation (RQ2), regime
analysis (RQ3), energy-model validation against deployed telemetry (RQ4), and
the corpus composition itself (RQ5). The paper's centre of gravity has since
moved. Its lead claims are now:

  N1  a CORPUS MEASUREMENT of what the ISAC energy literature foregrounds as its
      energy variable, against what a battery-powered device is dominated by
  N2  an EVALUABILITY result: that quantity cannot presently be measured from
      deployed telemetry, demonstrated by controlled negatives across four
      populations with the power limits quantified
  N3  an INSTRUMENTATION SPECIFICATION derived from which analysis each missing
      quantity blocked

None of those three was searched. This script searches them, plus two supporting
claims the restructure promotes (the field service record itself, and the use of
power analysis in IoT energy studies), so that no lead claim rests on an
unsearched assumption.

RULE OF THIS PROJECT, restated because it applies with special force here: a
count is not a verdict. A low count means the string found little, which can mean
the idea is novel or that the string was wrong. Every string below is therefore
paired with a CONTROL string that must return a healthy count, and the returned
titles are printed so a null can be inspected rather than trusted.

Author: Vullnet Laniku
"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request

# The console here is cp1252 and paper titles are not. Without this the report
# crashes part-way through printing and takes the unsaved fetch with it.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results')
MAILTO = 'lanikuvullnet@gmail.com'
BASE = 'https://api.openalex.org/works'
FROM_YEAR = 2015          # wider than systematic_search.py: meta-research is older

QUERIES = {
    # -- N1: has the composition of an engineering literature's energy models
    #        ever been measured as a corpus, for ISAC or anything adjacent? ----
    'N1_a': '("integrated sensing and communication" OR "ISAC") AND ("bibliometric" OR "systematic review" OR "scoping review" OR "meta-analysis")',
    'N1_b': '("bibliometric" OR "systematic review" OR "meta-analysis") AND "energy efficiency" AND ("wireless" OR "6G" OR "IoT") AND ("system model" OR "modelling assumption" OR "modeling assumption")',
    'N1_c': '("survey" OR "review") AND ("energy model" OR "power model") AND ("circuit power" OR "static power" OR "sleep current") AND ("wireless" OR "IoT")',
    'N1_d': '("what the literature optimises" OR "what the literature optimizes" OR "modelling assumptions" OR "modeling assumptions") AND "energy" AND ("corpus" OR "papers")',

    # -- N2: evaluability / measurability of energy claims from field telemetry -
    'N2_a': '("IoT" OR "LPWAN" OR "wireless sensor") AND "telemetry" AND ("energy" OR "battery") AND ("instrumentation" OR "measurability" OR "observability")',
    'N2_b': '("energy" OR "battery") AND ("IoT" OR "wireless sensor network") AND ("cannot be measured" OR "not measurable" OR "unmeasurable" OR "untestable")',
    'N2_c': '("reproducibility" OR "replication crisis" OR "replication study") AND ("energy" OR "battery") AND ("IoT" OR "wireless sensor" OR "networking")',
    'N2_d': '("negative result" OR "null result" OR "failed to replicate") AND ("IoT" OR "LPWAN" OR "wireless sensor") AND ("energy" OR "battery")',

    # -- N3: instrumentation specifications derived from failed analyses -------
    'N3_a': '("instrumentation requirements" OR "measurement requirements" OR "data requirements") AND ("IoT" OR "LPWAN") AND ("energy" OR "battery" OR "lifetime")',
    'N3_b': '("what should be logged" OR "logging requirements" OR "telemetry design") AND ("IoT" OR "wireless sensor")',

    # -- supporting: the field service record ---------------------------------
    'S1_a': '("parking sensor" OR "parking occupancy") AND ("LoRaWAN" OR "LPWAN") AND ("battery life" OR "service life" OR "lifetime")',
    'S1_b': '"radar" AND "magnetometer" AND ("LoRaWAN" OR "LPWAN" OR "LoRa")',
    'S1_c': '("LoRaWAN" OR "LPWAN") AND ("multi-year" OR "two-year" OR "long-term") AND "deployment" AND ("battery" OR "lifetime") AND ("field" OR "operational")',

    # -- supporting: statistical power in IoT energy studies -------------------
    'S2_a': '("IoT" OR "wireless sensor" OR "LPWAN") AND ("statistical power" OR "minimum detectable effect" OR "power analysis") AND ("energy" OR "battery")',
    'S2_b': '("crossover design" OR "within-subject" OR "within-device") AND ("IoT" OR "LPWAN" OR "wireless sensor") AND ("energy" OR "battery")',

    # -- CONTROLS: strings that MUST return healthy counts. If a control comes
    #    back near zero the query syntax is broken and every null above is
    #    meaningless. This is the check that caught a malformed string in the
    #    20 Aug run of systematic_search.py.
    'CTRL_isac': '("integrated sensing and communication" OR "ISAC") AND "energy efficiency"',
    'CTRL_lora': '("LoRaWAN" OR "LPWAN") AND "energy"',
    'CTRL_biblio': '"bibliometric analysis"',
    'CTRL_power': '"statistical power"',
    'CTRL_replic': '"reproducibility"',
}

# A control must clear this to certify the strings around it.
CONTROL_FLOOR = 200


def fetch(q, per_page=25, page=1):
    flt = 'title_and_abstract.search:%s,from_publication_date:%d-01-01' % (q, FROM_YEAR)
    url = '%s?filter=%s&per-page=%d&page=%d&mailto=%s' % (
        BASE, urllib.parse.quote(flt, safe=':,'), per_page, page,
        urllib.parse.quote(MAILTO))
    req = urllib.request.Request(url, headers={'User-Agent': 'academic-research (%s)' % MAILTO})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def inv_abstract(inv):
    if not inv:
        return ''
    out = {}
    for w, ps in inv.items():
        for p in ps:
            out[p] = w
    return ' '.join(out[k] for k in sorted(out))


def main():
    rec = {'_method': {
        'source': 'OpenAlex REST API (api.openalex.org/works)',
        'field': 'title_and_abstract.search',
        'date_filter': 'from_publication_date >= %d-01-01' % FROM_YEAR,
        'run_date': time.strftime('%Y-%m-%d'),
        'control_floor': CONTROL_FLOOR,
        'note': ('Companion to systematic_search.py, which searched the claims of the '
                 'deployment-led paper. This searches the three lead claims of the '
                 'restructured one, plus two supporting claims. Controls are included '
                 'so a null can be distinguished from a broken string.'),
        'screening': 'title and abstract only; full text required before citation'},
        'queries': {}}

    print("=" * 100)
    print("  NOVELTY AUDIT — restructured paper, OpenAlex, title+abstract, %d onwards" % FROM_YEAR)
    print("=" * 100)
    print("  %-12s %8s   %s" % ("id", "hits", "string"))
    for qid, q in QUERIES.items():
        try:
            d = fetch(q, per_page=25)
            n = d['meta']['count']
            items = [{'title': w.get('display_name'),
                      'year': w.get('publication_year'),
                      'doi': w.get('doi'),
                      'venue': ((w.get('primary_location') or {}).get('source') or {}).get('display_name'),
                      'cited_by': w.get('cited_by_count'),
                      'abstract': inv_abstract(w.get('abstract_inverted_index'))[:700]}
                     for w in d['results']]
            rec['queries'][qid] = {'string': q, 'hits': n, 'retrieved': len(items),
                                   'results': items}
            print("  %-12s %8d   %s" % (qid, n, q[:74]))
        except Exception as e:
            rec['queries'][qid] = {'string': q, 'error': str(e)}
            print("  %-12s %8s   %s" % (qid, 'ERR', str(e)[:60]))
        time.sleep(0.4)

    # ------------------------------------------------- controls come first ---
    print()
    print("=" * 100)
    print("  CONTROLS — if these are not healthy, every null above is meaningless")
    print("=" * 100)
    bad = []
    for qid in [k for k in QUERIES if k.startswith('CTRL_')]:
        n = rec['queries'][qid].get('hits')
        ok = isinstance(n, int) and n >= CONTROL_FLOOR
        if not ok:
            bad.append(qid)
        print("  %-12s %8s   %s" % (qid, n, "OK" if ok else "*** SYNTAX SUSPECT ***"))
    rec['controls_ok'] = not bad
    if bad:
        print("\n  Controls failed: %s. Stop; do not read the nulls as novelty." % ", ".join(bad))

    # --------------------------------------------------------- the verdict ---
    print()
    print("=" * 100)
    print("  TITLES RETURNED BY THE LEAD-CLAIM STRINGS")
    print("=" * 100)
    for qid in [k for k in QUERIES if not k.startswith('CTRL_')]:
        e = rec['queries'][qid]
        if 'hits' not in e:
            continue
        print("\n  --- %s  (%d hits) ---" % (qid, e['hits']))
        if e['hits'] == 0:
            print("      (nothing)")
        for it in e['results'][:6]:
            print("      %-4s %-58s %s"
                  % (it['year'], (it['title'] or '')[:58], (it['venue'] or '')[:34]))

    out = os.path.join(RESULTS, 'novelty_check_restructured.json')
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(rec, fh, indent=2, ensure_ascii=False)
    print("\nSaved %s" % out)
    print()
    print("  NEXT STEP, and it is not optional: anything above with a plausible")
    print("  title must be read before the paper claims novelty against it.")
    print("  A count is evidence about a string, not about the literature.")


if __name__ == '__main__':
    main()
