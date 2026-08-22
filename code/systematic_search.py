"""
Systematic literature search — recorded strings, reproducible counts.

WHY. LITERATURE_CHECK.md rests on nine ad-hoc web queries, and STATE.md 5.9 flags
that as an open item because the channel-predictability novelty of Sec V-E is
load-bearing. A reviewer is entitled to a method, not a list of things we
happened to find.

WHAT THIS IS AND IS NOT. Scopus requires an institutional licence and is not
available here. This uses OpenAlex (~250M indexed works, includes the IEEE,
Springer, Elsevier and ACM corpora) and Crossref as a cross-check. Coverage
differs from Scopus in detail, not in kind, for this subject area. Every string
and every count is recorded so the search can be re-run or replaced with a Scopus
run using the same strings.

Screening is by title and abstract only. Anything that survives screening must be
read in full before it is cited -- that step needs institutional access and is
handed off, not automated.

RESEARCH QUESTIONS

  RQ1  Has anyone measured TEMPORAL predictability of LPWAN link quality from a
       node's own history, at its own decision cadence?          <- Sec V-E, load-bearing
  RQ2  Has anyone used channel predictability to BOUND what adaptive resource
       allocation can achieve?                                    <- Sec V-A, Sec VI
  RQ3  Is there a regime analysis of when adaptation beats a static choice?  <- Sec VI
  RQ4  Has anyone validated an LPWAN energy model against deployed telemetry
       rather than a lab testbed?                                 <- Sec II-C, Sec VII
  RQ5  How many ISAC energy-efficiency papers model a sleep / idle / quiescent
       term at all?                                               <- Sec VI, quantifies the claim

RQ5 is the one that can produce a number rather than an absence, and that number
is a direct measurement of the paper's central assertion.

Author: Vullnet Laniku
"""

import json
import os
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results')
MAILTO = 'lanikuvullnet@gmail.com'
BASE = 'https://api.openalex.org/works'

# ------------------------------------------------------------ the strings ----
# OpenAlex `title_and_abstract.search` accepts AND / OR / NOT and quoted phrases.
QUERIES = {
    'RQ1_a': '("LoRaWAN" OR "LoRa" OR "LPWAN") AND ("RSSI" OR "link quality") AND ("autocorrelation" OR "temporal correlation")',
    'RQ1_b': '("LoRaWAN" OR "LPWAN") AND ("channel prediction" OR "link prediction") AND ("time series" OR "temporal")',
    'RQ1_c': '("LoRa" OR "LPWAN") AND "coherence time"',
    'RQ1_d': '("LoRaWAN" OR "LPWAN") AND "RSSI" AND ("forecast" OR "predictability")',
    'RQ2_a': '"resource allocation" AND ("predictability" OR "channel prediction") AND ("upper bound" OR "fundamental limit")',
    'RQ2_b': '("adaptive" AND "allocation") AND "value of information" AND wireless',
    'RQ3_a': '"static allocation" AND "adaptive allocation" AND (energy OR "energy efficiency")',
    'RQ3_b': '("when" AND "adaptation") AND ("regime" OR "condition") AND "resource allocation" AND wireless',
    'RQ4_a': '("LPWAN" OR "LoRaWAN") AND "energy model" AND ("validation" OR "validated" OR "measurement")',
    'RQ4_b': '("LoRaWAN" OR "LPWAN") AND "battery" AND ("deployment" OR "field") AND ("lifetime" OR "depletion")',
    'RQ5_a': '("integrated sensing and communication" OR "ISAC") AND "energy efficiency"',
    'RQ5_b': '("integrated sensing and communication" OR "ISAC") AND "energy efficiency" AND ("sleep" OR "idle" OR "quiescent" OR "duty cycle")',
    'RQ5_c': '("integrated sensing and communication" OR "ISAC") AND ("battery life" OR "battery lifetime")',
}

FROM_YEAR = 2018


def fetch(q, per_page=25, page=1):
    flt = 'title_and_abstract.search:%s,from_publication_date:%d-01-01' % (q, FROM_YEAR)
    url = '%s?filter=%s&per-page=%d&page=%d&mailto=%s' % (
        BASE, urllib.parse.quote(flt, safe=':,'), per_page, page,
        urllib.parse.quote(MAILTO))
    req = urllib.request.Request(url, headers={'User-Agent': 'academic-research (%s)' % MAILTO})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.load(r)


def inv_abstract(inv):
    """OpenAlex stores abstracts as an inverted index."""
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
        'note': ('Scopus unavailable (institutional licence). OpenAlex indexes the IEEE, '
                 'Springer, Elsevier and ACM corpora. Strings are recorded verbatim so the '
                 'search can be re-run on Scopus unchanged.'),
        'screening': 'title and abstract only; full text required before citation'},
        'queries': {}}

    print("=" * 96)
    print("  SYSTEMATIC SEARCH — OpenAlex, title+abstract, %d onwards" % FROM_YEAR)
    print("=" * 96)
    print("  %-8s %8s   %s" % ("id", "hits", "string"))
    for qid, q in QUERIES.items():
        try:
            d = fetch(q, per_page=25)
            n = d['meta']['count']
            items = [{'title': w.get('display_name'),
                      'year': w.get('publication_year'),
                      'doi': w.get('doi'),
                      'venue': ((w.get('primary_location') or {}).get('source') or {}).get('display_name'),
                      'cited_by': w.get('cited_by_count'),
                      'abstract': inv_abstract(w.get('abstract_inverted_index'))[:900]}
                     for w in d['results']]
            rec['queries'][qid] = {'string': q, 'hits': n, 'retrieved': len(items),
                                   'results': items}
            print("  %-8s %8d   %s" % (qid, n, q[:78]))
        except Exception as e:
            rec['queries'][qid] = {'string': q, 'error': str(e)}
            print("  %-8s %8s   %s" % (qid, 'ERR', str(e)[:60]))
        time.sleep(0.4)

    out = os.path.join(RESULTS, 'systematic_search_raw.json')
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(rec, fh, indent=2, ensure_ascii=False)
    print("\nSaved %s" % out)

    # -------------------------------------------------- the RQ5 arithmetic ---
    a = rec['queries'].get('RQ5_a', {}).get('hits')
    b = rec['queries'].get('RQ5_b', {}).get('hits')
    c = rec['queries'].get('RQ5_c', {}).get('hits')
    if a and b is not None:
        print()
        print("=" * 96)
        print("  RQ5 — the one that yields a number")
        print("=" * 96)
        print("  ISAC + energy efficiency                              : %5d" % a)
        print("  ...of which mention sleep / idle / quiescent / duty   : %5d  (%.1f%%)"
              % (b, 100.0 * b / a))
        print("  ...that mention battery life at all                   : %5d  (%.1f%%)"
              % (c, 100.0 * c / a))
        print()
        print("  This is a screening-level count, not a full-text audit: a mention is")
        print("  not a model. It bounds the claim from above -- the true fraction that")
        print("  MODEL a standing-charge term is at most this.")


if __name__ == '__main__':
    main()
