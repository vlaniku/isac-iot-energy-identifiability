"""
The deeper RQ4_b pull: LPWAN battery/lifetime work against deployment evidence.

WHY THIS EXISTS. `results/rq4b_lorawan_battery_screen.json` is the record of how
the LPWAN energy-modelling papers discussed in Sec. II were found. It was
produced by a one-off script that was not kept, which left the release with a
result file no released code regenerates -- a gap in a paper that makes
reproducibility a contribution. This restores it.

WHAT IT IS. `systematic_search.py` runs RQ4_b at 25 records; the screen behind
Sec. II went deeper, to 75, because the question there is not a count but which
papers exist and what evidence each offers. The query string is identical and is
imported from `systematic_search` rather than restated, so the two cannot drift
apart.

WHAT IT IS NOT. This is a retrieval, not an adjudication. Which of these papers
model a standing charge, and how each calibrates it, was decided by reading them
-- see Sec. II and the full-text screen. Counts here drift as OpenAlex is
updated; the file records its own retrieval date, and the comparison against the
archived pull is printed so drift is visible rather than silent.

Author: Vullnet Laniku
"""

import json
import os
import time
import urllib.parse
import urllib.request

from systematic_search import QUERIES, FROM_YEAR, MAILTO, BASE, inv_abstract

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results')
ARCHIVED = os.path.join(RESULTS, 'rq4b_lorawan_battery_screen.json')

QID = 'RQ4_b'
TARGET = 75
PER_PAGE = 25


def fetch_page(q, page):
    flt = ('title_and_abstract.search:%s,from_publication_date:%d-01-01'
           % (q, FROM_YEAR))
    url = '%s?filter=%s&per-page=%d&page=%d&mailto=%s' % (
        BASE, urllib.parse.quote(flt, safe=':,'), PER_PAGE, page,
        urllib.parse.quote(MAILTO))
    req = urllib.request.Request(
        url, headers={'User-Agent': 'academic-research (%s)' % MAILTO})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.load(r)


def main():
    q = QUERIES[QID]
    print("=" * 96)
    print("  RQ4_b DEEP SCREEN   target %d records, %d per page" % (TARGET, PER_PAGE))
    print("=" * 96)
    print("  string: %s" % q)
    print()

    recs, page = [], 1
    while len(recs) < TARGET:
        d = fetch_page(q, page)
        got = d.get('results', [])
        if not got:
            break
        for w in got:
            recs.append({
                'title': w.get('display_name'),
                'year': w.get('publication_year'),
                'doi': w.get('doi'),
                'cited': w.get('cited_by_count'),
                'venue': ((w.get('primary_location') or {}).get('source')
                          or {}).get('display_name'),
                'abs': inv_abstract(w.get('abstract_inverted_index'))[:900]})
        print("  page %d: +%d  (total %d of %d reported)"
              % (page, len(got), len(recs), d['meta']['count']))
        page += 1
        time.sleep(0.4)
    recs = recs[:TARGET]

    # ------------------------------------------------ drift against archive --
    print()
    print("=" * 96)
    print("  DRIFT AGAINST THE ARCHIVED PULL")
    print("=" * 96)
    if os.path.exists(ARCHIVED):
        # The archived file was written without an explicit encoding and is not
        # valid UTF-8 -- it carries cp1252 smart quotes from paper titles. Read
        # it the way it was written rather than pretending it is clean.
        try:
            old = json.load(open(ARCHIVED, encoding='utf-8'))
        except UnicodeDecodeError:
            print("  NOTE: archived file is not UTF-8; reading as cp1252")
            old = json.load(open(ARCHIVED, encoding='cp1252'))
        od = {r.get('doi') for r in old if r.get('doi')}
        nd = {r.get('doi') for r in recs if r.get('doi')}
        print("  archived %d records, %d with a DOI" % (len(old), len(od)))
        print("  refetched %d records, %d with a DOI" % (len(recs), len(nd)))
        print("  in both        : %d" % len(od & nd))
        print("  archive only   : %d" % len(od - nd))
        print("  new since      : %d" % len(nd - od))
        overlap = len(od & nd) / max(len(od), 1)
        print("  overlap %.0f%% -- indexes are updated, so this is expected to "
              "drift" % (100 * overlap))
    else:
        overlap = None
        print("  no archived pull to compare against")

    out = os.path.join(RESULTS, 'rq4b_deep_screen.json')
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump({'_method': {
            'query_id': QID, 'string': q, 'from_year': FROM_YEAR,
            'target': TARGET, 'per_page': PER_PAGE,
            'run_date': time.strftime('%Y-%m-%d'),
            'source': 'OpenAlex title_and_abstract.search',
            'note': ('Retrieval only. Which papers model a standing charge was '
                     'decided by reading them, not here.'),
            'overlap_with_archive': overlap},
            'records': recs}, fh, indent=2, ensure_ascii=False)
    print()
    print("Saved %s" % out)


if __name__ == '__main__':
    main()
