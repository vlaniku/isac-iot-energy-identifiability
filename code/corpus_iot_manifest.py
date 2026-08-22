"""
Enumerate the IoT-facing ISAC energy-efficiency corpus and write a download
manifest, so the full-text screen (M3) reads a fixed, recorded population
rather than whatever a search happens to return on the day.

WHY A MANIFEST AND NOT A SEARCH. The whole point of the full-text pass is to
convert the corpus claim from SALIENCE (what a paper foregrounds in its
abstract) into PRESENCE (what its system model actually contains). That is only
worth anything if the population is frozen first. OpenAlex counts drift as
records are added; if the denominator moves between the screen and the write-up
the result is unciteable. So the manifest is written once, with the retrieval
date and the query, and the screen runs against it.

The manifest is also the download list: one row per paper, with DOI and venue,
for retrieval through an institutional subscription.

Author: Vullnet Laniku
"""

import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results')
LITDIR = os.path.join(HERE, '..', 'docs', 'lit_corpus')
MAILTO = 'lanikuvullnet@gmail.com'
BASE = 'https://api.openalex.org/works'
FROM_YEAR = 2018

ISAC = '("integrated sensing and communication" OR "ISAC")'
EE = '"energy efficiency"'
IOT = '("IoT" OR "Internet of Things" OR "sensor network" OR "LPWAN" OR "NB-IoT" OR "LoRa")'
QUERY = ISAC + ' AND ' + EE + ' AND ' + IOT

# Terms the full-text screen must look for. Salience was measured on title and
# abstract; presence must be measured on the whole text, including the system
# model and any table of simulation parameters.
SCREEN_TERMS = ['circuit power', 'static power', 'sleep', 'idle', 'quiescent',
                'standby', 'duty cycle', 'battery life', 'battery lifetime',
                'hardware power', 'constant power', 'P_c', 'P_circ']


def fetch(page, per_page=50, tries=6):
    flt = 'title_and_abstract.search:%s,from_publication_date:%d-01-01' % (QUERY, FROM_YEAR)
    url = '%s?filter=%s&per-page=%d&page=%d&mailto=%s' % (
        BASE, urllib.parse.quote(flt, safe=':,'), per_page, page,
        urllib.parse.quote(MAILTO))
    req = urllib.request.Request(url, headers={'User-Agent': 'academic-research (%s)' % MAILTO})
    delay = 2.0
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                time.sleep(1.1)
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == tries - 1:
                raise
            time.sleep(delay)
            delay *= 2


def inv_abstract(inv):
    if not inv:
        return ''
    out = {}
    for w, ps in inv.items():
        for p in ps:
            out[p] = w
    return ' '.join(out[k] for k in sorted(out))


def main():
    os.makedirs(LITDIR, exist_ok=True)
    d = fetch(1)
    n = d['meta']['count']
    works = list(d['results'])
    page = 2
    while len(works) < n:
        d = fetch(page)
        if not d['results']:
            break
        works += d['results']
        page += 1

    print("=" * 100)
    print("  IoT-FACING ISAC ENERGY-EFFICIENCY CORPUS — frozen manifest")
    print("=" * 100)
    print("  query      : %s" % QUERY)
    print("  date filter: from %d-01-01" % FROM_YEAR)
    print("  retrieved  : %s" % time.strftime('%Y-%m-%d'))
    print("  count      : %d (index reported %d)" % (len(works), n))
    print()

    rows = []
    for i, w in enumerate(works, 1):
        loc = (w.get('primary_location') or {})
        src = (loc.get('source') or {})
        bib = (w.get('biblio') or {})
        auth = [a['author']['display_name'] for a in (w.get('authorships') or [])]
        doi = (w.get('doi') or '').replace('https://doi.org/', '')
        oa = (w.get('open_access') or {})
        rows.append({
            'n': i,
            'slug': 'P%02d' % i,
            'title': w.get('display_name') or '',
            'first_author': auth[0] if auth else '',
            'n_authors': len(auth),
            'year': w.get('publication_year') or '',
            'venue': src.get('display_name') or '',
            'volume': bib.get('volume') or '',
            'pages': bib.get('first_page') or '',
            'doi': doi,
            'oa_status': oa.get('oa_status') or '',
            'oa_url': oa.get('oa_url') or '',
            'openalex_id': w.get('id') or '',
            'abstract': inv_abstract(w.get('abstract_inverted_index'))[:1200],
        })
        print("  %-5s %-4s %-58s %s"
              % (rows[-1]['slug'], rows[-1]['year'],
                 rows[-1]['title'][:58], rows[-1]['venue'][:30]))

    # ------------------------------------------------------------- outputs --
    with open(os.path.join(RESULTS, 'corpus_iot_manifest.json'), 'w', encoding='utf-8') as fh:
        json.dump({'_method': {'query': QUERY, 'from_year': FROM_YEAR,
                               'retrieved': time.strftime('%Y-%m-%d'),
                               'index_count': n, 'retrieved_count': len(works),
                               'screen_terms': SCREEN_TERMS,
                               'note': ('Population FROZEN on the retrieval date. The full-text '
                                        'screen must run against this list, not against a fresh '
                                        'search, so the denominator cannot move under the result.')},
                   'works': rows}, fh, indent=2, ensure_ascii=False)

    csv_path = os.path.join(LITDIR, 'DOWNLOAD_LIST.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as fh:
        wtr = csv.DictWriter(fh, fieldnames=['slug', 'year', 'first_author', 'title',
                                             'venue', 'volume', 'pages', 'doi',
                                             'oa_status', 'oa_url', 'downloaded'])
        wtr.writeheader()
        for r in rows:
            wtr.writerow({k: r.get(k, '') for k in
                          ['slug', 'year', 'first_author', 'title', 'venue',
                           'volume', 'pages', 'doi', 'oa_status', 'oa_url']} | {'downloaded': ''})

    print()
    print("  Saved ../results/corpus_iot_manifest.json")
    print("  Saved %s" % csv_path)
    n_oa = sum(1 for r in rows if r['oa_url'])
    print()
    print("  %d of %d have an open-access URL in the index; the rest need the" % (n_oa, len(rows)))
    print("  institutional subscription. Save each PDF into docs/lit_corpus/ as")
    print("  <slug>.pdf -- P01.pdf, P02.pdf, ... -- and the screen will pick them up.")


if __name__ == '__main__':
    main()
