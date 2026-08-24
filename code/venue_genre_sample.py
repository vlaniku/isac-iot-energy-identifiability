"""
The same question as venue_genre_sweep.py, answered from a random sample.

WHY BOTH. The sweep asks OpenAlex for one filtered count per genre string, which
is exact but costs a request each and is heavily rate-limited. This pulls a
random sample of the journal's output in a handful of requests and does the
term matching locally. It is less precise on rare genres, but it returns the
TITLES, which answer the question better than a percentage does: the point is
not what share of IoT-J is measurement work, it is whether a paper shaped like
this one has ever got in.

Author: Vullnet Laniku
"""

import json
import os
import re
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results')
MAILTO = 'lanikuvullnet@gmail.com'
BASE = 'https://api.openalex.org/works'
SRC = 'S2480266640'          # IEEE Internet of Things Journal
FROM_YEAR = 2018
SAMPLE = 2000
SEED = 42

GENRE = {
    'measurement / empirical study': r'measurement study|empirical study|measurement campaign',
    'negative or null result': r'negative result|null result|no significant|fails to|does not improve',
    'reproducibility': r'reproducib|replicat',
    'lessons / experience': r'lessons learned|deployment experience|experience report',
    'critical re-examination': r'reality check|revisit|re-examin|rethink|myth|pitfall|misconcept',
    'evaluation methodology': r'evaluation methodolog|benchmarking methodolog|how to evaluate',
    'statistical power / detectability': r'statistical power|minimum detectable|detectability limit',
    'limits of what can be measured': r'fundamental limit|limitation of|cannot be (measured|estimated|identified)',
}


def get(url, tries=8):
    for a in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={'User-Agent': 'academic-research (%s)' % MAILTO})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 or a == tries - 1:
                raise
            w = 8 * (a + 1)
            print("    (429, waiting %ds)" % w, flush=True)
            time.sleep(w)


def inv_abstract(inv):
    if not inv:
        return ''
    out = {}
    for w, ps in inv.items():
        for p in ps:
            out[p] = w
    return ' '.join(out[k] for k in sorted(out))


def main():
    flt = ('primary_location.source.id:%s,from_publication_date:%d-01-01'
           % (SRC, FROM_YEAR))
    # OpenAlex sampling does not support cursor paging; it pages with `page`,
    # up to sample/per-page. Using a cursor silently returns one page, which is
    # how the first run came back with 200 records and uninformative zeros.
    recs, page = [], 1
    while len(recs) < SAMPLE:
        url = ('%s?filter=%s&sample=%d&seed=%d&per-page=200&page=%d&mailto=%s'
               % (BASE, urllib.parse.quote(flt, safe=':,'), SAMPLE, SEED,
                  page, urllib.parse.quote(MAILTO)))
        d = get(url)
        got = d.get('results', [])
        if not got:
            break
        for w in got:
            recs.append({'title': w.get('display_name') or '',
                         'year': w.get('publication_year'),
                         'abs': inv_abstract(w.get('abstract_inverted_index'))})
        print("  fetched %d" % len(recs), flush=True)
        page += 1
        time.sleep(2.0)

    with_abs = [r for r in recs if r['abs']]
    print()
    print("=" * 94)
    print("  IEEE IoT-J, random sample of %d works from %d onward (%d with abstracts)"
          % (len(recs), FROM_YEAR, len(with_abs)))
    print("=" * 94)
    print("  %-38s %7s %8s   %s" % ("genre marker", "hits", "share", "example title"))

    out = {}
    for lab, pat in GENRE.items():
        rx = re.compile(pat, re.I)
        hits = [r for r in with_abs
                if rx.search(r['title']) or rx.search(r['abs'])]
        ex = hits[0]['title'][:44] + '...' if hits else '--'
        out[lab] = {'hits': len(hits), 'share': len(hits) / max(len(with_abs), 1),
                    'examples': [h['title'] for h in hits[:6]]}
        print("  %-38s %7d %7.1f%%   %s"
              % (lab, len(hits), 100 * len(hits) / max(len(with_abs), 1), ex))

    print()
    print("=" * 94)
    print("  TITLES, for the markers closest to this paper")
    print("=" * 94)
    for lab in ('measurement / empirical study', 'negative or null result',
                'critical re-examination', 'limits of what can be measured'):
        print("\n  %s:" % lab)
        for t in out[lab]['examples']:
            print("    - %s" % t[:96])

    res = {'_method': {'source': SRC, 'sample': len(recs), 'seed': SEED,
                       'from_year': FROM_YEAR, 'with_abstract': len(with_abs),
                       'run_date': time.strftime('%Y-%m-%d'),
                       'note': ('Random sample, local term matching. Shares are '
                                'estimates with sampling error; the titles are '
                                'the point.')},
           'genre': out}
    p = os.path.join(RESULTS, 'venue_genre_sample.json')
    with open(p, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print("\nSaved %s" % p)


if __name__ == '__main__':
    main()
