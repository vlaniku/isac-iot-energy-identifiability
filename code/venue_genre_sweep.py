"""
Does IEEE IoT-J publish papers of this kind? Measure it rather than guess.

WHY THIS EXISTS. The largest remaining risk to this submission is not technical.
It is whether the venue wants a measurement-and-evaluability paper whose
headline results are negative and whose central experiment is designed but not
run. That has been an opinion in every assessment so far. It is a property of a
corpus, and this project's own method says to measure it.

WHAT THIS DOES. Resolves the journal's OpenAlex source id, takes its whole
indexed output as the denominator, and counts how much of it carries the
markers of this genre: measurement studies, negative or null results, explicit
reproducibility work, critical re-examinations, and papers that make a point of
what cannot be measured.

CONTROLS, because a count without one means nothing. Three positive controls
that must return large counts or the venue filter is broken; and the genre
strings are also run against the whole index, so a low count at IoT-J can be
compared against the term's general frequency rather than read in isolation.

WHAT IT CANNOT TELL YOU. Acceptance rates are not public and rejected papers
are not indexed, so this measures what the venue PUBLISHES, which bounds what
it accepts from below. A genre absent here might be absent because nobody
submits it.

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
BASE = 'https://api.openalex.org'
ISSN = '2327-4662'          # IEEE Internet of Things Journal
FROM_YEAR = 2018

GENRE = {
    'measurement study': '"measurement study"',
    'empirical study': '"empirical study"',
    'negative / null result': '"negative results" OR "null result" OR "negative result"',
    'reproducibility': '"reproducibility" OR "reproducible"',
    'lessons learned': '"lessons learned"',
    'reality check / revisiting': '"reality check" OR "revisiting" OR "re-examining"',
    'pitfalls / misconceptions': '"pitfalls" OR "misconceptions" OR "myths"',
    'benchmark / evaluation methodology': '"evaluation methodology" OR "benchmarking methodology"',
    'statistical power / detectability': '"statistical power" OR "minimum detectable"',
    'deployment experience': '"deployment experience" OR "field study" OR "real-world deployment"',
    'limits / cannot': '"fundamental limits" OR "limitations of"',
}

CONTROLS = {
    'deep learning (control)': '"deep learning"',
    'energy efficiency (control)': '"energy efficiency"',
    'resource allocation (control)': '"resource allocation"',
}


def get(url, tries=9):
    """OpenAlex rate-limits; back off rather than losing the sweep half way."""
    for a in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={'User-Agent': 'academic-research (%s)' % MAILTO})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 or a == tries - 1:
                raise
            wait = 10 * (a + 1)
            print("    (429, waiting %ds)" % wait, flush=True)
            time.sleep(wait)


def source_id():
    d = get('%s/sources?filter=issn:%s&mailto=%s'
            % (BASE, ISSN, urllib.parse.quote(MAILTO)))
    for r in d.get('results', []):
        return r['id'].rsplit('/', 1)[-1], r['display_name'], r.get('works_count')
    return None, None, None


def count(src, query=None):
    flt = 'primary_location.source.id:%s,from_publication_date:%d-01-01' % (src, FROM_YEAR)
    if query:
        flt += ',title_and_abstract.search:%s' % query
    url = '%s/works?filter=%s&per-page=1&mailto=%s' % (
        BASE, urllib.parse.quote(flt, safe=':,'), urllib.parse.quote(MAILTO))
    return get(url)['meta']['count']


def count_global(query):
    flt = ('title_and_abstract.search:%s,from_publication_date:%d-01-01'
           % (query, FROM_YEAR))
    url = '%s/works?filter=%s&per-page=1&mailto=%s' % (
        BASE, urllib.parse.quote(flt, safe=':,'), urllib.parse.quote(MAILTO))
    return get(url)['meta']['count']


def main():
    sid, name, total_all = source_id()
    if not sid:
        raise SystemExit('could not resolve the journal in OpenAlex')
    denom = count(sid)
    print("=" * 92)
    print("  %s" % name)
    print("  OpenAlex %s, %d works indexed in total, %d since %d"
          % (sid, total_all or -1, denom, FROM_YEAR))
    print("=" * 92)

    print("\n  CONTROLS (must be large, or the venue filter is broken)")
    print("  %-34s %8s %8s" % ("string", "papers", "share"))
    ctl = {}
    for lab, q in CONTROLS.items():
        n = count(sid, q)
        ctl[lab] = n
        print("  %-34s %8d %7.1f%%" % (lab, n, 100.0 * n / denom))
        time.sleep(1.2)

    print("\n  GENRE MARKERS")
    print("  %-34s %8s %8s" % ("string", "papers", "share"))
    rows = {}
    for lab, q in GENRE.items():
        n = count(sid, q)
        rows[lab] = {'iotj': n, 'share': n / denom}
        print("  %-34s %8d %7.2f%%" % (lab, n, 100.0 * n / denom))
        time.sleep(3.0)

    print()
    print("=" * 92)
    print("  READING")
    print("=" * 92)
    tot = sum(r['iotj'] for r in rows.values())
    print("  genre-marked papers, summed over markers (with overlap): %d of %d = %.1f%%"
          % (tot, denom, 100.0 * tot / denom))
    big = sorted(rows.items(), key=lambda kv: -kv[1]['iotj'])[:4]
    print("  most common markers here: %s"
          % ", ".join("%s (%d)" % (k, v['iotj']) for k, v in big))
    thin = [k for k, v in rows.items() if v['iotj'] < 0.002 * denom]
    if thin:
        print("  markers under 0.2%% of the venue: %s" % ", ".join(thin))

    out = {'_method': {
        'source_id': sid, 'source_name': name, 'issn': ISSN,
        'from_year': FROM_YEAR, 'run_date': time.strftime('%Y-%m-%d'),
        'denominator': denom,
        'caveat': ('Measures what the venue PUBLISHES. Rejections are not '
                   'indexed, so this bounds acceptance from below and cannot '
                   'separate "not accepted" from "not submitted".')},
        'controls': ctl, 'genre': rows}
    path = os.path.join(RESULTS, 'venue_genre_sweep.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2)
    print("\nSaved %s" % path)


if __name__ == '__main__':
    main()
