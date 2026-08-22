"""
Corpus robustness: does the result survive a change of index?

THE OBJECTION. The paper's corpus claim rests on a single index. OpenAlex is
broad, but a systematic-literature claim resting on one retrieval source invites
the obvious question: would Scopus, or Crossref, or anything else, say something
different? We cannot answer for Scopus (no licence) but we can answer for two
other open indexes, and an answer from three sources is a different kind of
claim from an answer from one.

THE DESIGN, and this is the part that makes the comparison mean anything.
Indexes differ in two ways at once: what they retrieve, and what metadata they
carry. If we let each index apply its own boolean query we confound those. So:

    RETRIEVAL varies by index. SCREENING is identical code for all three.

Each index is asked for a broad ISAC candidate set. Every record then passes
through the same local term screen, run over title+abstract, using the same term
lists as the published measurement. Differences in the final numbers are then
attributable to coverage rather than to query syntax.

THREE CONTROLS, without which the comparison is worthless:

  1. ABSTRACT COVERAGE. A standing-charge term cannot be found in an abstract
     that the index does not store. Crossref in particular carries abstracts for
     a minority of records. Coverage is reported per index, and any index with
     poor coverage has its null discounted explicitly rather than quietly.
  2. THE "ISAC" HOMONYM. ImmunoCAP ISAC is an allergy diagnostic. It appeared in
     our earlier novelty search and it will appear here. Records matching ISAC
     with no wireless term are counted and excluded, and the count is reported,
     because if it is large the published denominator of 360 is wrong.
  3. OVERLAP. Reported by DOI, so a reader can see whether three indexes are
     three samples or one sample counted three times.

WHAT WOULD FALSIFY THE PAPER'S CLAIM: an index in which the IoT-facing subset is
a large fraction of the ISAC energy corpus, or in which standing-charge terms
are common within it. The script is written to surface that.

Author: Vullnet Laniku
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results')
MAILTO = 'lanikuvullnet@gmail.com'
UA = 'academic-research (%s)' % MAILTO
FROM_YEAR = 2018

# ---------------------------------------------------------- the term screen --
# Identical for all three indexes. Applied to lower-cased title + abstract.
RE_ISAC = re.compile(r'integrated sensing and communication|\bisac\b|'
                     r'joint (?:communication|radar) and (?:sensing|communication)', re.I)
RE_ENERGY = re.compile(r'energy efficien|energy-efficien', re.I)
RE_IOT = re.compile(r'\biot\b|internet of things|sensor network|\blpwan\b|'
                    r'\bnb-iot\b|\blora\b|lorawan', re.I)
# a wireless anchor, to catch the ImmunoCAP ISAC homonym
RE_WIRELESS = re.compile(r'wireless|radar|beamform|spectrum|antenna|mimo|'
                         r'communication|network|6g|5g|base station|channel', re.I)

TX_TERMS = {
    'beamforming': r'beamform',
    'transmit power': r'transmit power',
    'power allocation': r'power allocation',
    'power control': r'power control',
    'waveform design': r'waveform design',
}
SC_TERMS = {
    'circuit power': r'circuit power',
    'static power': r'static power',
    'sleep': r'\bsleep\b',
    'idle': r'\bidle\b',
    'quiescent': r'quiescent',
    'standby': r'standby',
    'duty cycle': r'duty[- ]cycle',
    'battery life': r'battery life|battery lifetime',
}


def get(url, tries=6, headers=None):
    hdr = {'User-Agent': UA}
    if headers:
        hdr.update(headers)
    delay = 3.0
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=hdr)
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 503) or attempt == tries - 1:
                raise
            time.sleep(delay)
            delay *= 2
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(delay)
            delay *= 2


def norm(doi):
    if not doi:
        return None
    return doi.lower().replace('https://doi.org/', '').strip()


# ------------------------------------------------------------- OpenAlex -----
def fetch_openalex(cap=1200):
    q = '("integrated sensing and communication" OR "ISAC") AND "energy efficiency"'
    out, cursor = [], '*'
    while len(out) < cap:
        flt = ('title_and_abstract.search:%s,from_publication_date:%d-01-01'
               % (q, FROM_YEAR))
        url = ('https://api.openalex.org/works?filter=%s&per-page=200&cursor=%s&mailto=%s'
               % (urllib.parse.quote(flt, safe=':,'), cursor, urllib.parse.quote(MAILTO)))
        d = get(url)
        for w in d['results']:
            inv = w.get('abstract_inverted_index')
            ab = ''
            if inv:
                pos = {}
                for word, ps in inv.items():
                    for p in ps:
                        pos[p] = word
                ab = ' '.join(pos[k] for k in sorted(pos))
            out.append({'doi': norm(w.get('doi')), 'title': w.get('display_name') or '',
                        'abstract': ab, 'year': w.get('publication_year')})
        cursor = (d.get('meta') or {}).get('next_cursor')
        if not cursor or not d['results']:
            break
        time.sleep(1.0)
    return out


# ------------------------------------------------------------- Crossref -----
def fetch_crossref(cap=1200):
    out, cursor = [], '*'
    while len(out) < cap:
        url = ('https://api.crossref.org/works?query.bibliographic=%s'
               '&filter=from-pub-date:%d-01-01,type:journal-article'
               '&rows=200&cursor=%s&mailto=%s'
               % (urllib.parse.quote('integrated sensing and communication energy efficiency'),
                  FROM_YEAR, urllib.parse.quote(cursor), urllib.parse.quote(MAILTO)))
        d = get(url)
        msg = d['message']
        items = msg.get('items') or []
        for w in items:
            ab = re.sub(r'<[^>]+>', ' ', w.get('abstract') or '')
            out.append({'doi': norm(w.get('DOI')),
                        'title': ' '.join(w.get('title') or []),
                        'abstract': ab,
                        'year': ((w.get('issued') or {}).get('date-parts') or [[None]])[0][0]})
        cursor = msg.get('next-cursor')
        if not cursor or not items:
            break
        time.sleep(1.0)
    return out


# ------------------------------------------------- Semantic Scholar ---------
def fetch_s2(cap=1200):
    out, token = [], None
    base = ('https://api.semanticscholar.org/graph/v1/paper/search/bulk'
            '?query=%s&fields=title,abstract,year,externalIds&year=%d-'
            % (urllib.parse.quote('"integrated sensing and communication" + "energy efficiency"'),
               FROM_YEAR))
    while len(out) < cap:
        url = base + ('&token=%s' % token if token else '')
        d = get(url)
        for w in (d.get('data') or []):
            ext = w.get('externalIds') or {}
            out.append({'doi': norm(ext.get('DOI')), 'title': w.get('title') or '',
                        'abstract': w.get('abstract') or '', 'year': w.get('year')})
        token = d.get('token')
        if not token or not d.get('data'):
            break
        time.sleep(1.2)
    return out


# ------------------------------------------------------------- screening ----
def screen(records):
    """Identical screen for every index. Returns the per-index statistics."""
    seen, uniq = set(), []
    for r in records:
        key = r['doi'] or re.sub(r'[^a-z0-9]', '', (r['title'] or '').lower())[:80]
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(r)

    n_abs = sum(1 for r in uniq if len(r['abstract'] or '') > 40)
    isac, homonym = [], 0
    for r in uniq:
        hay = (r['title'] + ' ' + r['abstract'])
        if not RE_ISAC.search(hay):
            continue
        if not RE_WIRELESS.search(hay):
            homonym += 1
            continue
        isac.append(r)

    energy = [r for r in isac if RE_ENERGY.search(r['title'] + ' ' + r['abstract'])]
    iot = [r for r in energy if RE_IOT.search(r['title'] + ' ' + r['abstract'])]

    def hits(pool, terms):
        per = {}
        for name, pat in terms.items():
            rx = re.compile(pat, re.I)
            per[name] = sum(1 for r in pool if rx.search(r['title'] + ' ' + r['abstract']))
        anyhit = sum(1 for r in pool
                     if any(re.search(p, r['title'] + ' ' + r['abstract'], re.I)
                            for p in terms.values()))
        return per, anyhit

    tx_e, tx_any_e = hits(energy, TX_TERMS)
    sc_e, sc_any_e = hits(energy, SC_TERMS)
    tx_i, tx_any_i = hits(iot, TX_TERMS)
    sc_i, sc_any_i = hits(iot, SC_TERMS)

    return {
        'retrieved': len(records), 'unique': len(uniq),
        'abstract_coverage': n_abs / len(uniq) if uniq else 0.0,
        'isac_homonym_excluded': homonym,
        'isac': len(isac), 'isac_energy': len(energy), 'iot_facing': len(iot),
        'iot_share_of_energy': len(iot) / len(energy) if energy else None,
        'energy_tx_any': tx_any_e, 'energy_sc_any': sc_any_e,
        'iot_tx_any': tx_any_i, 'iot_sc_any': sc_any_i,
        'energy_tx_per': tx_e, 'energy_sc_per': sc_e,
        'iot_tx_per': tx_i, 'iot_sc_per': sc_i,
        'dois': sorted({r['doi'] for r in energy if r['doi']}),
    }


def main():
    sources = {}
    for name, fn in (('OpenAlex', fetch_openalex),
                     ('Crossref', fetch_crossref),
                     ('SemanticScholar', fetch_s2)):
        print("fetching %s ..." % name)
        try:
            recs = fn()
            sources[name] = screen(recs)
            print("   %d retrieved, %d unique, abstracts on %.0f%%"
                  % (sources[name]['retrieved'], sources[name]['unique'],
                     100 * sources[name]['abstract_coverage']))
        except Exception as e:
            sources[name] = {'error': str(e)}
            print("   FAILED: %s" % e)

    print()
    print("=" * 104)
    print("  CORPUS ROBUSTNESS ACROSS INDEXES  (identical local screen applied to all)")
    print("=" * 104)
    hdr = ('index', 'abs cov', 'ISAC', '+energy', '+IoT', 'IoT share', 'tx any', 'sc any')
    print("  %-16s %8s %8s %9s %7s %10s %8s %8s" % hdr)
    for name, r in sources.items():
        if 'error' in r:
            print("  %-16s  %s" % (name, r['error'][:70]))
            continue
        print("  %-16s %7.0f%% %8d %9d %7d %9s %8d %8d"
              % (name, 100 * r['abstract_coverage'], r['isac'], r['isac_energy'],
                 r['iot_facing'],
                 ('%.0f%%' % (100 * r['iot_share_of_energy'])) if r['iot_share_of_energy'] is not None else '-',
                 r['iot_tx_any'], r['iot_sc_any']))

    print()
    print("  CONTROLS")
    for name, r in sources.items():
        if 'error' in r:
            continue
        print("   %-16s ISAC homonym records excluded: %d   abstracts present: %.0f%%"
              % (name, r['isac_homonym_excluded'], 100 * r['abstract_coverage']))

    ok = {k: v for k, v in sources.items() if 'error' not in v}
    if len(ok) > 1:
        keys = list(ok)
        print()
        print("  OVERLAP by DOI over the ISAC+energy sets")
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = set(ok[keys[i]]['dois']), set(ok[keys[j]]['dois'])
                inter = len(a & b)
                print("   %-16s vs %-16s  %d and %d, shared %d (%.0f%% of the smaller)"
                      % (keys[i], keys[j], len(a), len(b), inter,
                         100 * inter / max(1, min(len(a), len(b)))))

    print()
    print("=" * 104)
    print("  VERDICT")
    print("=" * 104)
    shares = [v['iot_share_of_energy'] for v in ok.values()
              if v.get('iot_share_of_energy') is not None]
    scs = [v['iot_sc_any'] for v in ok.values()]
    if shares:
        print("  IoT-facing share of the ISAC energy corpus: %s"
              % ", ".join('%.0f%%' % (100 * x) for x in shares))
        print("  standing-charge hits inside the IoT subset: %s"
              % ", ".join(str(x) for x in scs))
        print()
        print("  The published claim survives a change of index only if the IoT share stays")
        print("  a small minority AND the standing-charge count stays near zero in every")
        print("  index whose abstract coverage is high enough to find one.")

    dest = os.path.join(RESULTS, 'corpus_multi_index.json')
    with open(dest, 'w', encoding='utf-8') as fh:
        json.dump({'_method': {'from_year': FROM_YEAR,
                               'run_date': time.strftime('%Y-%m-%d'),
                               'note': ('Retrieval differs by index; the term screen is '
                                        'identical code for all three, applied to '
                                        'title+abstract. Differences are therefore '
                                        'attributable to coverage, not query syntax.')},
                   'sources': sources}, fh, indent=2)
    print("\nSaved %s" % dest)


if __name__ == '__main__':
    main()
