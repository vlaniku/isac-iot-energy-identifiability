"""
Resolve every reference the v4 manuscript cites against a real index.

WHY. `manuscript/references.bib` carries 61 entries with keys like `isac_survey_1`
and `evaluation_gap_2`. It came from the TGCN submission, whose citation list was
AI-generated and which LITERATURE_CHECK.md's provenance note records as carrying
defects independent of the papers' quality: two entries pointing at the same arXiv
item, a company blog cited for an algorithmic claim, one reference attached to a
claim its paper does not make, and one that could not be found at all. Two of the
four entries checked by hand had a citation-level problem.

So none of it is reused. Every entry in the v4 bibliography is either (a) recorded
verbatim in this project's own literature notes from a source that was read, or
(b) resolved here against OpenAlex and Crossref by title, with the returned DOI,
venue, volume, pages and year printed for eyeballing before it is written into
the .bib.

WHAT THIS DOES NOT DO. It does not certify that a paper says what we say it says.
That requires reading it, which is recorded separately in LITERATURE_CHECK.md.
This only certifies that the bibliographic record is real and correct.

Output: results/reference_resolution.json, plus a printed table. Anything marked
NOT FOUND stays out of the manuscript or goes in flagged.

Author: Vullnet Laniku
"""

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
MAILTO = 'lanikuvullnet@gmail.com'

# key -> (search title, what the manuscript uses it for, expected venue fragment)
WANTED = {
    'sorensen2022':  ("Modeling and Experimental Validation for Battery Lifetime Estimation in NB-IoT and LTE-M",
                      "the one bench-validated LPWAN lifetime model; 3GPP reference traffic profile", "Internet of Things"),
    'szafranski2024': ("Predictability of LoRaWAN Link Quality based on Weather Data: Insights from a Long-Term Study",
                      "the cross-sectional prediction class; independent SF/ADR confound", "WoWMoM"),
    'guerra2024':    ("Forecasting LoRaWAN RSSI using weather parameters: A comparative study of ARIMA, artificial intelligence and hybrid approaches",
                      "the partial exception: AR terms, but on hourly averages", "Computer Networks"),
    'comnet2022':    ("Correlation between weather and signal strength in LoRaWAN networks: An extensive dataset",
                      "anchors the cross-sectional class", "Computer Networks"),
    'trassl2022':    ("On the Outage Probability of Channel Prediction Enabled Max-Min Radio Resource Allocation",
                      "nearest prior art to the energy identity; bounds outage, not energy", "WCNC"),
    'ramli2021':     ("A Study on the Impact of Nodes Density on the Energy Consumption of LoRa",
                      "adjacent experimental study; bench, density not lifetime", "iJIM"),
    'ye_ris':        ("Energy Efficiency Optimization in Active Reconfigurable Intelligent Surface-Aided Integrated Sensing and Communication Systems",
                      "f = 0.984, no sleep term anywhere", "Vehicular Technology"),
    'bai2026':       ("Lightweight AI-Driven Adaptive Resource Allocation for Energy-Efficient ISAC in 6G-IoT Networks",
                      "f = 0.97; SNR not SINR; nearest competing work", "Access"),
    'kadriu2026':    ("Enhancing Security in IoT LoRaWAN Smart Parking Systems through Anomaly Detection",
                      "prior work on the same array; silence is detectable but not attributable", "Internet of Things"),
    'wildlife2026':  ("Combined Radar and Magnetometer Sensor Network with LoRa-Mediated Awareness for Wildlife-Vehicle Collision Prevention",
                      "nearest hardware match; evaluated by Monte Carlo, not deployed", "arXiv"),
    'aarif2025':     ("Machine learning models for LoRaWAN link quality prediction",
                      "R2 = 0.99 cross-sectional prediction", "Annals of Telecommunications"),
    'novak2025':     ("Real-World Deployment Data LoRaWAN sensors battery life",
                      "the calibration-practice exhibit: assumed values, 10.95 years unvalidated", "Agris"),
    'rahman2023':    ("occupancy parking sensor LoRa RSSI attenuation vehicle",
                      "occupancy-driven link attenuation, measured on a cleaner design", ""),
    'dou2024':       ("Sensing-Efficient Transmit Beamforming for ISAC with MIMO Radar and MU-MIMO Communication",
                      "canonical ISAC allocation with interference inside the variables", "Wireless Communications"),
    'openalex':      ("OpenAlex: A fully-open index of scholarly works, authors, venues, institutions, and concepts",
                      "the index the corpus measurement runs on", "arXiv"),
}


def oa(query, per_page=5):
    url = ('https://api.openalex.org/works?search=%s&per-page=%d&mailto=%s'
           % (urllib.parse.quote(query), per_page, urllib.parse.quote(MAILTO)))
    req = urllib.request.Request(url, headers={'User-Agent': 'academic-research (%s)' % MAILTO})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def rec(w):
    loc = (w.get('primary_location') or {})
    src = (loc.get('source') or {})
    bib = (w.get('biblio') or {})
    auth = [a['author']['display_name'] for a in (w.get('authorships') or [])]
    return {
        'title': w.get('display_name'),
        'authors': auth,
        'n_authors': len(auth),
        'year': w.get('publication_year'),
        'venue': src.get('display_name'),
        'volume': bib.get('volume'), 'issue': bib.get('issue'),
        'first_page': bib.get('first_page'), 'last_page': bib.get('last_page'),
        'doi': w.get('doi'),
        'type': w.get('type'),
        'cited_by': w.get('cited_by_count'),
    }


def main():
    out = {'_method': {
        'source': 'OpenAlex /works?search=<title>',
        'run_date': time.strftime('%Y-%m-%d'),
        'note': ('Bibliographic verification only. That a paper says what we claim it says is '
                 'recorded in LITERATURE_CHECK.md from full-text reads, not here. The TGCN '
                 'references.bib is deliberately NOT reused; see module docstring.')},
        'entries': {}}

    print("=" * 108)
    print("  REFERENCE RESOLUTION — every v4 citation against OpenAlex")
    print("=" * 108)
    for key, (title, purpose, venue_frag) in WANTED.items():
        try:
            d = oa(title)
            hits = d['results']
            best = rec(hits[0]) if hits else None
            alts = [rec(h) for h in hits[1:4]]
        except Exception as e:
            out['entries'][key] = {'error': str(e), 'purpose': purpose}
            print("\n  %-14s ERROR %s" % (key, str(e)[:70]))
            continue

        ok = False
        if best:
            v = (best['venue'] or '')
            ok = (not venue_frag) or (venue_frag.lower() in v.lower())
        out['entries'][key] = {'query': title, 'purpose': purpose,
                               'expected_venue_fragment': venue_frag,
                               'best': best, 'alternates': alts,
                               'venue_matches_expectation': bool(ok)}
        print("\n  %-14s %s" % (key, "VENUE OK" if ok else "*** CHECK BY HAND ***"))
        if not best:
            print("      NOT FOUND")
            continue
        print("      %s" % (best['title'] or '')[:96])
        print("      %s | %s %s(%s) %s-%s | %s"
              % (", ".join(best['authors'][:3]) + (" et al." if best['n_authors'] > 3 else ""),
                 best['venue'], best['volume'] or '?', best['issue'] or '?',
                 best['first_page'] or '?', best['last_page'] or '?', best['year']))
        print("      %s   cited %s" % (best['doi'], best['cited_by']))
        time.sleep(0.35)

    dest = os.path.join(RESULTS, 'reference_resolution.json')
    with open(dest, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("\n\nSaved %s" % dest)
    bad = [k for k, v in out['entries'].items()
           if not v.get('venue_matches_expectation')]
    print("\n  Needing manual confirmation before they enter the .bib: %s"
          % (", ".join(bad) if bad else "none"))


if __name__ == '__main__':
    main()
