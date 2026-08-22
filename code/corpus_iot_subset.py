"""
THE REVIEWER'S FIRST ATTACK, run against ourselves before a reviewer runs it.

THE OBJECTION. The paper's headline is that "the ISAC energy literature
foregrounds a transmit-side variable" at 35.7 to one. That ratio is measured
over a corpus defined as ISAC AND "energy efficiency" -- with no IoT term in
it. Most ISAC work is infrastructure-side: base stations, RIS, MIMO radar,
mains-powered, where transmit power genuinely IS the dominant energy term and
beamforming genuinely IS the right variable to optimise.

If that is what the 360 papers mostly are, then the 35.7 ratio is not evidence
that the field optimises the wrong variable. It is evidence that we measured
the wrong corpus, and the paper's central claim collapses into a category
error. This is exactly the failure mode that sank the earlier submission: a
number that is real, and an inference from it that the artifact does not
support.

WHAT THIS SCRIPT DOES. It partitions the corpus and re-runs the ratio inside
the partition the paper is actually about:

  A. the full ISAC + energy-efficiency corpus            (what we published)
  B. the IoT-facing subset                                (what we claim about)
  C. the explicitly battery/energy-constrained subset     (the strongest form)

If the ratio survives in B and C, the claim survives and gets NARROWER and
stronger. If it does not, the paper's lead claim has to be rewritten before
submission, and better now than after review.

Controls are included for the same reason as everywhere else in this project:
a small count in a subset is meaningless unless the subset itself is populated.

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
BASE = 'https://api.openalex.org/works'
FROM_YEAR = 2018

ISAC = '("integrated sensing and communication" OR "ISAC")'
EE = '"energy efficiency"'
IOT = '("IoT" OR "Internet of Things" OR "sensor network" OR "LPWAN" OR "NB-IoT" OR "LoRa")'
BATT = '("battery" OR "battery-powered" OR "energy-constrained" OR "battery life" OR "lifetime")'

TX_TERMS = ['"beamforming"', '"transmit power"', '"power allocation"',
            '"transmit beamforming"', '"power control"', '"waveform design"']
SC_TERMS = ['"circuit power"', '"sleep"', '"idle"', '"quiescent"',
            '"duty cycle"', '"static power"', '"standby"',
            '("battery life" OR "battery lifetime")']

TX_ANY = '(' + ' OR '.join(TX_TERMS) + ')'
SC_ANY = '(' + ' OR '.join(SC_TERMS) + ')'

POPULATIONS = {
    'A_full':  ISAC + ' AND ' + EE,
    'B_iot':   ISAC + ' AND ' + EE + ' AND ' + IOT,
    'C_batt':  ISAC + ' AND ' + EE + ' AND ' + IOT + ' AND ' + BATT,
}


def count(q, tries=6):
    """One count, with backoff. OpenAlex 429s at this call rate without it."""
    flt = 'title_and_abstract.search:%s,from_publication_date:%d-01-01' % (q, FROM_YEAR)
    url = '%s?filter=%s&per-page=1&mailto=%s' % (
        BASE, urllib.parse.quote(flt, safe=':,'), urllib.parse.quote(MAILTO))
    req = urllib.request.Request(url, headers={'User-Agent': 'academic-research (%s)' % MAILTO})
    delay = 2.0
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                time.sleep(1.1)                     # be a good citizen between calls
                return json.load(r)['meta']['count']
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == tries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError('unreachable')


def main():
    out = {'_method': {'source': 'OpenAlex title_and_abstract.search',
                       'from_year': FROM_YEAR,
                       'run_date': time.strftime('%Y-%m-%d'),
                       'purpose': ('partition the corpus and re-run the transmit-side vs '
                                   'standing-charge ratio inside the IoT-facing subset the '
                                   'paper actually makes a claim about')},
           'populations': {}}

    print("=" * 96)
    print("  IS THE 35.7 RATIO AN ARTEFACT OF AN INFRASTRUCTURE-HEAVY CORPUS?")
    print("=" * 96)

    for name, base_q in POPULATIONS.items():
        n = count(base_q)
        tx = count(base_q + ' AND ' + TX_ANY)
        sc = count(base_q + ' AND ' + SC_ANY)
        per_tx, per_sc = {}, {}
        for t in TX_TERMS:
            per_tx[t] = count(base_q + ' AND ' + t)
        for t in SC_TERMS:
            per_sc[t] = count(base_q + ' AND ' + t)

        ratio = (tx / sc) if sc else None
        out['populations'][name] = {
            'query': base_q, 'n': n, 'tx_any': tx, 'sc_any': sc,
            'tx_share': tx / n if n else None,
            'sc_share': sc / n if n else None,
            'ratio': ratio, 'per_tx': per_tx, 'per_sc': per_sc}

        print("\n  %s   n = %d" % (name, n))
        print("    any transmit-side term : %4d  (%.1f%%)" % (tx, 100 * tx / n if n else 0))
        print("    any standing-charge    : %4d  (%.1f%%)" % (sc, 100 * sc / n if n else 0))
        if ratio is not None:
            print("    ratio                  : %.1f to one" % ratio)
        else:
            print("    ratio                  : undefined (zero standing-charge hits)")
        print("    standing-charge breakdown: %s"
              % ", ".join("%s %d" % (k.strip('"')[:14], v) for k, v in per_sc.items() if v))

    # ------------------------------------------------------------- verdict --
    print()
    print("=" * 96)
    print("  VERDICT")
    print("=" * 96)
    A, B, C = (out['populations'][k] for k in ('A_full', 'B_iot', 'C_batt'))
    print("  full corpus      n = %4d  ratio %s"
          % (A['n'], ("%.1f" % A['ratio']) if A['ratio'] else "undefined"))
    print("  IoT-facing       n = %4d  ratio %s   (%.0f%% of the full corpus)"
          % (B['n'], ("%.1f" % B['ratio']) if B['ratio'] else "undefined",
             100 * B['n'] / A['n'] if A['n'] else 0))
    print("  battery-framed   n = %4d  ratio %s   (%.0f%% of the full corpus)"
          % (C['n'], ("%.1f" % C['ratio']) if C['ratio'] else "undefined",
             100 * C['n'] / A['n'] if A['n'] else 0))
    print()
    print("  READ THIS AS:")
    print("  - if the IoT-facing subset is a small minority, the paper MUST report")
    print("    the ratio for that subset, not for the full corpus, or a reviewer will")
    print("    say the transmit-side finding is just infrastructure ISAC behaving")
    print("    correctly for its own platform.")
    print("  - if the ratio HOLDS inside the IoT subset, the claim survives and gets")
    print("    narrower: even the papers that name IoT foreground the transmit side.")
    print("    That is the version worth publishing.")

    dest = os.path.join(RESULTS, 'corpus_iot_subset.json')
    with open(dest, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("\nSaved %s" % dest)


if __name__ == '__main__':
    main()
