"""
Pull the Newcastle Urban Observatory archive.

The archive at https://archive.newcastle.urbanobservatory.ac.uk/ publishes raw
per-variable CSVs, monthly and yearly, from 2013 to the present. `Battery` sits
under "Sensor Metrics". The covariates that FIEK cannot supply -- Solar Radiation
and Temperature -- are published alongside it.

URL patterns, confirmed from the archive's own links on 2026-08-19:

    year  : /file/year_file/{YYYY}-{Variable}.csv.zip
    month : /file/month_file/{YYYY}-{M}-{Variable}.csv.zip

`data/2025-9-Battery.csv` already in this repo is one month_file, unzipped.

WHY THIS MATTERS. The FIEK deployment gives five devices and two end-of-life
events, which is why every field claim in PAPER_v3 is a bound rather than a
measurement. Thirteen years of a reference urban observatory gives device
turnover -- installs and cessations -- across multiple power architectures, with
each device's own reporting workload computable from its payload streams, and
with temperature as a covariate. That is the population the n=5 design cannot be.

USAGE

    python fetch_uo_archive.py --list                 # sizes only, downloads nothing
    python fetch_uo_archive.py --years 2019-2026      # Battery only
    python fetch_uo_archive.py --years 2019-2026 --vars Battery "Solar Radiation" Temperature

Files land in data/uo_archive/ and are left zipped; loaders read them in place.

Author: Vullnet Laniku
"""

import argparse
import os
import sys
import urllib.parse
import urllib.request

BASE = "https://archive.newcastle.urbanobservatory.ac.uk/file"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'data', 'uo_archive')

DEFAULT_VARS = ["Battery"]
UA = {"User-Agent": "academic-research-fetch/1.0"}


def year_url(year, var):
    return "%s/year_file/%s-%s.csv.zip" % (BASE, year, urllib.parse.quote(var))


def month_url(year, month, var):
    return "%s/month_file/%s-%s-%s.csv.zip" % (BASE, year, month,
                                               urllib.parse.quote(var))


def head(url):
    """Content-Length without downloading the body. Returns bytes or None."""
    req = urllib.request.Request(url, headers=UA, method='HEAD')
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            n = r.headers.get('Content-Length')
            return int(n) if n else -1
    except Exception as e:
        return None


def fetch(url, dest):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, 'wb') as f:
        total = 0
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
    return total


def parse_years(spec):
    if '-' in spec:
        a, b = spec.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(',')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2019-2026')
    ap.add_argument('--vars', nargs='+', default=DEFAULT_VARS)
    ap.add_argument('--list', action='store_true',
                    help='report availability and size, download nothing')
    ap.add_argument('--monthly', action='store_true',
                    help='fetch month files instead of year files')
    a = ap.parse_args()

    years = parse_years(a.years)
    os.makedirs(OUT, exist_ok=True)

    jobs = []
    for var in a.vars:
        for y in years:
            if a.monthly:
                for m in range(1, 13):
                    jobs.append((year_url(y, var) if False else month_url(y, m, var),
                                 "%s-%s-%s.csv.zip" % (y, m, var)))
            else:
                jobs.append((year_url(y, var), "%s-%s.csv.zip" % (y, var)))

    print("%-46s %14s %s" % ("file", "size", "status"))
    total = 0
    todo = []
    for url, name in jobs:
        dest = os.path.join(OUT, name)
        if os.path.exists(dest):
            print("%-46s %14s already present" % (name, "%.1f MB" % (os.path.getsize(dest) / 1e6)))
            continue
        n = head(url)
        if n is None:
            print("%-46s %14s not available" % (name, "-"))
            continue
        total += max(n, 0)
        todo.append((url, dest, name, n))
        print("%-46s %14s available" % (name, "%.1f MB" % (n / 1e6) if n > 0 else "unknown"))

    print()
    print("%d files to fetch, %.1f MB total" % (len(todo), total / 1e6))
    if a.list:
        print("--list given: nothing downloaded.")
        return
    if not todo:
        return

    for url, dest, name, n in todo:
        sys.stdout.write("  fetching %-40s " % name)
        sys.stdout.flush()
        try:
            got = fetch(url, dest)
            print("%.1f MB" % (got / 1e6))
        except Exception as e:
            print("FAILED: %s" % e)

    print("\nSaved to %s" % os.path.abspath(OUT))


if __name__ == '__main__':
    main()
