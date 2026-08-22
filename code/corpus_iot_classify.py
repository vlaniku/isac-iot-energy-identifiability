"""
Hand-adjudication support for the full-text screen, and extraction of the
controllable share f where the paper states enough to compute it.

WHY THIS EXISTS. `corpus_iot_screen.py` found a standing-charge TERM in 7 of 14
screened papers. Taken at face value that refutes the paper's lead claim. Taken
carefully it does not, because the regex is deliberately generous and several
hits are obviously not what we are looking for: a reference titled "Advanced
sleep modes..." in a bibliography is not a system model, and "tracking vital
signs during sleep" is not a power state.

So the screen's output is evidence, not a verdict, and this script produces what
is needed to reach a verdict: a wide context window around every hit, grouped by
paper, plus any nearby numeric power values, so that each paper can be placed in
one of three classes BY A PERSON:

  MODELS   a standing charge appears in the system model or parameter table
  MENTIONS a standing-charge word appears only in prose, related work or a
           reference title, with no corresponding model term
  ABSENT   no occurrence at all

AND THE POINT THAT MATTERS MORE THAN THE CLASS. Having a circuit-power term is
not the same as representing a device whose budget is dominated by one. Ye et
al. carries P_ST and still sits at f = 0.984; Bai carries 5 mW of circuit power
against a 200 mW budget and sits at f = 0.97. So the interesting quantity is not
presence but MAGNITUDE: what value of f does each paper's own parameter table
imply? This script surfaces the numbers needed to answer that, which is a
stronger claim than presence/absence and is the one the evidence actually
supports.

Author: Vullnet Laniku
"""

import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results')
LITDIR = os.path.join(HERE, '..', 'docs', 'lit_corpus')

# Terms, split by what they would mean if found in a system model.
STANDING = [r'circuit\s+power', r'circuitry\s+power', r'static\s+power',
            r'hardware\s+power', r'constant\s+power', r'\bquiescent\b',
            r'\bstandby\b', r'sleep\s+(?:power|mode|current|state)',
            r'idle\s+(?:power|mode|current|state)', r'duty[- ]cycle',
            r'\bp_?\{?c\}?\b', r'\bp_?\{?cir', r'\bp_?\{?st']
# Contexts that mean the hit is NOT a system-model term.
BIBLIO = [r'ieee\s+j', r'ieee\s+trans', r'ieee\s+commun', r'proc\.', r'vol\.',
          r'no\.\s*\d', r'pp\.\s*\d', r'\bet\s+al\.', r'\[\d+\]']

POWER_VAL = re.compile(
    r'(\d+(?:\.\d+)?)\s*(dbm|mw|w|uw|µw|nw)\b', re.I)


def read_pdf(path):
    from pypdf import PdfReader
    try:
        r = PdfReader(path)
        return "\n".join((p.extract_text() or '') for p in r.pages)
    except Exception as e:
        print("   ! %s" % e)
        return ''


def looks_bibliographic(ctx):
    return sum(1 for p in BIBLIO if re.search(p, ctx)) >= 2


def main():
    scr = json.load(open(os.path.join(RESULTS, 'corpus_iot_screen.json'), encoding='utf-8'))
    man = {w['slug']: w for w in
           json.load(open(os.path.join(RESULTS, 'corpus_iot_manifest.json'),
                          encoding='utf-8'))['works']}
    slugs = scr['fulltext_screened']

    out = {}
    for slug in slugs:
        path = os.path.join(LITDIR, slug + '.pdf')
        if not os.path.exists(path):
            continue
        txt = read_pdf(path)
        low = re.sub(r'\s+', ' ', txt.lower())

        found = []
        for pat in STANDING:
            for m in re.finditer(pat, low):
                ctx = low[max(0, m.start() - 200): m.end() + 200]
                found.append({'term': pat, 'context': ctx,
                              'bibliographic': looks_bibliographic(ctx)})

        real = [f for f in found if not f['bibliographic']]
        out[slug] = {'title': man[slug]['title'], 'venue': man[slug]['venue'],
                     'n_hits': len(found), 'n_nonbiblio': len(real),
                     'hits': real[:12]}

        print("=" * 104)
        print("  %s  %s" % (slug, man[slug]['title'][:80]))
        print("  %s" % (man[slug]['venue'] or '')[:80])
        print("  %d hits, %d outside a bibliography" % (len(found), len(real)))
        print("=" * 104)
        seen = set()
        for f in real[:8]:
            key = f['context'][80:140]
            if key in seen:
                continue
            seen.add(key)
            print("\n  [%s]" % f['term'])
            print("   ...%s..." % f['context'][:360])
        # any power magnitudes anywhere near a standing-charge word
        vals = []
        for f in real:
            for m in POWER_VAL.finditer(f['context']):
                vals.append(m.group(0))
        if vals:
            print("\n  power magnitudes near standing-charge terms: %s"
                  % ", ".join(sorted(set(vals))[:14]))
        print()

    dest = os.path.join(RESULTS, 'corpus_iot_classify.json')
    with open(dest, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("Saved %s" % dest)
    print()
    print("NOW CLASSIFY BY HAND. The counts above are inputs to a judgement,")
    print("not the judgement. Record the verdict per paper in the paper's table.")


if __name__ == '__main__':
    main()
