# Bench protocol — the two measurements that convert this paper's conditionals

**Time: one afternoon for Session A, then Session B runs unattended for days.**

Everything downstream is already written. `code/bench_analysis.py` consumes the two
capture files defined below and writes `results/bench_uplink_energy.json` and
`results/bench_response.json`. Its self-test passes, so the analysis is validated
before the data exists — run `python bench_analysis.py --selftest` to see it.

---

## Why these two

**Session A settles `k`.** The amplification factor is currently derived from the LoRa
airtime formula plus an assumed SX1276 radio model (3.3 V, 43.9 mA transmit, 11.0 mA
receive, 3.66 mJ fixed overhead). Transmit-only it is **17.4×**; including the class-A
receive windows it is **11.6×**. Because the detectable share goes as `s = ε/(k−1)`,
that is a **1.58× swing in the headline number**, and no further modelling closes it.
The paper already calls this a precondition of the campaign rather than a refinement.

**Session B settles the response.** The identifiability criterion assumes the reported
battery byte is proportional to energy drawn. Sec. VII-G says this cannot presently be
verified. This measures it directly, as byte against coulombs.

---

## Session A — energy of one uplink cycle

### Equipment

- One spare parking sensor, **in the deployed firmware configuration**. If the firmware
  differs from the fleet's, the measurement does not transfer and the session is wasted.
- A current-measuring instrument spanning **microamps to ~50 mA**. This dynamic range is
  the hard part: a single fixed shunt cannot resolve both sleep (µA) and transmit
  (~44 mA). An auto-ranging tool (Nordic PPK2, Otii Arc, or equivalent) handles it
  directly. With a bench setup instead, use a low-side shunt plus a differential amp and
  take **two captures per uplink at different gains**, or accept a poor sleep estimate
  and record that you did.
- A stable supply at the deployed rail voltage, or the cell itself with voltage logged.

### Procedure

For each spreading factor in **7, 9, 10, 12** — 7 and 12 are the minimum, 9 and 10 check
the model's shape between them:

1. Pin the device to that SF (disable ADR for the session; record how).
2. Trigger one uplink with the **deployed 12-byte payload**.
3. Capture the whole cycle at **≥ 10 kHz** (20 kHz used in the self-test). SF12 airtime
   is 1.71 s, so a capture of about 2.5 s covers wake → TX → RX1 → RX2 → sleep.
4. **Begin the capture at least 10 ms before the radio wakes.** This is not optional.
   The analysis estimates the sleep baseline from the pre-activity window, and it does
   so *because* the obvious alternative is wrong: sleep is ~20% of an SF7 capture but
   only ~3% of an SF12 one, so a percentile-based baseline picks up receive current at
   SF12, subtracts it from every phase, and understates `k`. The self-test reproduced
   this exactly — 12.0 against a true 16.5, a 27% error, in the one number the session
   exists to measure.
5. **Five repetitions per SF.** The spread across repetitions is what tells you whether
   the number is worth quoting.

### File: `data/bench_uplink_traces.csv`

One row per sample, all SFs and repetitions concatenated:

```
sf,rep,t_s,i_a
7,0,0.00000,0.0000021
7,0,0.00005,0.0000020
...
12,4,2.49995,0.0000019
```

- `sf` — 7, 9, 10 or 12
- `rep` — 0-based repetition index
- `t_s` — seconds from the start of *that* capture (each rep restarts at ~0)
- `i_a` — supply current in **amps** (not mA)

If the rail is not 3.3 V, say so and I will parameterise it; it is currently a constant
in the script.

### What to check before you pack up

Run `python bench_analysis.py`. At the bench, confirm:

- The per-SF **transmit duration** matches the LoRa airtime for that SF (SF7 ≈ 0.0617 s,
  SF12 ≈ 1.7125 s). If it does not, the SF did not take effect.
- Both receive windows appear — RX1 shortly after TX, then **RX2 ≈ 164 ms**. If RX2 is
  missing the device is not behaving as class-A and `k` will come out near the
  transmit-only value for the wrong reason.
- The sleep baseline is microamps, not milliamps. Milliamps means the pre-trigger window
  was too short or the instrument floor is too high.

---

## Session B — battery response

### Procedure

1. Take one cell of the deployed type. Record chemistry and nominal capacity.
2. Discharge it through a **known resistor**, logging the terminal voltage.
3. In parallel, log the device's **reported battery byte** at every uplink, from the
   network server or the serial console.
4. Run until the byte has moved across as much of its range as patience allows. Span is
   what buys resolution: the analysis fits byte against cumulative charge, so a byte that
   moves 5 counts tells you far less than one that moves 50.

### File: `data/bench_discharge.csv`

```
t_s,i_a,byte
0,0.0201,255
1800,0.0201,255
3600,0.0200,254
...
```

- `t_s` — seconds from start
- `i_a` — load current in amps. Either measured, or computed as V/R from the logged
  terminal voltage and the known resistor. **Use the logged voltage, not the nominal
  cell voltage** — the whole point is that the voltage sags.
- `byte` — the reported battery byte at rows where an uplink occurred; blank or `nan`
  everywhere else.

### The caveat that must go in the paper

An accelerated discharge is **not** the deployed discharge. These cells are typically
Li-SOCl₂, which has strong rate dependence and passivation behaviour; a cell pulled at
20 mA does not necessarily report the way the same cell does at the fleet's microamp
average. Two consequences, both honest to state rather than hide:

- Keep the acceleration as **modest as the schedule allows**. Slower is better evidence.
- The result licenses the *shape* of the response — linear or not — more strongly than
  its slope. The criterion needs proportionality, which is the shape.

---

## What the analysis reports

**Session A:** per-SF energy with its spread, split into transmit and receive; `k` as
measured; and which of the two modelled values (17.4× or 11.6×) the measurement lands
nearer. It also reports what the measured `k` does to the detectable share relative to
the transmit-only figure the paper currently carries.

**Session B:** slope in byte per coulomb, R² of the linear fit, the largest residual as a
percentage of the byte span, and a quadratic-term *t*-test flagging curvature.

The curvature test states its own resolution, because a test that only catches a gross
departure is not evidence of linearity. On the self-test's synthetic capture — 400
readings, 0.3 byte of noise, integer quantisation — it detects a departure of about
**0.12 byte off the straight line at mid-discharge**. Your real resolution will differ
with your reading count and noise; the script recomputes it.

---

## After the bench

1. Drop both CSVs into `data/`.
2. `python code/bench_analysis.py`
3. The two JSONs land in `results/`.
4. Sec. IX and Sec. VII-G stop saying "modelled" and start saying "measured" — which is
   the whole point, and the paper's two largest conditionals close.
5. Add the new claims to `code/verify_manuscript.py` so they are checked on every build,
   and to `code/build_release.py`'s data allow-list so the captures ship with the paper.
