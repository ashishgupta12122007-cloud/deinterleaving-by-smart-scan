# SIH 26055 - Smart Scan strategy for Electronic Warfare

A machine-learned dwell scheduler for a receiver that can only listen to a 1 GHz slice of 0-18 GHz at a time,
plus a deinterleaver/identifier that separates overlapping radar pulses. Built and tested on the five files
`config_95, 230, 614, 1283, 2433`.

## Results (leave-one-scenario-out)

Every scenario's ML schedule is learned from the *other four* files only. "twin" = the digital twin described below.
K = number of recorded pulses an emitter needs to count (20 = detected, 100 = characterized).

| scenario | K | real baseline | twin baseline | twin ML | gain | t80 base (s) | t80 ML (s) |
|---|---|---|---|---|---|---|---|
| mean of 5 | 20 | 72.0 | 71.8 | 80.7 | +8.8 | 7.8 | 5.7 |
| mean of 5 | 100 | 59.0 | 58.8 | 72.0 | +13.2 | 11.0 | 9.7 |

Per-scenario numbers: `out/results.md`. Dwell time spent on windows that contain no radar: 25-35% (fixed sweep) vs 0-3% (ML).

Signal side (real pulses, 60 hardest dwells per file): adjusted Rand index 0.96-1.00 for the deinterleaver, 72-88% of
radars recovered cleanly, random-forest radar-type identification 92% top-1 / 97% top-3 on files it never saw
(scored on clusters the deinterleaver got at least 90% pure).

## Run it

    pip install -r requirements.txt
    python run_all.py plan   --data /path/to/h5/files --out out     # twin validation + leave-one-out ML schedules (~20 s)
    python run_all.py signal --data /path/to/h5/files --out out     # deinterleaving, type identification, lab snapshots (~1 min)
    python run_all.py export --data /path/to/h5/files --out out     # prototype_data.json, results.md
    python run_all.py deploy --data /path/to/h5/files --out out     # schedule for a NEW scenario, learned from every file
    python proto/build.py proto/sih26055_smart_scan.html            # rebuild the HTML prototype

`out/smartscan_schedule_all_scenarios.csv` (t_start_s, dwell_s, center_mhz) is the deployable plan; its allocation per centre
is in `smartscan_allocation_all_scenarios.csv`. `out/ml_schedule_scenario_<id>.csv` are the held-out schedules used in the table.
The deployable plan has no held-out score of its own.

## How it works

1. `sim.py` - digital twin. Rebuilt from the metadata and checked against the pulses:
   AoA is the compass bearing receiver->emitter; emitters move as p0 + v*t*(cos a, sin a); a circular beam points at
   az0 + 6*rpm*t degrees (compass, clockwise) and hits the receiver when it equals the bearing emitter->receiver; boresight power
   = Pt - free-space loss + receiver gain; off-axis loss follows an empirical pattern (with a far-lobe floor of gain+12 dB);
   a pulse is recorded with a soft probability of its amplitude (two logistics, fitted by Poisson likelihood). Over the five
   baseline sweeps it predicts per-emitter pulse counts with Spearman 0.86 and median ratio 1.07.
2. `scheduler.py` - for each radar in the training scenarios, Monte-Carlo dwell placement gives P(>= K pulses) against seconds of
   dwell; isotonic regression calibrates twin pulse counts against real counts (removes the twin's optimism). A greedy planner then
   adds 0.1 s chunks where the marginal expected gain is highest, and stride scheduling interleaves visits with jitter.
3. `deinterleave.py` - DBSCAN on bearing (unit-circle) and log pulse width, then merges fragments with the same bearing and
   overlapping carrier support. `identify.py` - random forest on frequency, pulse width, PRI, agility, size.
4. `proto/` - the interactive prototype. The waveform and FFT in the signal lab are synthesized from pulse descriptors
   (arrival time, width, frequency, amplitude); the files contain no raw IQ.

## Things to know before you rely on it

* **Bandwidth.** The files say `bandwith_mhz = 500`, but 99.985% of pulses fall within +-500 MHz of the dwell centre, so the twin uses
  a half-width of 500 MHz (a 1 GHz window). If the official scorer uses +-250 MHz, set `D["rx"]["bandwith_mhz"] = 250` before building
  the twin and re-run `plan`; nothing else needs to change.
* **The ML numbers are twin predictions.** The fixed-sweep numbers are measured; the twin lands within about 4 emitters of them per
  file (worst case 4.1) and matches the mean (71.8 vs 72.0), but nobody has replayed the ML schedule through the organizers' generator.
  The twin is less accurate about *when* things are detected than about the final counts (it detects earlier than the real sweep did).
* **Out of reach.** 3-6 radars per scenario transmit above 18 GHz (35-94 GHz). No schedule can hear them.
* **Not helpful, so left out.** I tried an online layer that re-weights the learned curves with live detections and re-plans every 3 s.
  In the twin it added +0.2 emitters at K=20 (+0.9 at K=100) and did not help when the environment disagreed with the prior, so it is not included.
* `mini_h5.py` is a small pure-python HDF5 reader written for these files (superblock v0, symbol-table groups, chunked/deflate).
  `data_io.load_config_h5py` is an h5py version that has not been run.

## Files

    mini_h5.py  data_io.py     reading the .h5 files
    sim.py                      digital twin
    scheduler.py evaluate.py    learned prior, planner, scoring
    deinterleave.py identify.py pulse separation and radar-type classifier
    run_all.py                  pipeline (plan | signal | export | deploy)
    proto/template.html build.py  the prototype (data is inlined at build time)
    out/                        schedules (CSV), results.md, prototype_data.json
