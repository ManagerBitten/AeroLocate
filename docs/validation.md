# Validation

AeroLocate has been tested on synthetic recordings and on short real clips for sound classification only. It has **not** been tested on real recordings with known locations, so nothing here shows how accurate it is in the field. The figures below show what the method can do when its assumptions hold, and how it fails when they do not.

## Synthetic benchmark

`backend/validation/benchmark.py` builds each recording from first principles:

- **Phase:** taken from the retarded emission time, so propagation delay is exact
- **Distortion:** engine speed wander, turns, climbs and pink noise
- **Wrong assumptions:** a temperature and wind that differ from what the analysis assumes
- **Impaired flight data:** reports every 5–15 s, with 15 m position noise, 1 m/s velocity noise, 10 m altitude noise and a constant barometric altitude bias of up to about 60 m
- **Decoys:** flights in the same speed range, so a matching speed alone proves nothing

Each scenario has 20 recordings. All 200 completed without errors. The raw results are in `validation_results.json`.

| Scenario | Localized | Region only | Ambiguous | Not localizable | Median error (m) | Worst error (m) | Within stated radius | Truth in a candidate region |
|---|---|---|---|---|---|---|---|---|
| Two aircraft, ideal data | 13 | 0 | 6 | 1 | 6 | 52 | 13/13 | 95% |
| One aircraft | 0 | 0 | 20 | 0 | – | – | – | 100% |
| Two aircraft | 14 | 0 | 4 | 2 | 56 | 257 | 14/14 | 94% |
| Three aircraft | 9 | 0 | 10 | 1 | 66 | 91 | 9/9 | 95% |
| Two aircraft, clock unknown ±60 s | 0 | 5 | 12 | 3 | – | – | – | 94% |
| Two aircraft, heavy noise | 13 | 0 | 6 | 1 | 37 | 1915 | 12/13 | 100% |
| Cruise altitude, 9–11.5 km | 0 | 1 | 13 | 6 | – | – | – | 43% |
| Control: right flight missing | 0 | 0 | 18 | 2 | – | – | – | 0% |
| Control: both right flights missing | 1 | 0 | 17 | 2 | 6244 | 6244 | 0/1 | 6% |
| Control: no aircraft | 0 | 0 | 0 | 20 | – | – | – | – |

Error figures cover `localized` results only. "Truth in a candidate region" covers every result that returned regions.

### What this shows

- **When a location is given, it is usually right.** Across all aircraft scenarios, 48 of 49 localized results fell within their stated uncertainty radius. The exception was a heavy-noise recording 1.9 km out.
- **One aircraft never produces a location,** as the physics requires. The true position was inside a candidate region in all 20 cases.
- **An unknown clock needs three aircraft,** so the two-aircraft scenario gives regions only.
- **Cruise altitude is not supported.** No location is ever claimed, but the candidate regions often miss, because the curves are weak and long-range sound bends in ways the model ignores.
- **There is one known way to be confidently wrong.** If neither aircraft heard is in the flight data and other flights of similar speed are, two curves can line up with the wrong pair by chance. That happened in 1 of 20 trials of this deliberately adversarial control, 6.2 km from the truth. Use a tight search region and time window, because both reduce how many wrong combinations are possible.

### Rules adopted because of this benchmark

These were changed after earlier runs exposed failures:

| Problem found | Change |
|---|---|
| 3 of 7 single-aircraft results claimed a location 7 km off | A location needs at least two matched flights |
| Credible regions held the truth only 59–77% of the time | Track errors are treated as correlated over 20 s |
| A 10% speed mismatch could still match | The measured speed must agree with the flight's speed |
| Unknown-clock and cruise cases gave confident wrong answers | The clock-free case needs three aircraft, and curves beyond 6 km cannot confirm a location |
| Crossing harmonics produced aircraft that do not exist | Ridges that follow another aircraft's harmonics are rejected |

A further rule was tested and **rejected**: excluding curves that fit a straight pass-by poorly. Correct matches turned out to be just as irregular as wrong ones, because real turns and engine speed changes cause the same misfit. The rule removed 42 of 50 correct locations to prevent one wrong one.

## Sound classification on real audio

The AudioSet model (`MIT/ast-finetuned-audioset-10-10-0.4593`) and the fallback spectral heuristic were run on 96 real five-second clips from the [ESC-50](https://github.com/karolpiczak/ESC-50) research dataset (K. J. Piczak, "ESC: Dataset for Environmental Sound Classification", ACM Multimedia 2015), 12 per class. The clips are not redistributed here. The model flags a clip as aircraft at a score of 0.05 or more.

| Class | Model flags as aircraft | Heuristic flags as aircraft |
|---|---|---|
| Helicopter | 100% | 100% |
| Airplane | 83% | 92% |
| Vacuum cleaner | 25% | 100% |
| Wind | 8% | 58% |
| Engine | 8% | 100% |
| Train | 0% | 83% |
| Car horn | 0% | 100% |
| Chirping birds | 0% | 100% |

The heuristic cannot tell aircraft from other sounds, so it no longer names aircraft types and is used only when the model cannot be loaded. The model is the default. Twelve clips per class is a small sample.

Detection does not gate localization: a Doppler curve is used whether or not the model recognized an aircraft. When it did not, the result says so and asks you to check the sound by listening.

## Real stack check

A full upload ran through the Docker stack: nginx, FastAPI, Redis, Celery with the AudioSet model on CPU, and PostgreSQL 16 with PostGIS 3.4. The synthetic demo recording was localized to within 113 m of its known position. PostGIS stored a valid credible region, both flight tracks as 3D lines, 240 raw provider observations and 293 Doppler observations. The same flow ran through the web interface in Microsoft Edge with no console errors and no horizontal scrolling at phone width.

These checks found four faults that tests on SQLite could not:

- A spatial library hook stopped the API starting on PostgreSQL
- Responses were sent before the database commit, so an immediate login failed
- nginx held a stale address after the API container was recreated
- The flight dataset folder was missing on a fresh volume. Local datasets are now read from `flight_datasets/` in the project folder.

## What has not been tested

- Real recordings at known locations. This is the most important gap. Record your own aircraft passes with a GPS-logged position and compare.
- Any flight data provider with a live key
- Real flight data. The ADS-B Exchange free first-of-month samples, loaded through the local dataset provider, are the cheapest way to start.
- Jets, recordings longer than a few minutes, and multi-channel or heavily compressed audio

## Reproducing

```
PYTHONPATH=. .venv/Scripts/python -m backend.validation.benchmark --trials 20 --workers 8
```

Results go to `docs/validation_results.json`. Progress is saved per trial to a `.jsonl` journal, so an interrupted run continues where it stopped. A full run takes about 1.5 hours on 8 cores.
