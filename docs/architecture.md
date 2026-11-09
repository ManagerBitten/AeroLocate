# Architecture

AeroLocate turns a recording into either a supported set of candidate locations or an explicit statement that no location is supported. Every stage keeps enough information to trace a result back to its measurements.

## Pipeline

```
upload -> probe + extract audio -> waveform + spectrogram
       -> aircraft sound detection
       -> Doppler curve extraction
       -> flight retrieval -> fusion -> trajectory reconstruction
       -> posterior search -> regions -> refinement + sensitivity -> status decision
```

| Stage | Module | Output |
|---|---|---|
| Ingestion | `backend/audio/ingest.py` | Validated media, original-rate WAV, unverified timestamp and location hints |
| Spectral views | `backend/audio/analysis.py` | Waveform, spectrogram, optional noise reduction on a copy |
| Detection | `backend/detection/` | Events with category and a score labelled as heuristic or model output |
| Doppler | `backend/doppler/` | Curves of time, frequency and uncertainty, each with a class and quality |
| Flight data | `backend/flight_data/` | Fused observations with provenance, reconstructed trajectories |
| Propagation | `backend/propagation/simulator.py` | Predicted Doppler factor using retarded time |
| Localization | `backend/geolocation/`, `backend/optimization/` | Posterior, credible regions, status, explanation |
| Jobs | `backend/workers/` | Celery or thread execution, progress, cancellation, retention |
| API | `backend/api/` | REST, WebSocket progress, reports |

## Physics

For a stationary microphone, sound heard at time `t` left the aircraft at an earlier emission time `τ`:

```
t = τ + R(τ) / c_eff
```

This is solved per sample by safeguarded Newton iteration. Convergence requires subsonic radial motion, so supersonic geometry is flagged invalid rather than used. The observed frequency is:

```
f_observed = f_source × c_eff / (c_eff − v_radial)
c_eff = c + w · û
```

`c` follows the ISA lapse rate. Propagation delay uses the harmonic mean sound speed along the path, and the Doppler factor uses the sound speed at the source. Rays are straight, so refraction is not modelled.

## What one curve can tell you

A pass-by constrains three things: aircraft speed, the time of closest approach and the slant range at closest approach. That has direct consequences:

- With a known altitude, slant range gives cross-track distance but not its sign. One aircraft on a straight track leaves two mirror-image solutions.
- With an unknown recording clock, the closest-approach time cannot be placed on the track, so the microphone can slide along it.
- Any flight of a similar speed can fit the same curve at some location. One curve cannot show which flight was heard.
- Two aircraft on different headings share one microphone and one clock, which removes all three problems.

For this reason `localized` requires at least two events matched to different flights, whatever the likelihood says. Each pass gives two position constraints and the unknowns are two coordinates plus the clock if it is free, so an unknown clock needs three.

## Doppler extraction

1. STFT at three window lengths, 0.35 s, 0.7 s and 1.4 s, trading frequency against time resolution.
2. Local spectral whitening, then Viterbi ridge tracking on a log-frequency axis, where a physical slope limit is the same at every frequency.
3. Rejection of noise chains by path jitter, and of sidelobes next to much stronger ridges.
4. Star grouping by common fate. A ridge joins a group only by matching its strongest member, so a hybrid ridge cannot chain two aircraft together.
5. Per-frame median merging, so a ridge contributes only where it agrees with its group.
6. Hampel filtering and an RTS Kalman smoother, then decimation to about one window per sample.
7. A robust closed-form pass-by fit, with trimming limited so it cannot remove most of a curve.
8. Classification by BIC against constant and linear models. Rising curves, flat tones, mains hum and ridge crossing artifacts are not treated as Doppler.

Crossing artifacts are single ridges that spend at least half their samples on harmonics of a confirmed aircraft. Without this check, trimming can produce a plausible curve for an aircraft that does not exist.

## Trajectory reconstruction

Provider observations are kept unchanged. For reconstruction, implausible jumps, stale points and provider-estimated points are removed, then each node gets a local cubic fit over ±20 s to positions and reported velocities together. Passing a spline exactly through raw reports turns 15 m of position noise into about 1.5 m/s of false velocity, which is many times the measurement error of a Doppler curve. A wider window or a lower-order fit would bias standard-rate turns.

## Likelihood

For event `e`, flight `a`, microphone `m` and recording start `T`, the residual is `log f_observed − log D(m, T)`. Three nuisance terms are removed in closed form:

- **Source frequency**, a free offset.
- **Slow throttle drift**, with a prior of 0.06 % per second.
- **Amplitude scale**, with a prior from temperature, wind and track speed uncertainty. These errors scale the whole curve together, so they are one parameter, not independent noise per sample.

The speed measured from the curve must agree with the flight's speed, within its uncertainty plus 3 % for temperature and wind. This enters as a likelihood factor weighed against the chance of a coincidental match, not as a hard cut.

Track velocity error is correlated over the smoothing window, so its variance is scaled by the number of samples in that window.

Each event is a mixture over candidate flights and an "untracked aircraft" hypothesis:

```
log P(m, T) = Σ_e log[ w0 L0(e) + Σ_a w_a L(e | a, m, T) ]
```

`L0` comes from the free pass-by fit with an Occam factor. The untracked hypothesis is given the same expected path error as a tracked one, so the comparison is fair. An event that fits no flight falls to `L0`, which rejects inconsistent combinations without forcing a match. Harmonics are merged and repeated geometry is removed first, so correlated evidence is not counted twice.

## Search

Tiles over position and recording start are refined from 1600 m and 8 s down to 50 m and 0.25 s. At coarse levels the noise is widened by how much the curve can change inside a tile, so the true peak cannot be pruned early. Flights are rejected early by speed, altitude and time coverage, and a gate limits each pair to tiles near its closest approach. If the posterior stays too broad for the tile budget, refinement stops and the result reports the coarser resolution.

## Decisions

| Status | Requirement |
|---|---|
| `localized` | P(tracked) ≥ 0.9, one region with ≥ 80 % of the posterior, radius ≤ 2.5 km, and at least two constraints to spare: two matched flights with a known clock, three with an unknown one, each within 6 km slant range |
| `region_only` | As above, but the region is broad or resolution was limited |
| `ambiguous` | P(tracked) ≥ 0.5 with several regions, moderate evidence, or only one matched flight |
| `not_localizable` | No measurable curve, no flights, or the untracked hypothesis is preferred |

Coordinates are shown only for `localized`, rounded to the uncertainty and never beyond three decimals (about 110 m) until real recordings justify more. Probabilities are model posteriors, not calibrated frequencies.

## Data fusion

Records are joined by ICAO hex, then registration, then callsign with trajectory agreement. Near-duplicates keep the better observation, and altitude disagreements are recorded. Agreement between providers relaying the same ADS-B transmissions is not treated as independent. Independence is counted by signal family, such as ADS-B, MLAT or radar.

## Storage

PostgreSQL with PostGIS holds users, analyses, Doppler observations, flight candidates, raw provider observations, candidate regions and the audit log. Geometry columns are GiST indexed. SQLite is used for tests and single-user development, with geometry stored as text.

The schema is created at startup and there is no migration tool yet.

## Security

- JWT bearer tokens, Argon2 password hashes, per-user ownership on every analysis route
- A required authorization basis on every upload, recorded in the audit log
- Extension, signature and FFmpeg container validation, with size, duration, time and memory limits
- FFmpeg restricted to the file protocol, with no shell invocation
- Rate limiting, retention with automatic purge, and full deletion of files and rows
- Secrets from the environment only, and only nginx is published, on localhost
