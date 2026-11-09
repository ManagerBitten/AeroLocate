# AeroLocate

AeroLocate estimates where a recording was made by measuring the Doppler shift of aircraft heard in it and comparing that with real flight tracks. It is built to say when a recording cannot be located, and it only gives coordinates when at least two independent aircraft support them.

> **Status:** experimental. It has been validated on synthetic recordings only, not on real recordings at known locations. Treat every result as unconfirmed until you have tested it on your own data. See [docs/validation.md](docs/validation.md).

Use it only on recordings you made or are authorized to analyze. Every upload records an authorization basis in an audit log, and the tool has no monitoring or person-tracking features.

## What to expect

- **One aircraft gives candidate areas, not a location.** A single pass-by measures speed, closest approach and range. That leaves a mirror-image ambiguity and cannot show which of several similar flights was heard.
- **Two or more aircraft on different headings can give a location.** With an unknown recording time, three are needed.
- **Clear engine tones matter.** Propeller aircraft and low flights work best. Jets at cruise altitude rarely produce a usable curve.
- **You need flight data covering the place and time.** Without it you still get the acoustic analysis, but no location. See [Flight data](#flight-data).

In synthetic tests, when the system gave a location it was within its stated radius 48 times out of 49, with median errors of 6–66 m. These figures describe the method under controlled conditions, not real-world accuracy.

## Quick start with Docker

You need [Docker](https://docs.docker.com/get-docker/) with Compose v2, Python 3 (only to generate secrets), about 6 GB of disk space and 4 GB of RAM.

```
git clone <your-fork-url> aerolocate
cd aerolocate
python3 scripts/init_env.py
docker compose up -d --build
```

On Windows, use `python` instead of `python3`.

Open http://localhost:8080 and create an account. The first build takes several minutes. The first analysis also downloads the aircraft sound model, about 330 MB.

`init_env.py` creates `.env` with a random database password and token secret. Add any flight data API keys to `.env`, then run `docker compose up -d` again.

### Lighter install

The sound model needs PyTorch, which adds about 1 GB to the image. To build without it, set `AEROLOCATE_INSTALL_ML=false` and `AEROLOCATE_CLASSIFIER=spectral` in `.env`. Localization still works, but sounds are no longer classified as aircraft or not.

### Accounts

Anyone who can reach the site can sign up while registration is open. Once your accounts exist, set `AEROLOCATE_ALLOW_REGISTRATION=false` in `.env` and run `docker compose up -d`.

To create an administrator, who can also read the audit log:

```
docker compose exec api python -m backend.cli create-user you@example.com --admin
```

To promote an existing account:

```
docker compose exec api python -m backend.cli promote you@example.com
```

### Stopping and removing

```
docker compose stop
```

Your data is kept in Docker volumes. To delete it as well, run `docker compose down -v`.

## Try it with synthetic data

This creates a recording and matching flight tracks with a known location of 51.47, -0.45. Both are marked `SYNTHETIC`.

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
PYTHONPATH=. .venv/bin/python scripts/make_demo.py
```

On Windows, use `.venv\Scripts\` instead of `.venv/bin/`, and run `set PYTHONPATH=.` first.

The flight file goes into `flight_datasets/`, which the containers read directly. In the app, upload `data/demo/SYNTHETIC_recording.wav` with these settings:

| Setting | Value |
|---|---|
| Date | 2025-06-15 |
| Time | 15:06:40 |
| Timezone | UTC |
| South, West, North, East | 51.38, -0.59, 51.52, -0.34 |
| Flight data | `local_dataset` only |

The result should be **Localized** at about 51.470, -0.450, from two matched flights.

## Flight data

| Provider | Finds aircraft in an area and time | Gets a known aircraft's track | Setting |
|---|---|---|---|
| [Flightradar24](https://fr24api.flightradar24.com/) | Yes, back to 2016 depending on plan | Yes | `AEROLOCATE_FR24_API_KEY` |
| [ADS-B Exchange](https://www.adsbexchange.com/data-products/) | No | Yes, on plans with traces | `AEROLOCATE_ADSBX_API_KEY` |
| [OpenSky Network](https://opensky-network.org/) | Last hour only | Last 30 days | `AEROLOCATE_OPENSKY_CLIENT_ID`, `AEROLOCATE_OPENSKY_CLIENT_SECRET` |
| Local files | Yes | Yes | Put files in `flight_datasets/` |

For historical recordings you need Flightradar24 or your own data files. ADS-B Exchange cannot find aircraft by area in the past, so it only adds detail to aircraft found another way.

The cheapest way to try real data is ADS-B Exchange's free sample of the first day of each month. Download `trace_full_*.json` files and put them in `flight_datasets/`. Supported formats are in [docs/flight-data-providers.md](docs/flight-data-providers.md). Follow each provider's licensing and retention terms.

## Reading a result

| Status | Meaning |
|---|---|
| Localized | One small region, supported by enough independent flights. Coordinates are shown, rounded to about 110 m at most. |
| Broad region only | One region, too large for a point estimate. |
| Ambiguous | Several places remain plausible, or the evidence is too thin to confirm one. No coordinates are given. |
| Not localizable | No usable Doppler curve, no flight data, or an aircraft that is not in the flight data explains the sound better. |

Percentages are the model's probabilities under its stated assumptions, not calibrated real-world odds. The Results tab lists those assumptions, what was rejected and why.

For better results, give the temperature, wind and ground elevation at the time, the narrowest honest time window and the smallest search region that surely contains the recording. Lossless audio that covers the whole pass-by is best.

## Configuration

Settings are environment variables in `.env`. All of them are listed in [backend/core/config.py](backend/core/config.py), with the `AEROLOCATE_` prefix.

| Variable | Default | Purpose |
|---|---|---|
| `AEROLOCATE_ALLOW_REGISTRATION` | `true` | Whether new accounts can sign up |
| `AEROLOCATE_RETENTION_DAYS` | `30` | When analyses and uploaded files are deleted |
| `AEROLOCATE_FLIGHT_CACHE_TTL_HOURS` | `24` | How long provider responses are cached |
| `AEROLOCATE_MAX_UPLOAD_BYTES` | 2 GiB | Upload size limit |
| `AEROLOCATE_INSTALL_ML` | `true` | Build the image with the sound model |
| `AEROLOCATE_CLASSIFIER` | `ast` | `ast` for the sound model, `spectral` if it is not installed |
| `WEB_PORT` | `8080` | Local port for the web app |
| `VITE_MAP_STYLE_URL` | OpenFreeMap dark | MapLibre style for the maps |

The site is published on `127.0.0.1` only. To expose it on a network, put a TLS reverse proxy in front of it.

## Development

The backend needs Python 3.12 or later and FFmpeg on `PATH`. Without Docker it uses SQLite and runs jobs in a thread pool.

```
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m uvicorn backend.api.main:app --reload
```

Add `-r requirements-ml.txt` and PyTorch if you want the sound model locally.

```
cd frontend
npm install
npm run dev
```

The dev frontend runs on http://localhost:5173 and forwards `/api` to port 8000.

To run the tests:

```
.venv/bin/python -m pytest
cd frontend && npm test
```

To run the synthetic benchmark (about 1.5 hours on 8 cores, and it resumes if stopped):

```
PYTHONPATH=. .venv/bin/python -m backend.validation.benchmark --trials 20
```

See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## Project layout

```
backend/          FastAPI app, signal processing, physics, flight data, inference, jobs
  api/            HTTP and WebSocket endpoints, auth, reports
  audio/          media validation, extraction, spectrograms
  detection/      aircraft sound classification
  doppler/        ridge tracking, curve fitting, artifact rejection
  propagation/    retarded-time Doppler simulation
  flight_data/    provider adapters, fusion, trajectory reconstruction
  geolocation/    likelihood, search, regions, status rules
  validation/     synthetic data generator and benchmark
frontend/         React and TypeScript dashboard
docker/           Dockerfiles and nginx config
scripts/          setup, demo data, stack smoke test
flight_datasets/  your local flight data files
docs/             architecture, providers, validation
```

[docs/architecture.md](docs/architecture.md) explains the physics, the statistics and why each rule exists.

## Known limitations

- Not validated on real recordings, and no provider has been tested with a live key.
- If neither aircraft heard is in the flight data, two curves can occasionally match the wrong flights. This gave a confident wrong location in 1 of 20 synthetic trials of that case. A tight region and time window reduce the risk.
- Sound travels in straight lines in the model, with no refraction or ground reflection. Temperature follows a standard profile and wind is uniform.
- Supersonic aircraft are not modelled.
- There are no database migrations yet, so schema changes need a manual step.
- No multi-factor authentication or password reset.

## Credits

- Sound classification uses the [Audio Spectrogram Transformer](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) fine-tuned on AudioSet. It is downloaded at runtime under its own license.
- Map tiles are from [OpenFreeMap](https://openfreemap.org/), with data © OpenStreetMap contributors.
- The classifier check in the validation report used clips from the [ESC-50 dataset](https://github.com/karolpiczak/ESC-50) (Piczak, 2015). The clips are not included in this repository.

## License

[MIT](LICENSE)
