# Flight data providers

Checked in November 2026. Plans and limits change, so confirm them against each provider before relying on this page. No provider has been exercised with a live key yet. All adapters are tested against mocked responses that follow the published schemas.

## Capabilities

| Provider | Historical area search | Historical track | Credentials |
|---|---|---|---|
| Flightradar24 | Yes, as point-in-time snapshots | Yes, by flight ID | Bearer token and a plan with enough history |
| ADS-B Exchange | No | Yes, by ICAO hex | `x-api-key` on a plan that includes traces |
| OpenSky | Last hour only | Last 30 days, experimental | OAuth2 client credentials |
| Local dataset | Yes | Yes | None |

This shapes how they are used. An analysis needs an area search to find candidate aircraft, then track lookups to get detail. **ADS-B Exchange cannot discover candidates in the past**, so it needs Flightradar24, OpenSky or a local dataset alongside it.

## Flightradar24

- Base URL `https://fr24api.flightradar24.com/api`, headers `Accept-Version: v1` and `Authorization: Bearer`
- `GET /historic/flight-positions/full` with `timestamp`, `bounds` as `north,south,west,east`, and `limit`
- `GET /flight-tracks` with `flight_id`
- History reaches back to 2016 depending on plan, and requests consume credits
- Altitude is treated as barometric. `ESTIMATED` points are marked inferred and left out of reconstruction.
- Snapshots are taken every `AEROLOCATE_FR24_SNAPSHOT_STEP_S` seconds. The upload page shows the request count before you start.

The public documentation pages render client-side and could not be read directly, so this schema comes from Flightradar24's own [`fr24api-mcp`](https://github.com/Flightradar24/fr24api-mcp) client. The `/usage` call behind the availability check is not in that client and is unverified.

## ADS-B Exchange

- Base URL `https://gateway.adsbexchange.com/api/aircraft/v2`
- `GET /traces-hist/{year}/{month}/{day}/traces/{last two hex}/trace_full_{hex}.json`
- Rate limits use a token bucket shared across all endpoints on a key
- Commercial use needs written authorization, and historical backfills are for subscription customers

Trace rows are positional. The parser reads flags for stale positions and geometric altitude, and prefers geometric altitude when a row includes it.

## OpenSky

- OAuth2 client credentials only; basic authentication is no longer accepted
- `/states/all` reaches back one hour, and older requests return HTTP 400
- `/tracks/all` is documented as experimental, has no speed data and stops at 30 days
- Older data requires OpenSky's Trino access, which is not implemented. Exports can be loaded through the local dataset provider.

## Local datasets

Files in `flight_datasets/`, or `AEROLOCATE_LOCAL_FLIGHT_DATASET_DIR` outside Docker, are loaded on each search. Hidden files are ignored.

- `.jsonl` or `.csv`, optionally gzipped, with columns `icao24`, `callsign`, `timestamp`, `latitude`, `longitude`, `altitude_m`, `altitude_kind`, `ground_speed_mps`, `track_deg`, `vertical_rate_mps`, `on_ground`, `source_type`, `origin`
- ADS-B Exchange `trace_full` JSON files, gzipped or plain

ADS-B Exchange publishes complete data for the first day of each month at no charge for evaluation, which makes this the cheapest way to test against real tracks.

## Licensing and retention

Cached responses expire after `AEROLOCATE_FLIGHT_CACHE_TTL_HOURS`. Provider observations stored with an analysis follow `AEROLOCATE_RETENTION_DAYS`. Check that both meet your provider's terms. No adapter scrapes restricted endpoints.
