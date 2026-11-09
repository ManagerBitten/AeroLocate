import { useEffect, useMemo, useState } from "react";
import { useProgress } from "../hooks/useProgress";
import { ResultMap } from "../maps/ResultMap";
import {
  cancelAnalysis,
  deleteAnalysis,
  downloadReport,
  fetchBlobUrl,
  getAnalysis,
  getSpectrogram,
  getWaveform,
} from "../services/api";
import {
  formatArea,
  formatDistance,
  formatProbability,
  formatUtc,
  SERIES_COLORS,
  STATUS_EXPLANATIONS,
  STATUS_LABELS,
} from "../services/format";
import type { AnalysisDetail, DopplerCurveView, LocalizationPayload } from "../services/types";
import { DopplerChart, PredictionChart, SpectrogramChart, WaveformChart } from "../visualizations/AudioCharts";

const TABS = ["Results", "Audio", "Flights", "Map"] as const;
const TERMINAL = ["completed", "failed", "cancelled", "rejected"];

type Tab = (typeof TABS)[number];

interface AnalysisPageProps {
  analysisId: string;
  onDeleted: () => void;
  onStateChanged: () => void;
}

export function AnalysisPage({ analysisId, onDeleted, onStateChanged }: AnalysisPageProps) {
  const [analysis, setAnalysis] = useState<AnalysisDetail | null>(null);
  const [tab, setTab] = useState<Tab>("Results");
  const [error, setError] = useState<string | null>(null);
  const progress = useProgress(analysisId);
  const state = progress?.state ?? analysis?.state;

  useEffect(() => {
    getAnalysis(analysisId)
      .then(setAnalysis)
      .catch((failure: Error) => setError(failure.message));
  }, [analysisId, state]);

  useEffect(() => {
    if (state) onStateChanged();
  }, [state, onStateChanged]);

  if (error) return <p className="text-[#e66767]">{error}</p>;
  if (!analysis) return <p className="text-ink-muted">Loading analysis.</p>;

  const running = state !== undefined && !TERMINAL.includes(state);

  return (
    <div className="mx-auto max-w-6xl space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="min-w-0">
          <h1 className="truncate text-lg font-semibold">{analysis.filename}</h1>
          <p className="text-xs text-ink-muted">Authorization basis: {analysis.authorization_basis.replaceAll("_", " ")}</p>
        </div>
        <div className="flex flex-wrap gap-2">
          {running && (
            <button className="button-quiet" onClick={() => void cancelAnalysis(analysisId)}>
              Cancel
            </button>
          )}
          {analysis.result && (
            <>
              <button className="button-quiet" onClick={() => void downloadReport(analysisId, "json")}>
                Export JSON
              </button>
              <button className="button-quiet" onClick={() => void downloadReport(analysisId, "pdf")}>
                Export PDF
              </button>
            </>
          )}
          {!running && (
            <button
              className="button-quiet"
              onClick={() => {
                if (window.confirm("Permanently delete this analysis and its files?")) {
                  void deleteAnalysis(analysisId).then(onDeleted);
                }
              }}
            >
              Delete
            </button>
          )}
        </div>
      </div>

      {running && (
        <div className="panel">
          <div className="mb-2 flex justify-between text-sm">
            <span>{progress?.stage ?? analysis.stage}</span>
            <span className="text-ink-muted">{Math.round((progress?.progress ?? analysis.progress) * 100)}%</span>
          </div>
          <div className="h-1.5 overflow-hidden rounded bg-surface-sunken">
            <div className="h-full bg-accent transition-all" style={{ width: `${(progress?.progress ?? analysis.progress) * 100}%` }} />
          </div>
        </div>
      )}

      {analysis.error && <div className="panel text-sm text-[#e66767]">{analysis.error}</div>}

      <nav className="flex gap-1 border-b border-surface-line" role="tablist">
        {TABS.map((name) => (
          <button
            key={name}
            role="tab"
            aria-selected={tab === name}
            className={`border-b-2 px-3 py-2 text-sm ${tab === name ? "border-accent text-ink" : "border-transparent text-ink-muted hover:text-ink"}`}
            onClick={() => setTab(name)}
          >
            {name}
          </button>
        ))}
      </nav>

      {tab === "Results" && <ResultsTab analysis={analysis} />}
      {tab === "Audio" && <AudioTab analysis={analysis} ready={!running || Boolean(analysis.acoustic)} />}
      {tab === "Flights" && <FlightsTab result={analysis.result} />}
      {tab === "Map" && <MapTab analysis={analysis} />}
    </div>
  );
}

function Empty({ text }: { text: string }) {
  return <p className="panel text-sm text-ink-muted">{text}</p>;
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-surface-line bg-surface-sunken p-3">
      <p className="text-xs text-ink-muted">{label}</p>
      <p className="mt-1 font-mono text-lg">{value}</p>
    </div>
  );
}

function ResultsTab({ analysis }: { analysis: AnalysisDetail }) {
  const result = analysis.result;
  if (!result) return <Empty text="Results appear here when the analysis finishes." />;
  return (
    <div className="space-y-4">
      <section className="panel space-y-3">
        <div>
          <p className="text-xs uppercase tracking-wide text-ink-muted">Outcome</p>
          <h2 className="text-2xl font-semibold">{STATUS_LABELS[result.status]}</h2>
          <p className="text-sm text-ink-soft">{STATUS_EXPLANATIONS[result.status]}</p>
        </div>
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
          <Stat label="Match probability, if the aircraft is in the flight data" value={formatProbability(result.located_probability)} />
          <Stat
            label="Estimate"
            value={
              result.latitude !== null && result.longitude !== null
                ? `${result.latitude.toFixed(result.coordinate_decimals ?? 2)}, ${result.longitude.toFixed(result.coordinate_decimals ?? 2)}`
                : "None justified"
            }
          />
          <Stat label="Uncertainty radius" value={result.uncertainty_radius_m === null ? "Not available" : formatDistance(result.uncertainty_radius_m)} />
          <Stat label="Candidate regions" value={String(result.regions.length)} />
        </div>
        <ul className="list-disc space-y-1 pl-5 text-sm text-ink-soft">
          {result.reasons.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ul>
        <p className="rounded-md border border-surface-line bg-surface-sunken p-3 text-xs text-ink-soft">{result.calibration_note}</p>
      </section>

      {result.regions.length > 0 && (
        <section className="panel overflow-x-auto">
          <h3 className="mb-2 font-semibold">Candidate regions</h3>
          <table className="w-full text-left text-sm">
            <thead className="text-xs text-ink-muted">
              <tr>
                <th className="py-1 pr-4 font-medium">Rank</th>
                <th className="py-1 pr-4 font-medium">Posterior share</th>
                <th className="py-1 pr-4 font-medium">Centre</th>
                <th className="py-1 pr-4 font-medium">Credible area</th>
                <th className="py-1 font-medium">Extent</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {result.regions.map((region) => (
                <tr key={region.rank} className="border-t border-surface-line">
                  <td className="py-1 pr-4">{region.rank}</td>
                  <td className="py-1 pr-4">{formatProbability(region.probability)}</td>
                  <td className="py-1 pr-4">
                    {region.latitude.toFixed(3)}, {region.longitude.toFixed(3)}
                  </td>
                  <td className="py-1 pr-4">{formatArea(region.area_km2)}</td>
                  <td className="py-1">
                    {formatDistance(region.semi_major_m)} × {formatDistance(region.semi_minor_m)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {result.comparison.length > 0 && (
        <section className="panel overflow-x-auto">
          <h3 className="mb-2 font-semibold">One aircraft against all</h3>
          <table className="w-full text-left text-sm">
            <thead className="text-xs text-ink-muted">
              <tr>
                <th className="py-1 pr-4 font-medium">Evidence</th>
                <th className="py-1 pr-4 font-medium">Outcome</th>
                <th className="py-1 pr-4 font-medium">Credible area</th>
                <th className="py-1 font-medium">Regions</th>
              </tr>
            </thead>
            <tbody>
              {result.comparison.map((row) => (
                <tr key={row.label} className="border-t border-surface-line">
                  <td className="py-1 pr-4">{row.label}</td>
                  <td className="py-1 pr-4">{STATUS_LABELS[row.status]}</td>
                  <td className="py-1 pr-4 font-mono">{formatArea(row.credible_area_km2)}</td>
                  <td className="py-1 font-mono">{row.region_count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {result.recording_start_utc !== null && (
        <section className="panel text-sm">
          <h3 className="mb-1 font-semibold">Assumptions behind this result</h3>
          <p className="text-ink-soft">
            Recording start {formatUtc(result.recording_start_utc)}
            {result.recording_start_sigma_s !== null && ` ± ${result.recording_start_sigma_s.toFixed(1)} s`}. Surface temperature{" "}
            {String(result.assumptions.temperature_c)} °C{result.assumptions.temperature_is_assumed ? " (assumed)" : ""}, wind{" "}
            {result.assumptions.wind_is_assumed ? "assumed calm" : "as supplied"}.
            {result.sensitivity_sigma_m !== null &&
              ` Varying these assumptions moves the solution by about ${formatDistance(result.sensitivity_sigma_m)}.`}
          </p>
        </section>
      )}
    </div>
  );
}

function curveLabel(curve: DopplerCurveView): string {
  return curve.curve_class.replaceAll("_", " ");
}

function AudioTab({ analysis, ready }: { analysis: AnalysisDetail; ready: boolean }) {
  const [waveform, setWaveform] = useState<Awaited<ReturnType<typeof getWaveform>> | null>(null);
  const [spectrogram, setSpectrogram] = useState<Awaited<ReturnType<typeof getSpectrogram>> | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const acoustic = analysis.acoustic;

  useEffect(() => {
    if (!ready) return;
    void getWaveform(analysis.id).then(setWaveform).catch(() => setWaveform(null));
    void getSpectrogram(analysis.id).then(setSpectrogram).catch(() => setSpectrogram(null));
    let url: string | null = null;
    void fetchBlobUrl(`/api/analyses/${analysis.id}/audio`)
      .then((value) => {
        url = value;
        setAudioUrl(value);
      })
      .catch(() => setAudioUrl(null));
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  }, [analysis.id, ready]);

  const curves = acoustic?.curves ?? [];
  const usable = useMemo(() => curves.filter((curve) => curve.usable), [curves]);
  const focus = curves.find((curve) => curve.event_id === selected) ?? usable[0] ?? curves[0];
  const maxFrequency = useMemo(() => {
    const top = Math.max(200, ...usable.flatMap((curve) => curve.frequency_hz));
    return Math.min(4000, top * 1.6);
  }, [usable]);

  if (!ready || !acoustic) return <Empty text="Audio analysis is still running." />;

  return (
    <div className="space-y-4">
      {audioUrl && <audio src={audioUrl} controls className="w-full" />}
      {waveform && (
        <section className="panel">
          <h3 className="mb-2 font-semibold">Waveform</h3>
          <WaveformChart waveform={waveform} detections={acoustic.detections} />
        </section>
      )}
      {spectrogram && (
        <section className="panel">
          <h3 className="mb-2 font-semibold">Spectrogram and tracked curves</h3>
          <SpectrogramChart spectrogram={spectrogram} curves={usable} maxFrequency={maxFrequency} />
        </section>
      )}
      <section className="panel overflow-x-auto">
        <h3 className="mb-2 font-semibold">Detected aircraft sound</h3>
        {acoustic.classifier_note && <p className="mb-2 text-sm text-ink-soft">{acoustic.classifier_note}</p>}
        {acoustic.detections.length === 0 && <p className="text-sm text-ink-muted">No aircraft-like sound event was detected.</p>}
        <table className="w-full text-left text-sm">
          <tbody>
            {acoustic.detections.map((event) => (
              <tr key={event.detection_id} className="border-t border-surface-line">
                <td className="py-1 pr-4 font-mono">{event.detection_id}</td>
                <td className="py-1 pr-4">{event.category.replaceAll("_", " ")}</td>
                <td className="py-1 pr-4 font-mono">
                  {event.start_s.toFixed(0)}–{event.end_s.toFixed(0)} s
                </td>
                <td className="py-1 pr-4 font-mono">{event.snr_db.toFixed(0)} dB</td>
                <td className="py-1 text-ink-muted">
                  {event.score_kind === "heuristic" ? "Not classified" : `${event.score.toFixed(2)} model score`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      <section className="panel space-y-3">
        <h3 className="font-semibold">Doppler curves</h3>
        {curves.length === 0 && <p className="text-sm text-ink-muted">No sustained tonal track was found.</p>}
        <div className="flex flex-wrap gap-2">
          {curves.map((curve) => (
            <button
              key={curve.event_id}
              className={curve === focus ? "button" : "button-quiet"}
              onClick={() => setSelected(curve.event_id)}
            >
              {curve.event_id} · {curveLabel(curve)}
            </button>
          ))}
        </div>
        {focus && (
          <>
            <DopplerChart curve={focus} color={SERIES_COLORS[Math.max(usable.indexOf(focus), 0) % SERIES_COLORS.length]} />
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
              <Stat label="Curve quality" value={focus.quality.toFixed(2)} />
              <Stat label="Median SNR" value={`${focus.median_snr_db.toFixed(0)} dB`} />
              <Stat label="Implied speed" value={focus.implied_speed_mps === null ? "None" : `${focus.implied_speed_mps.toFixed(0)} m/s`} />
              <Stat label="Implied slant range" value={focus.implied_slant_range_m === null ? "None" : formatDistance(focus.implied_slant_range_m)} />
            </div>
          </>
        )}
      </section>
    </div>
  );
}

function FlightsTab({ result }: { result: LocalizationPayload | null }) {
  const [selected, setSelected] = useState<string | null>(null);
  if (!result) return <Empty text="Flight correlation appears here when the analysis finishes." />;
  if (result.candidates.length === 0) return <Empty text="No flight trajectories were retrieved for this region and time." />;
  const match = result.matches.find((item) => item.flight_key === selected) ?? result.matches[0];

  return (
    <div className="space-y-4">
      <section className="panel overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="text-xs text-ink-muted">
            <tr>
              <th className="py-1 pr-4 font-medium">Aircraft</th>
              <th className="py-1 pr-4 font-medium">Flight</th>
              <th className="py-1 pr-4 font-medium">Type</th>
              <th className="py-1 pr-4 font-medium">Match</th>
              <th className="py-1 pr-4 font-medium">Track completeness</th>
              <th className="py-1 font-medium">Providers</th>
            </tr>
          </thead>
          <tbody>
            {result.candidates.map((candidate) => {
              const scores = Object.values(result.associations).map((row) => row[candidate.key] ?? 0);
              const best = scores.length ? Math.max(...scores) : 0;
              return (
                <tr
                  key={candidate.key}
                  className="cursor-pointer border-t border-surface-line hover:bg-surface-raised"
                  onClick={() => setSelected(candidate.key)}
                >
                  <td className="py-1 pr-4 font-mono">{candidate.identity.icao24 ?? candidate.key}</td>
                  <td className="py-1 pr-4">{candidate.identity.flight_number ?? candidate.identity.callsign ?? "Unknown"}</td>
                  <td className="py-1 pr-4">{candidate.identity.aircraft_type ?? "Unknown"}</td>
                  <td className="py-1 pr-4 font-mono">{formatProbability(best)}</td>
                  <td className="py-1 pr-4 font-mono">{Math.round(candidate.completeness * 100)}%</td>
                  <td className="py-1 text-ink-soft">
                    {candidate.providers.join(", ")}
                    {candidate.providers.length > 1 && !candidate.provider_agreement_is_independent && (
                      <span className="block text-xs text-ink-muted">Same underlying ADS-B data, not independent</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>
      {match ? (
        <section className="panel">
          <h3 className="mb-2 font-semibold">
            {match.event_id} against {match.flight_key}
          </h3>
          <PredictionChart times={match.predicted_times} observed={match.observed_frequency_hz} predicted={match.predicted_frequency_hz} />
        </section>
      ) : (
        <Empty text="No flight was matched to an acoustic event." />
      )}
      {result.rejected_pairs.length > 0 && (
        <details className="panel text-sm">
          <summary className="cursor-pointer">Rejected pairings ({result.rejected_pairs.length})</summary>
          <ul className="mt-2 space-y-1 text-ink-soft">
            {result.rejected_pairs.map((item) => (
              <li key={`${item.event_id}-${item.flight_key}`}>
                {item.event_id} / {item.flight_key}: {item.reason}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

function MapTab({ analysis }: { analysis: AnalysisDetail }) {
  const result = analysis.result;
  const box = useMemo<[number, number, number, number]>(() => {
    const region = analysis.request.region;
    return [region.west, region.south, region.east, region.north];
  }, [analysis.request.region]);
  if (!result) return <Empty text="The map appears here when the analysis finishes." />;
  return (
    <div className="space-y-2">
      <ResultMap result={result} searchBox={box} selectedFlight={null} />
      <p className="text-xs text-ink-muted">
        Blue areas are 95% credible regions, darker where the posterior share is higher. Orange lines are matched tracks and grey lines are other
        candidates. A pin appears only for a localized result.
      </p>
    </div>
  );
}
