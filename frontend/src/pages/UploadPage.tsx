import { DragEvent, useEffect, useMemo, useState } from "react";
import { RegionPicker } from "../maps/RegionPicker";
import { createAnalysis, estimateWorkload, listProviders } from "../services/api";
import { formatArea, formatDuration, zonedToUtcSeconds } from "../services/format";
import type { BoxInput, ProviderStatus, WorkloadEstimate } from "../services/types";

const ACCEPTED = ".mp4,.mov,.mkv,.avi,.wav,.flac,.mp3,.m4a";
const VIDEO_PATTERN = /\.(mp4|mov|mkv|avi)$/i;
const TIME_MODES = [
  { id: "exact", label: "Exact time", hint: "Within about two seconds" },
  { id: "approximate", label: "Approximate", hint: "Set a margin in minutes" },
  { id: "date", label: "Date only", hint: "Searches the whole local day" },
] as const;
const AUTHORIZATION = [
  { id: "own_recording", label: "I made this recording" },
  { id: "consent_of_recorder", label: "I have the recorder's consent" },
  { id: "organizational_authority", label: "I am authorized by my organization" },
] as const;

const CONDITIONS = [
  { key: "temperature_c", label: "Temperature (°C)", placeholder: "15 assumed" },
  { key: "wind_east_mps", label: "Wind toward east (m/s)", placeholder: "Calm assumed" },
  { key: "wind_north_mps", label: "Wind toward north (m/s)", placeholder: "Calm assumed" },
  { key: "ground_elevation_m", label: "Ground elevation (m)", placeholder: "0 assumed" },
] as const;

const BOX_FIELDS = ["south", "west", "north", "east"] as const;

type TimeMode = (typeof TIME_MODES)[number]["id"];
type ConditionKey = (typeof CONDITIONS)[number]["key"];

export function UploadPage({ onCreated }: { onCreated: (id: string) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [date, setDate] = useState("");
  const [time, setTime] = useState("12:00:00");
  const [zone, setZone] = useState(Intl.DateTimeFormat().resolvedOptions().timeZone);
  const [mode, setMode] = useState<TimeMode>("exact");
  const [margin, setMargin] = useState(10);
  const [box, setBoxValue] = useState<BoxInput | null>(null);
  const [boxText, setBoxText] = useState<Record<keyof BoxInput, string>>({ south: "", west: "", north: "", east: "" });
  const [providers, setProviders] = useState<ProviderStatus[]>([]);
  const [chosen, setChosen] = useState<string[]>([]);
  const [preprocessing, setPreprocessing] = useState("none");
  const [authorization, setAuthorization] = useState("");
  const [note, setNote] = useState("");
  const [conditions, setConditions] = useState<Record<ConditionKey, string>>({
    temperature_c: "",
    wind_east_mps: "",
    wind_north_mps: "",
    ground_elevation_m: "",
  });
  const [workload, setWorkload] = useState<WorkloadEstimate | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const preview = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);
  useEffect(() => () => void (preview && URL.revokeObjectURL(preview)), [preview]);

  useEffect(() => {
    listProviders()
      .then((items) => {
        setProviders(items);
        setChosen(items.filter((item) => item.configured).map((item) => item.name));
      })
      .catch(() => setProviders([]));
  }, []);

  const window = useMemo(() => {
    if (!date) return null;
    if (mode === "date") {
      const start = zonedToUtcSeconds(date, "00:00:00", zone);
      return { start_earliest_utc: start, start_latest_utc: start + 86400 };
    }
    const centre = zonedToUtcSeconds(date, time.length === 5 ? `${time}:00` : time, zone);
    const half = mode === "exact" ? 2 : margin * 60;
    return { start_earliest_utc: centre - half, start_latest_utc: centre + half };
  }, [date, time, zone, mode, margin]);

  useEffect(() => {
    if (!box || !window) {
      setWorkload(null);
      return;
    }
    const handle = globalThis.setTimeout(() => {
      estimateWorkload({ region: box, ...window, providers: chosen })
        .then(setWorkload)
        .catch(() => setWorkload(null));
    }, 300);
    return () => globalThis.clearTimeout(handle);
  }, [box, window, chosen]);

  function setBox(next: BoxInput) {
    setBoxValue(next);
    setBoxText({
      south: next.south.toFixed(4),
      west: next.west.toFixed(4),
      north: next.north.toFixed(4),
      east: next.east.toFixed(4),
    });
  }

  function updateBox(field: keyof BoxInput, value: string) {
    const text = { ...boxText, [field]: value };
    setBoxText(text);
    const south = Number(text.south);
    const west = Number(text.west);
    const north = Number(text.north);
    const east = Number(text.east);
    const complete = Object.values(text).every((item) => item.trim() !== "");
    const valid = complete && south < north && west < east && south >= -90 && north <= 90 && west >= -180 && east <= 180;
    setBoxValue(valid ? { south, west, north, east } : null);
  }

  function drop(event: DragEvent) {
    event.preventDefault();
    setDragging(false);
    const dropped = event.dataTransfer.files[0];
    if (dropped) setFile(dropped);
  }

  async function start() {
    if (!file || !box || !window || !authorization) return;
    setBusy(true);
    setError(null);
    try {
      const created = await createAnalysis(file, {
        region: box,
        ...window,
        providers: chosen,
        preprocessing,
        ...Object.fromEntries(
          Object.entries(conditions)
            .filter(([, value]) => value.trim() !== "")
            .map(([key, value]) => [key, Number(value)]),
        ),
        authorization_basis: authorization,
        authorization_note: note,
      });
      onCreated(created.id);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  }

  const missing = [
    !file && "a recording",
    !window && "a recording date",
    !box && "a search region",
    !authorization && "your authorization basis",
  ].filter(Boolean);

  return (
    <div className="mx-auto grid max-w-6xl gap-4 lg:grid-cols-2">
      <section className="panel space-y-3">
        <h2 className="font-semibold">Recording</h2>
        <label
          className={`flex h-32 cursor-pointer items-center justify-center rounded-md border border-dashed p-4 text-center text-sm ${
            dragging ? "border-accent text-ink" : "border-surface-line text-ink-soft"
          }`}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={drop}
        >
          <input type="file" className="hidden" accept={ACCEPTED} onChange={(event) => setFile(event.target.files?.[0] ?? null)} />
          {file ? file.name : "Drop a video or audio file, or click to choose"}
        </label>
        {preview && file && VIDEO_PATTERN.test(file.name) && <video src={preview} controls className="max-h-60 w-full rounded-md" />}
        {preview && file && !VIDEO_PATTERN.test(file.name) && <audio src={preview} controls className="w-full" />}
        <div>
          <label className="label" htmlFor="preprocessing">
            Noise preprocessing
          </label>
          <select id="preprocessing" className="field" value={preprocessing} onChange={(event) => setPreprocessing(event.target.value)}>
            <option value="none">None</option>
            <option value="wind">Wind rumble filter</option>
            <option value="broadband">Wind and broadband noise reduction</option>
          </select>
          <p className="mt-1 text-xs text-ink-muted">The unmodified audio is always kept for comparison.</p>
        </div>
        <h2 className="pt-2 font-semibold">When it was recorded</h2>
        <div className="flex flex-wrap gap-2">
          {TIME_MODES.map((item) => (
            <button
              key={item.id}
              type="button"
              title={item.hint}
              className={mode === item.id ? "button" : "button-quiet"}
              onClick={() => setMode(item.id)}
            >
              {item.label}
            </button>
          ))}
        </div>
        <div className="grid gap-3 sm:grid-cols-3">
          <div>
            <label className="label" htmlFor="date">
              Date
            </label>
            <input id="date" className="field" type="date" value={date} onChange={(event) => setDate(event.target.value)} />
          </div>
          {mode !== "date" && (
            <div>
              <label className="label" htmlFor="time">
                Local time
              </label>
              <input id="time" className="field" type="time" step="1" value={time} onChange={(event) => setTime(event.target.value)} />
            </div>
          )}
          <div>
            <label className="label" htmlFor="zone">
              Timezone
            </label>
            <input id="zone" className="field" list="zones" value={zone} onChange={(event) => setZone(event.target.value)} />
            <datalist id="zones">
              {Intl.supportedValuesOf("timeZone").map((name) => (
                <option key={name} value={name} />
              ))}
            </datalist>
          </div>
        </div>
        {mode === "approximate" && (
          <div>
            <label className="label" htmlFor="margin">
              Margin (± minutes)
            </label>
            <input
              id="margin"
              className="field"
              type="number"
              min={1}
              max={720}
              value={margin}
              onChange={(event) => setMargin(Number(event.target.value))}
            />
          </div>
        )}
        <p className="text-xs text-ink-muted">Embedded file timestamps are shown as hints and never trusted automatically.</p>
        <h2 className="pt-2 font-semibold">Conditions at the time (optional)</h2>
        <div className="grid gap-3 sm:grid-cols-2">
          {CONDITIONS.map((item) => (
            <div key={item.key}>
              <label className="label" htmlFor={item.key}>
                {item.label}
              </label>
              <input
                id={item.key}
                className="field"
                type="number"
                step="any"
                placeholder={item.placeholder}
                value={conditions[item.key]}
                onChange={(event) => setConditions({ ...conditions, [item.key]: event.target.value })}
              />
            </div>
          ))}
        </div>
      </section>

      <section className="panel space-y-3">
        <h2 className="font-semibold">Where to search</h2>
        <RegionPicker value={box} onChange={setBox} />
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {BOX_FIELDS.map((field) => (
            <div key={field}>
              <label className="label" htmlFor={`box-${field}`}>
                {field[0].toUpperCase() + field.slice(1)}
              </label>
              <input
                id={`box-${field}`}
                className="field"
                type="number"
                step="any"
                value={boxText[field]}
                onChange={(event) => updateBox(field, event.target.value)}
              />
            </div>
          ))}
        </div>
        <h2 className="pt-2 font-semibold">Flight data</h2>
        {providers.length === 0 && <p className="text-sm text-ink-muted">Provider status is unavailable.</p>}
        <ul className="space-y-1">
          {providers.map((provider) => (
            <li key={provider.name} className="flex items-start gap-2 text-sm">
              <input
                id={provider.name}
                type="checkbox"
                className="mt-1"
                disabled={!provider.configured}
                checked={chosen.includes(provider.name)}
                onChange={(event) =>
                  setChosen(event.target.checked ? [...chosen, provider.name] : chosen.filter((name) => name !== provider.name))
                }
              />
              <label htmlFor={provider.name}>
                {provider.name}
                <span className="block text-xs text-ink-muted">{provider.detail}</span>
              </label>
            </li>
          ))}
        </ul>
        {workload && (
          <div className="rounded-md border border-surface-line bg-surface-sunken p-3 text-sm">
            <p className="font-medium">
              {workload.feasible ? "Search workload" : "These constraints are too loose"}
            </p>
            <p className="text-ink-soft">
              {formatArea(workload.region_area_km2)} over {formatDuration(workload.time_window_s)},{" "}
              {workload.initial_search_tiles.toLocaleString()} initial search cells,{" "}
              {Object.values(workload.provider_requests).reduce((sum, value) => sum + value, 0)} provider requests.
            </p>
            {workload.advice.map((line) => (
              <p key={line} className="mt-1 text-ink-soft">
                {line}
              </p>
            ))}
          </div>
        )}
        <h2 className="pt-2 font-semibold">Authorization</h2>
        <ul className="space-y-1">
          {AUTHORIZATION.map((item) => (
            <li key={item.id} className="flex items-center gap-2 text-sm">
              <input
                id={item.id}
                type="radio"
                name="authorization"
                checked={authorization === item.id}
                onChange={() => setAuthorization(item.id)}
              />
              <label htmlFor={item.id}>{item.label}</label>
            </li>
          ))}
        </ul>
        <div>
          <label className="label" htmlFor="note">
            Note for the audit log (optional)
          </label>
          <input id="note" className="field" maxLength={2000} value={note} onChange={(event) => setNote(event.target.value)} />
        </div>
        {error && <p className="text-sm text-[#e66767]">{error}</p>}
        <button className="button w-full" disabled={busy || missing.length > 0 || workload?.feasible === false} onClick={start}>
          {busy ? "Uploading" : "Start analysis"}
        </button>
        {missing.length > 0 && <p className="text-xs text-ink-muted">Still needed: {missing.join(", ")}.</p>}
      </section>
    </div>
  );
}
