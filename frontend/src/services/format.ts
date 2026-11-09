import type { LocalizationStatus } from "./types";

export const SERIES_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];

export const STATUS_LABELS: Record<LocalizationStatus, string> = {
  localized: "Localized",
  region_only: "Broad region only",
  ambiguous: "Ambiguous",
  not_localizable: "Not localizable",
};

export const STATUS_EXPLANATIONS: Record<LocalizationStatus, string> = {
  localized: "One region holds the posterior and is small enough to report.",
  region_only: "One region holds the posterior, but it is too broad for a point estimate.",
  ambiguous: "Several places remain plausible, so no single estimate is given.",
  not_localizable: "The evidence does not support any location.",
};

export function zoneOffsetMinutes(utcMillis: number, timeZone: string): number {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone,
    hourCycle: "h23",
    year: "numeric",
    month: "numeric",
    day: "numeric",
    hour: "numeric",
    minute: "numeric",
    second: "numeric",
  }).formatToParts(new Date(utcMillis));
  const value = (type: string) => Number(parts.find((part) => part.type === type)?.value);
  const shown = Date.UTC(value("year"), value("month") - 1, value("day"), value("hour"), value("minute"), value("second"));
  return (shown - Math.floor(utcMillis / 1000) * 1000) / 60000;
}

export function zonedToUtcSeconds(date: string, time: string, timeZone: string): number {
  const [year, month, day] = date.split("-").map(Number);
  const [hour, minute, second = 0] = time.split(":").map(Number);
  const naive = Date.UTC(year, month - 1, day, hour, minute, second);
  const first = naive - zoneOffsetMinutes(naive, timeZone) * 60000;
  return (naive - zoneOffsetMinutes(first, timeZone) * 60000) / 1000;
}

export function formatDistance(meters: number): string {
  return meters >= 1000 ? `${(meters / 1000).toFixed(1)} km` : `${Math.round(meters)} m`;
}

export function formatArea(km2: number): string {
  return km2 >= 1 ? `${km2.toFixed(1)} km²` : `${(km2 * 100).toFixed(1)} ha`;
}

export function formatDuration(seconds: number): string {
  if (seconds < 90) return `${Math.round(seconds)} s`;
  if (seconds < 5400) return `${Math.round(seconds / 60)} min`;
  return `${(seconds / 3600).toFixed(1)} h`;
}

export function formatProbability(value: number): string {
  if (value > 0.995) return ">99%";
  if (value < 0.005) return "<1%";
  return `${Math.round(value * 100)}%`;
}

export function formatUtc(seconds: number): string {
  return `${new Date(Math.round(seconds) * 1000).toISOString().replace("T", " ").slice(0, 19)} UTC`;
}
