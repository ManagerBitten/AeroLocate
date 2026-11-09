import type {
  AnalysisDetail,
  AnalysisSummary,
  BoxInput,
  ProviderStatus,
  WorkloadEstimate,
} from "./types";

const TOKEN_KEY = "aerolocate.token";

export function readToken(): string | null {
  try {
    return sessionStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function storeToken(token: string | null): void {
  try {
    if (token) sessionStorage.setItem(TOKEN_KEY, token);
    else sessionStorage.removeItem(TOKEN_KEY);
  } catch {
    return;
  }
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function request(path: string, init: RequestInit = {}): Promise<Response> {
  const token = readToken();
  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(path, { ...init, headers });
  if (response.status === 401 && token) storeToken(null);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = typeof body?.detail === "string" ? body.detail : `Request failed (${response.status})`;
    throw new ApiError(detail, response.status);
  }
  return response;
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  return (await request(path, init)).json() as Promise<T>;
}

function postJson(body: unknown): RequestInit {
  return { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
}

export async function register(email: string, password: string): Promise<void> {
  await request("/api/auth/register", postJson({ email, password }));
}

export async function login(email: string, password: string): Promise<void> {
  const body = new URLSearchParams({ username: email, password });
  const result = await json<{ access_token: string }>("/api/auth/token", { method: "POST", body });
  storeToken(result.access_token);
}

export const listProviders = () => json<ProviderStatus[]>("/api/providers");
export const listAnalyses = () => json<AnalysisSummary[]>("/api/analyses");
export const getAnalysis = (id: string) => json<AnalysisDetail>(`/api/analyses/${id}`);
export const cancelAnalysis = (id: string) => json<AnalysisSummary>(`/api/analyses/${id}/cancel`, { method: "POST" });
export const getWaveform = (id: string) =>
  json<{ minimum: number[]; maximum: number[]; seconds_per_point: number }>(`/api/analyses/${id}/waveform`);
export const getSpectrogram = (id: string) =>
  json<{ times: number[]; frequencies: number[]; decibels: number[][] }>(`/api/analyses/${id}/spectrogram`);

export async function deleteAnalysis(id: string): Promise<void> {
  await request(`/api/analyses/${id}`, { method: "DELETE" });
}

export function estimateWorkload(body: {
  region: BoxInput;
  start_earliest_utc: number;
  start_latest_utc: number;
  providers: string[];
}): Promise<WorkloadEstimate> {
  return json<WorkloadEstimate>("/api/workload", postJson(body));
}

export function createAnalysis(file: File, parameters: unknown): Promise<AnalysisSummary> {
  const form = new FormData();
  form.append("file", file);
  form.append("parameters", JSON.stringify(parameters));
  return json<AnalysisSummary>("/api/analyses", { method: "POST", body: form });
}

export async function fetchBlobUrl(path: string): Promise<string> {
  const response = await request(path);
  return URL.createObjectURL(await response.blob());
}

export async function downloadReport(id: string, extension: "json" | "pdf"): Promise<void> {
  const url = await fetchBlobUrl(`/api/analyses/${id}/report.${extension}`);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `aerolocate-${id}.${extension}`;
  anchor.click();
  URL.revokeObjectURL(url);
}
