export type LocalizationStatus = "localized" | "region_only" | "ambiguous" | "not_localizable";

export interface Region {
  rank: number;
  probability: number;
  latitude: number;
  longitude: number;
  area_km2: number;
  radius_m: number;
  semi_major_m: number;
  semi_minor_m: number;
  polygons: [number, number][][];
}

export interface FlightMatch {
  flight_key: string;
  event_id: string;
  probability: number;
  predicted_times: number[];
  predicted_frequency_hz: number[];
  observed_frequency_hz: number[];
  source_frequency_hz: number | null;
  closest_range_m: number;
}

export interface Candidate {
  key: string;
  identity: {
    icao24: string | null;
    callsign: string | null;
    flight_number: string | null;
    registration: string | null;
    aircraft_type: string | null;
  };
  providers: string[];
  provider_agreement_is_independent: boolean;
  independent_signal_sources: number;
  conflicts: string[];
  completeness: number;
  observation_count: number;
  path: { times: number[]; latitude: number[]; longitude: number[]; altitude_m: number[] };
}

export interface ComparisonRow {
  label: string;
  status: LocalizationStatus;
  credible_area_km2: number;
  region_count: number;
  located_probability: number;
}

export interface LocalizationPayload {
  status: LocalizationStatus;
  reasons: string[];
  located_probability: number;
  regions: Region[];
  latitude: number | null;
  longitude: number | null;
  coordinate_decimals: number | null;
  uncertainty_radius_m: number | null;
  matches: FlightMatch[];
  untracked_probability: Record<string, number>;
  associations: Record<string, Record<string, number>>;
  rejected_pairs: { event_id: string; flight_key: string; reason: string }[];
  recording_start_utc: number | null;
  recording_start_sigma_s: number | null;
  sensitivity_sigma_m: number | null;
  comparison: ComparisonRow[];
  diagnostics: Record<string, unknown>;
  calibration_note: string;
  candidates: Candidate[];
  assumptions: Record<string, number | boolean>;
}

export interface DopplerCurveView {
  event_id: string;
  detection_id: string | null;
  times_s: number[];
  frequency_hz: number[];
  sigma_hz: number[];
  start_s: number;
  end_s: number;
  harmonic_numbers: number[];
  median_snr_db: number;
  curve_class: string;
  quality: number;
  usable: boolean;
  closest_approach_s: number | null;
  closest_approach_sigma_s: number | null;
  implied_speed_mps: number | null;
  implied_slant_range_m: number | null;
  source_frequency_hz: number | null;
}

export interface DetectionView {
  detection_id: string;
  start_s: number;
  end_s: number;
  peak_s: number;
  score: number;
  score_kind: string;
  classifier: string;
  category: string;
  dominant_frequencies_hz: number[];
  snr_db: number;
}

export interface AnalysisSummary {
  id: string;
  filename: string;
  state: string;
  stage: string;
  progress: number;
  error: string | null;
  localization_status: LocalizationStatus | null;
  created_at: string;
}

export interface AnalysisDetail extends AnalysisSummary {
  request: { region: BoxInput; start_earliest_utc: number; start_latest_utc: number };
  media: Record<string, unknown> | null;
  acoustic: {
    duration_s: number;
    sample_rate: number;
    classifier: string;
    classifier_note: string | null;
    detections: DetectionView[];
    curves: DopplerCurveView[];
  } | null;
  result: LocalizationPayload | null;
  authorization_basis: string;
}

export interface BoxInput {
  south: number;
  west: number;
  north: number;
  east: number;
}

export interface ProviderStatus {
  name: string;
  configured: boolean;
  available: boolean;
  capabilities: string[];
  detail: string;
}

export interface WorkloadEstimate {
  region_area_km2: number;
  time_window_s: number;
  provider_requests: Record<string, number>;
  initial_search_tiles: number;
  feasible: boolean;
  advice: string[];
}

export interface ProgressMessage {
  state: string;
  stage: string;
  progress: number;
  error: string | null;
  localization_status: LocalizationStatus | null;
}
