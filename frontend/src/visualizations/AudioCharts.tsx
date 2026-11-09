import { useMemo } from "react";
import { SERIES_COLORS } from "../services/format";
import type { DetectionView, DopplerCurveView } from "../services/types";
import { AXIS, Plot } from "./Plot";

interface WaveformProps {
  waveform: { minimum: number[]; maximum: number[]; seconds_per_point: number };
  detections: DetectionView[];
}

export function WaveformChart({ waveform, detections }: WaveformProps) {
  const data = useMemo(() => {
    const times = waveform.minimum.map((_, index) => index * waveform.seconds_per_point);
    return [
      {
        x: [...times, ...[...times].reverse()],
        y: [...waveform.maximum, ...[...waveform.minimum].reverse()],
        fill: "toself",
        mode: "lines",
        line: { width: 0 },
        fillcolor: SERIES_COLORS[0],
        hoverinfo: "skip",
        name: "Waveform",
      },
    ];
  }, [waveform]);
  const layout = useMemo(
    () => ({
      showlegend: false,
      xaxis: { ...AXIS, title: { text: "Recording time (s)" } },
      yaxis: { ...AXIS, title: { text: "Amplitude" } },
      shapes: detections.map((event) => ({
        type: "rect",
        xref: "x",
        yref: "paper",
        x0: event.start_s,
        x1: event.end_s,
        y0: 0,
        y1: 1,
        fillcolor: "rgba(240,239,236,0.07)",
        line: { width: 0 },
      })),
    }),
    [detections],
  );
  return <Plot data={data} layout={layout} height={200} label="Recording waveform with detected aircraft events shaded" />;
}

interface SpectrogramProps {
  spectrogram: { times: number[]; frequencies: number[]; decibels: number[][] };
  curves: DopplerCurveView[];
  maxFrequency: number;
}

export function SpectrogramChart({ spectrogram, curves, maxFrequency }: SpectrogramProps) {
  const floor = useMemo(() => {
    const values = spectrogram.decibels.flat().sort((a, b) => a - b);
    return values[Math.floor(values.length * 0.5)];
  }, [spectrogram]);
  const data = useMemo(
    () => [
      {
        type: "heatmap",
        x: spectrogram.times,
        y: spectrogram.frequencies,
        z: spectrogram.decibels,
        colorscale: [
          [0, "#131312"],
          [0.5, "#184f95"],
          [1, "#f0efec"],
        ],
        zmin: floor,
        colorbar: { title: { text: "dB" }, thickness: 10, outlinewidth: 0 },
        hovertemplate: "%{x:.1f} s<br>%{y:.0f} Hz<br>%{z:.0f} dB<extra></extra>",
      },
      ...curves.map((curve, index) => ({
        x: curve.times_s,
        y: curve.frequency_hz,
        mode: "lines",
        name: curve.event_id,
        line: { color: SERIES_COLORS[index % SERIES_COLORS.length], width: 2 },
        hovertemplate: `${curve.event_id}<br>%{x:.1f} s, %{y:.1f} Hz<extra></extra>`,
      })),
    ],
    [spectrogram, curves, floor],
  );
  const layout = useMemo(
    () => ({
      xaxis: { ...AXIS, title: { text: "Recording time (s)" } },
      yaxis: { ...AXIS, title: { text: "Frequency (Hz)" }, range: [0, maxFrequency] },
    }),
    [maxFrequency],
  );
  return <Plot data={data} layout={layout} height={380} label="Spectrogram with extracted frequency tracks overlaid" />;
}

export function DopplerChart({ curve, color }: { curve: DopplerCurveView; color: string }) {
  const data = useMemo(
    () => [
      {
        x: curve.times_s,
        y: curve.frequency_hz,
        mode: "lines+markers",
        marker: { size: 4, color },
        line: { color, width: 2 },
        error_y: { type: "data", array: curve.sigma_hz, visible: true, color, thickness: 1, width: 0 },
        name: curve.event_id,
        hovertemplate: "%{x:.1f} s<br>%{y:.2f} Hz<extra></extra>",
      },
    ],
    [curve, color],
  );
  const layout = useMemo(
    () => ({
      showlegend: false,
      xaxis: { ...AXIS, title: { text: "Recording time (s)" } },
      yaxis: { ...AXIS, title: { text: "Tracked frequency (Hz)" } },
      shapes:
        curve.closest_approach_s === null
          ? []
          : [
              {
                type: "line",
                x0: curve.closest_approach_s,
                x1: curve.closest_approach_s,
                yref: "paper",
                y0: 0,
                y1: 1,
                line: { color: "#8f8e85", width: 1, dash: "dot" },
              },
            ],
    }),
    [curve],
  );
  return <Plot data={data} layout={layout} height={240} label={`Extracted Doppler curve for event ${curve.event_id}`} />;
}

interface ComparisonProps {
  times: number[];
  observed: number[];
  predicted: number[];
}

export function PredictionChart({ times, observed, predicted }: ComparisonProps) {
  const data = useMemo(
    () => [
      {
        x: times,
        y: observed,
        mode: "markers",
        name: "Observed",
        marker: { size: 6, color: SERIES_COLORS[0] },
        hovertemplate: "Observed %{y:.2f} Hz<extra></extra>",
      },
      {
        x: times,
        y: predicted,
        mode: "lines",
        name: "Predicted from track",
        line: { color: SERIES_COLORS[1], width: 2 },
        hovertemplate: "Predicted %{y:.2f} Hz<extra></extra>",
      },
    ],
    [times, observed, predicted],
  );
  const layout = useMemo(
    () => ({
      hovermode: "x unified",
      xaxis: { ...AXIS, title: { text: "Recording time (s)" } },
      yaxis: { ...AXIS, title: { text: "Frequency (Hz)" } },
    }),
    [],
  );
  return <Plot data={data} layout={layout} height={300} label="Observed Doppler curve against the curve predicted from the flight track" />;
}
