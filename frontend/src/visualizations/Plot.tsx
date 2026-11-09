import Plotly from "plotly.js-dist-min";
import { useEffect, useRef } from "react";

export const CHART_LAYOUT = {
  paper_bgcolor: "#1a1a19",
  plot_bgcolor: "#1a1a19",
  font: { color: "#c3c2b7", size: 12 },
  margin: { l: 58, r: 16, t: 12, b: 42 },
  hoverlabel: { bgcolor: "#232322", bordercolor: "#34342f", font: { color: "#f0efec" } },
  legend: { orientation: "h", y: 1.1, x: 0, bgcolor: "rgba(0,0,0,0)" },
};

export const AXIS = {
  gridcolor: "#2a2a27",
  zerolinecolor: "#34342f",
  linecolor: "#34342f",
  tickcolor: "#34342f",
};

interface PlotProps {
  data: unknown[];
  layout: Record<string, unknown>;
  height: number;
  label: string;
}

export function Plot({ data, layout, height, label }: PlotProps) {
  const element = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const target = element.current;
    if (!target) return;
    void Plotly.react(target, data, { ...CHART_LAYOUT, ...layout, height }, { responsive: true, displaylogo: false });
    return () => Plotly.purge(target);
  }, [data, layout, height]);

  return <div ref={element} role="img" aria-label={label} style={{ height }} />;
}
