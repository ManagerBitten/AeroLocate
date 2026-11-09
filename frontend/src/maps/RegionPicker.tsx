import maplibregl from "maplibre-gl";
import { useEffect, useRef, useState } from "react";
import type { BoxInput } from "../services/types";
import { MAP_STYLE_URL } from "./mapStyle";

interface RegionPickerProps {
  value: BoxInput | null;
  onChange: (box: BoxInput) => void;
}

function outline(box: BoxInput) {
  return {
    type: "Feature" as const,
    properties: {},
    geometry: {
      type: "Polygon" as const,
      coordinates: [
        [
          [box.west, box.south],
          [box.east, box.south],
          [box.east, box.north],
          [box.west, box.north],
          [box.west, box.south],
        ],
      ],
    },
  };
}

export function RegionPicker({ value, onChange }: RegionPickerProps) {
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const anchor = useRef<maplibregl.LngLat | null>(null);
  const [ready, setReady] = useState(false);
  const [drawing, setDrawing] = useState(false);
  const report = useRef(onChange);
  report.current = onChange;

  useEffect(() => {
    if (!container.current) return;
    const instance = new maplibregl.Map({
      container: container.current,
      style: MAP_STYLE_URL,
      center: [0, 30],
      zoom: 1.4,
      attributionControl: { compact: true },
    });
    instance.addControl(new maplibregl.NavigationControl({ showCompass: false }));
    instance.on("load", () => {
      instance.addSource("region", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      instance.addLayer({ id: "region-fill", type: "fill", source: "region", paint: { "fill-color": "#3987e5", "fill-opacity": 0.15 } });
      instance.addLayer({ id: "region-line", type: "line", source: "region", paint: { "line-color": "#3987e5", "line-width": 2 } });
      setReady(true);
    });
    instance.on("click", (event) => {
      if (!anchor.current) {
        anchor.current = event.lngLat;
        setDrawing(true);
        return;
      }
      const first = anchor.current;
      anchor.current = null;
      setDrawing(false);
      const box = {
        south: Math.min(first.lat, event.lngLat.lat),
        north: Math.max(first.lat, event.lngLat.lat),
        west: Math.min(first.lng, event.lngLat.lng),
        east: Math.max(first.lng, event.lngLat.lng),
      };
      if (box.south < box.north && box.west < box.east) report.current(box);
    });
    map.current = instance;
    return () => {
      instance.remove();
      map.current = null;
      setReady(false);
    };
  }, []);

  useEffect(() => {
    const source = map.current?.getSource("region") as maplibregl.GeoJSONSource | undefined;
    if (ready && source && value) source.setData(outline(value));
  }, [ready, value]);

  return (
    <div>
      <div ref={container} className="h-64 overflow-hidden rounded-md border border-surface-line" />
      <p className="mt-1 text-xs text-ink-muted">
        {drawing ? "Click the opposite corner." : "Click two opposite corners to set the search region."}
      </p>
    </div>
  );
}
