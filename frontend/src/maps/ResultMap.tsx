import maplibregl from "maplibre-gl";
import { useEffect, useRef, useState } from "react";
import { formatProbability, SERIES_COLORS } from "../services/format";
import type { LocalizationPayload } from "../services/types";
import { MAP_STYLE_URL } from "./mapStyle";

function boundsFor(result: LocalizationPayload, fallback: [number, number, number, number]): maplibregl.LngLatBounds {
  const bounds = new maplibregl.LngLatBounds();
  for (const region of result.regions) {
    for (const ring of region.polygons) for (const point of ring) bounds.extend(point);
  }
  if (bounds.isEmpty()) {
    bounds.extend([fallback[0], fallback[1]]);
    bounds.extend([fallback[2], fallback[3]]);
  }
  return bounds;
}

interface ResultMapProps {
  result: LocalizationPayload;
  searchBox: [number, number, number, number];
  selectedFlight: string | null;
}

export function ResultMap({ result, searchBox, selectedFlight }: ResultMapProps) {
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!container.current) return;
    const instance = new maplibregl.Map({
      container: container.current,
      style: MAP_STYLE_URL,
      bounds: boundsFor(result, searchBox),
      fitBoundsOptions: { padding: 60, maxZoom: 13 },
      attributionControl: { compact: true },
    });
    instance.addControl(new maplibregl.NavigationControl());
    instance.on("load", () => setReady(true));
    map.current = instance;
    return () => {
      instance.remove();
      map.current = null;
      setReady(false);
    };
  }, [result, searchBox]);

  useEffect(() => {
    const instance = map.current;
    if (!ready || !instance) return;
    const regions = {
      type: "FeatureCollection" as const,
      features: result.regions.map((region) => ({
        type: "Feature" as const,
        properties: { rank: region.rank, probability: region.probability, label: formatProbability(region.probability) },
        geometry: { type: "MultiPolygon" as const, coordinates: region.polygons.map((ring) => [ring]) },
      })),
    };
    const tracks = {
      type: "FeatureCollection" as const,
      features: result.candidates.map((candidate) => ({
        type: "Feature" as const,
        properties: {
          key: candidate.key,
          matched: result.matches.some((match) => match.flight_key === candidate.key) ? 1 : 0,
          selected: candidate.key === selectedFlight ? 1 : 0,
        },
        geometry: {
          type: "LineString" as const,
          coordinates: candidate.path.longitude.map((longitude, index) => [longitude, candidate.path.latitude[index]]),
        },
      })),
    };
    const closest = {
      type: "FeatureCollection" as const,
      features: result.matches.flatMap((match) => {
        const candidate = result.candidates.find((item) => item.key === match.flight_key);
        if (!candidate || result.recording_start_utc === null) return [];
        const curve = match.predicted_times;
        const middle = result.recording_start_utc + curve[Math.floor(curve.length / 2)];
        const times = candidate.path.times;
        let index = 0;
        for (let i = 0; i < times.length; i += 1) if (Math.abs(times[i] - middle) < Math.abs(times[index] - middle)) index = i;
        return [
          {
            type: "Feature" as const,
            properties: { key: match.flight_key },
            geometry: { type: "Point" as const, coordinates: [candidate.path.longitude[index], candidate.path.latitude[index]] },
          },
        ];
      }),
    };

    instance.addSource("tracks", { type: "geojson", data: tracks });
    instance.addSource("regions", { type: "geojson", data: regions });
    instance.addSource("aircraft", { type: "geojson", data: closest });
    instance.addLayer({
      id: "tracks",
      type: "line",
      source: "tracks",
      paint: {
        "line-color": ["case", ["==", ["get", "selected"], 1], "#f0efec", ["==", ["get", "matched"], 1], SERIES_COLORS[1], "#8f8e85"],
        "line-width": ["case", ["==", ["get", "selected"], 1], 3, ["==", ["get", "matched"], 1], 2, 1],
        "line-opacity": ["case", ["==", ["get", "matched"], 1], 0.95, 0.5],
      },
    });
    instance.addLayer({
      id: "regions-fill",
      type: "fill",
      source: "regions",
      paint: { "fill-color": SERIES_COLORS[0], "fill-opacity": ["interpolate", ["linear"], ["get", "probability"], 0, 0.12, 1, 0.55] },
    });
    instance.addLayer({ id: "regions-line", type: "line", source: "regions", paint: { "line-color": SERIES_COLORS[0], "line-width": 2 } });
    instance.addLayer({
      id: "aircraft",
      type: "circle",
      source: "aircraft",
      paint: { "circle-radius": 5, "circle-color": SERIES_COLORS[1], "circle-stroke-color": "#1a1a19", "circle-stroke-width": 2 },
    });
    instance.on("click", "regions-fill", (event) => {
      const feature = event.features?.[0];
      if (!feature) return;
      new maplibregl.Popup()
        .setLngLat(event.lngLat)
        .setText(`Region ${feature.properties.rank}: ${feature.properties.label} of the posterior`)
        .addTo(instance);
    });
    const pin =
      result.status === "localized" && result.latitude !== null && result.longitude !== null
        ? new maplibregl.Marker({ color: "#f0efec" }).setLngLat([result.longitude, result.latitude]).addTo(instance)
        : null;
    return () => {
      pin?.remove();
      if (!instance.getStyle()) return;
      for (const layer of ["aircraft", "regions-line", "regions-fill", "tracks"]) if (instance.getLayer(layer)) instance.removeLayer(layer);
      for (const source of ["aircraft", "regions", "tracks"]) if (instance.getSource(source)) instance.removeSource(source);
    };
  }, [ready, result, selectedFlight]);

  return <div ref={container} className="h-[520px] overflow-hidden rounded-lg border border-surface-line" />;
}
