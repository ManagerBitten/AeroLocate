from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from backend.geolocation.search import Posterior

FloatArray = NDArray[np.float64]
CREDIBLE_MASS = 0.95
MIN_REGION_MASS = 0.01
MAX_RASTER_CELLS = 4_000_000


@dataclass(frozen=True)
class CandidateRegion:
    rank: int
    probability: float
    latitude: float
    longitude: float
    x_m: float
    y_m: float
    area_km2: float
    radius_m: float
    semi_major_m: float
    semi_minor_m: float
    orientation_deg: float
    polygons: tuple[tuple[tuple[float, float], ...], ...]


@dataclass(frozen=True)
class RegionSummary:
    regions: tuple[CandidateRegion, ...]
    total_area_km2: float
    start_mean: float
    start_sigma: float
    blur_sigma_m: float


def _rasterize(posterior: Posterior, keep: NDArray[np.bool_]) -> tuple[FloatArray, float, float]:
    cell = posterior.cell_m
    x_low = float(posterior.x[keep].min()) - cell / 2
    y_low = float(posterior.y[keep].min()) - cell / 2
    columns = np.floor((posterior.x[keep] - x_low) / cell).astype(int)
    rows = np.floor((posterior.y[keep] - y_low) / cell).astype(int)
    shape = (int(rows.max()) + 1, int(columns.max()) + 1)
    if shape[0] * shape[1] > MAX_RASTER_CELLS:
        raise ValueError("posterior raster exceeds the configured limit")
    raster = np.zeros(shape)
    weights = np.exp(posterior.log_weight[keep] - float(np.nanmax(posterior.log_weight[keep])))
    np.add.at(raster, (rows, columns), weights)
    return raster, x_low, y_low


def _credible_mask(raster: FloatArray) -> FloatArray:
    order = np.argsort(raster.ravel())[::-1]
    cumulative = np.cumsum(raster.ravel()[order])
    cutoff_index = int(np.searchsorted(cumulative, CREDIBLE_MASS * cumulative[-1]))
    threshold = raster.ravel()[order[min(cutoff_index, len(order) - 1)]]
    return raster >= threshold


def _polygons(mask: NDArray[np.bool_], x_low: float, y_low: float, cell: float, posterior: Posterior) -> tuple:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        points = contour.reshape(-1, 2).astype(float)
        if len(points) < 3:
            points = np.array(
                [
                    points[0] + [-0.5, -0.5],
                    points[0] + [0.5, -0.5],
                    points[0] + [0.5, 0.5],
                    points[0] + [-0.5, 0.5],
                ]
            )
        east = x_low + (points[:, 0] + 0.5) * cell
        north = y_low + (points[:, 1] + 0.5) * cell
        latitude, longitude, _ = posterior.frame.from_enu(np.column_stack([east, north, np.zeros(len(east))]))
        ring = [(float(lon), float(lat)) for lat, lon in zip(latitude, longitude)]
        ring.append(ring[0])
        polygons.append(tuple(ring))
    return tuple(polygons)


def summarize_regions(posterior: Posterior, blur_sigma_m: float) -> RegionSummary:
    finite = np.isfinite(posterior.log_weight)
    if not finite.any():
        return RegionSummary((), 0.0, float("nan"), float("nan"), blur_sigma_m)
    weights = np.exp(posterior.log_weight[finite] - float(posterior.log_weight[finite].max()))
    weights /= weights.sum()
    start_mean = float(np.sum(weights * posterior.start[finite]))
    start_sigma = float(np.sqrt(np.sum(weights * (posterior.start[finite] - start_mean) ** 2)))
    raster, x_low, y_low = _rasterize(posterior, finite)
    cell = posterior.cell_m
    pad = int(np.ceil(3.0 * blur_sigma_m / cell)) + 1
    raster = np.pad(raster, pad)
    x_low -= pad * cell
    y_low -= pad * cell
    if blur_sigma_m > 0:
        raster = ndimage.gaussian_filter(raster, blur_sigma_m / cell, truncate=3.0)
    mask = _credible_mask(raster)
    labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
    total = float(raster.sum())
    grid_y, grid_x = np.mgrid[0 : raster.shape[0], 0 : raster.shape[1]]
    east = x_low + (grid_x + 0.5) * cell
    north = y_low + (grid_y + 0.5) * cell
    found: list[CandidateRegion] = []
    for label in range(1, count + 1):
        member = labels == label
        mass = float(raster[member].sum())
        if mass / total < MIN_REGION_MASS:
            continue
        w = raster[member] / mass
        mean_x = float(np.sum(w * east[member]))
        mean_y = float(np.sum(w * north[member]))
        dx = east[member] - mean_x
        dy = north[member] - mean_y
        covariance = np.array(
            [[np.sum(w * dx * dx), np.sum(w * dx * dy)], [np.sum(w * dx * dy), np.sum(w * dy * dy)]]
        ) + np.eye(2) * cell**2 / 12.0
        values, vectors = np.linalg.eigh(covariance)
        semi_major, semi_minor = 2.45 * np.sqrt(values[1]), 2.45 * np.sqrt(values[0])
        orientation = float(np.degrees(np.arctan2(vectors[0, 1], vectors[1, 1])) % 180.0)
        area = float(member.sum()) * cell * cell / 1e6
        latitude, longitude, _ = posterior.frame.from_enu(np.array([mean_x, mean_y, 0.0]))
        found.append(
            CandidateRegion(
                rank=0,
                probability=mass / total,
                latitude=float(latitude),
                longitude=float(longitude),
                x_m=mean_x,
                y_m=mean_y,
                area_km2=area,
                radius_m=float(np.sqrt(area * 1e6 / np.pi)),
                semi_major_m=float(semi_major),
                semi_minor_m=float(semi_minor),
                orientation_deg=orientation,
                polygons=_polygons(member, x_low, y_low, cell, posterior),
            )
        )
    found.sort(key=lambda item: item.probability, reverse=True)
    ranked = tuple(replace(region, rank=index + 1) for index, region in enumerate(found))
    return RegionSummary(
        regions=ranked,
        total_area_km2=float(sum(region.area_km2 for region in ranked)),
        start_mean=start_mean,
        start_sigma=start_sigma,
        blur_sigma_m=blur_sigma_m,
    )
