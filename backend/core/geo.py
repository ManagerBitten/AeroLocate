from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
FloatArray = NDArray[np.float64]


def geodetic_to_ecef(lat_deg: ArrayLike, lon_deg: ArrayLike, alt_m: ArrayLike) -> FloatArray:
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    alt = np.asarray(alt_m, dtype=float)
    sin_lat = np.sin(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    x = (n + alt) * np.cos(lat) * np.cos(lon)
    y = (n + alt) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + alt) * sin_lat
    return np.stack([x, y, z], axis=-1)


def ecef_to_geodetic(xyz: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
    p = np.asarray(xyz, dtype=float)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    lon = np.arctan2(y, x)
    rho = np.hypot(x, y)
    lat = np.arctan2(z, rho * (1.0 - WGS84_E2))
    alt = np.zeros_like(lat)
    for _ in range(6):
        sin_lat = np.sin(lat)
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
        cos_lat = np.cos(lat)
        safe_cos = np.where(np.abs(cos_lat) < 1e-12, 1e-12, cos_lat)
        alt = rho / safe_cos - n
        lat = np.arctan2(z, rho * (1.0 - WGS84_E2 * n / (n + alt)))
    return np.degrees(lat), np.degrees(lon), alt


@dataclass(frozen=True)
class LocalFrame:
    lat0: float
    lon0: float
    alt0: float = 0.0

    def _rotation(self) -> FloatArray:
        lat = np.radians(self.lat0)
        lon = np.radians(self.lon0)
        return np.array(
            [
                [-np.sin(lon), np.cos(lon), 0.0],
                [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
                [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
            ]
        )

    def to_enu(self, lat_deg: ArrayLike, lon_deg: ArrayLike, alt_m: ArrayLike) -> FloatArray:
        origin = geodetic_to_ecef(self.lat0, self.lon0, self.alt0)
        delta = geodetic_to_ecef(lat_deg, lon_deg, alt_m) - origin
        return delta @ self._rotation().T

    def from_enu(self, enu: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
        origin = geodetic_to_ecef(self.lat0, self.lon0, self.alt0)
        ecef = np.asarray(enu, dtype=float) @ self._rotation() + origin
        return ecef_to_geodetic(ecef)


def haversine_m(lat1: ArrayLike, lon1: ArrayLike, lat2: ArrayLike, lon2: ArrayLike) -> FloatArray:
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2.0 * 6371008.8 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
