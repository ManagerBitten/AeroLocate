from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

GAMMA_DRY_AIR = 1.4
GAS_CONSTANT_DRY_AIR = 287.05
ISA_LAPSE_K_PER_M = 0.0065
TROPOPAUSE_M = 11000.0


def speed_of_sound(temperature_k: ArrayLike) -> NDArray[np.float64]:
    return np.sqrt(GAMMA_DRY_AIR * GAS_CONSTANT_DRY_AIR * np.asarray(temperature_k, dtype=float))


@dataclass(frozen=True)
class Atmosphere:
    surface_temperature_c: float = 15.0
    surface_elevation_m: float = 0.0
    wind_east_mps: float = 0.0
    wind_north_mps: float = 0.0
    temperature_sigma_k: float = 5.0
    wind_sigma_mps: float = 5.0
    temperature_is_assumed: bool = True
    wind_is_assumed: bool = True

    def temperature_k(self, altitude_m: ArrayLike) -> NDArray[np.float64]:
        h = np.minimum(np.asarray(altitude_m, dtype=float), TROPOPAUSE_M)
        return (
            self.surface_temperature_c
            + 273.15
            - ISA_LAPSE_K_PER_M * np.maximum(h - self.surface_elevation_m, 0.0)
        )

    def sound_speed(self, altitude_m: ArrayLike) -> NDArray[np.float64]:
        return speed_of_sound(self.temperature_k(altitude_m))

    def path_sound_speed(self, altitude_a_m: ArrayLike, altitude_b_m: ArrayLike) -> NDArray[np.float64]:
        c_a = self.sound_speed(altitude_a_m)
        c_b = self.sound_speed(altitude_b_m)
        close = np.abs(c_a - c_b) < 1e-9
        ratio = np.log(np.where(close, 1.0, c_a / np.where(close, 1.0, c_b)))
        safe = np.where(close, 1.0, ratio)
        return np.where(close, c_a, (c_a - c_b) / safe)

    def sound_speed_sigma(self, altitude_m: ArrayLike) -> NDArray[np.float64]:
        temperature = self.temperature_k(altitude_m)
        return 0.5 * self.sound_speed(altitude_m) * self.temperature_sigma_k / temperature
