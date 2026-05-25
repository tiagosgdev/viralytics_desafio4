"""Temporal smoothing for webcam body measurements."""

from __future__ import annotations

from collections import deque
from statistics import median


SMOOTHED_KEYS = ("shoulder_width", "hip_width", "shoulder_hip_ratio")


class MeasurementSmoother:
    """Median filter with simple spike rejection."""

    def __init__(self, window_size: int = 5, spike_threshold: float = 0.28) -> None:
        self.window_size = max(1, int(window_size))
        self.spike_threshold = spike_threshold
        self._history: dict[str, deque[float]] = {
            key: deque(maxlen=self.window_size) for key in SMOOTHED_KEYS
        }

    def update(self, measurements: dict[str, float]) -> dict[str, float]:
        """Add a measurement frame and return smoothed values."""
        smoothed = dict(measurements)
        for key in SMOOTHED_KEYS:
            value = float(measurements.get(key, 0.0))
            if value <= 0.0:
                continue

            history = self._history[key]
            if history:
                current_median = median(history)
                if current_median > 0 and abs(value - current_median) / current_median > self.spike_threshold:
                    value = current_median
            history.append(value)
            smoothed[key] = round(float(median(history)), 4)

        return smoothed

    def reset(self) -> None:
        """Clear smoothing state between unrelated sessions if needed."""
        for history in self._history.values():
            history.clear()
