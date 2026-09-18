"""Piecewise-linear temporal profiles for the DC4 catalog tutorial.

No FITS counts enter these calculations. Parallel normalization reproduces
cosi-sim's sample splitting; the number of simulation chunks is an input,
not something that can be inferred from a spectral .source file.
"""
import numpy as np


def simulation_slices(n_samples, num_bins):
    if num_bins < 1 or num_bins >= n_samples:
        raise ValueError("Require 1 <= simulation bins < light-curve samples")
    size, remainder = divmod(n_samples, num_bins)
    # Each chunk shares its last sample with the next chunk. An exactly
    # divisible sample count needs a shortened final slice, not an overrun.
    slices = [slice(i * size, min((i + 1) * size + 1, n_samples))
              for i in range(num_bins)]
    if remainder > 1:
        slices.append(slice(size * num_bins, n_samples))
    return [s for s in slices if s.stop - s.start > 1]


class LightcurveProfile:
    """Separate endpoint intensities preserve jumps at simulation boundaries."""

    def __init__(self, times, rates, repeating=False, mode="global_megalib", bins=1):
        times, rates = np.asarray(times, float), np.asarray(rates, float)
        if (times.ndim != 1 or times.size < 2 or times.shape != rates.shape
                or not np.all(np.isfinite(times)) or not np.all(np.isfinite(rates))
                or np.any(np.diff(times) <= 0) or np.any(rates < 0)):
            raise ValueError("Invalid light-curve samples")
        if mode not in {"raw", "global_megalib", "parallel_megalib"}:
            raise ValueError(f"Unknown normalization mode: {mode}")
        if repeating and mode == "parallel_megalib":
            raise ValueError("Periodic profiles must not be split into simulation chunks")
        self.origin = times[0]
        self.x = times - self.origin
        self.duration = self.x[-1]
        self.repeating = repeating
        self.left, self.right = rates[:-1].copy(), rates[1:].copy()
        slices = (simulation_slices(len(times), bins) if mode == "parallel_megalib"
                  else [slice(0, len(times))])
        self.active_chunks = 0
        for s in slices:
            area = np.trapezoid(rates[s], self.x[s])
            if area > 0:
                self.active_chunks += 1
            scale = 1. if mode == "raw" else (
                (self.x[s.stop - 1] - self.x[s.start]) / area if area > 0 else 0.)
            self.left[s.start:s.stop - 1] *= scale
            self.right[s.start:s.stop - 1] *= scale
        self.widths = np.diff(self.x)
        self.cumulative = np.r_[0., np.cumsum(.5 * (self.left + self.right) * self.widths)]

    def _primitive(self, values):
        values = np.clip(values, 0., self.duration)
        i = np.clip(np.searchsorted(self.x, values, side="right") - 1,
                    0, len(self.widths) - 1)
        offset = values - self.x[i]
        slope = (self.right[i] - self.left[i]) / self.widths[i]
        return self.cumulative[i] + self.left[i] * offset + .5 * slope * offset**2

    def integral(self, starts, stops):
        a, b = np.asarray(starts, float) - self.origin, np.asarray(stops, float) - self.origin
        if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)) or np.any(b <= a):
            raise ValueError("Invalid integration intervals")
        if self.repeating:
            ca, pa = np.divmod(a, self.duration)
            cb, pb = np.divmod(b, self.duration)
            result = ((cb - ca) * self.cumulative[-1]
                      + self._primitive(pb) - self._primitive(pa))
        else:
            result = self._primitive(b) - self._primitive(a)
        return np.maximum(result, 0.)

    def response_weights(self, starts, stops, mean_factor):
        """Use with amplitudes multiplied by mean_factor, exactly once.

        Integrate short bursts within orientation bins instead of sampling
        their midpoints. Attitude/livetime remain piecewise constant at the
        available orientation resolution (15 s for this DC4 file).
        """
        if not np.isfinite(mean_factor) or mean_factor <= 0:
            raise ValueError("Response mean factor must be finite and positive")
        starts, stops = np.asarray(starts, float), np.asarray(stops, float)
        return self.integral(starts, stops) / (stops - starts) / mean_factor
