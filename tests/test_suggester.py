import unittest

import numpy as np
import pandas as pd

from engine.suggester import (
    _smooth_derivative_and_noise,
    analyze_data_linearity,
    scout_hyperparameters,
)


class DataScoutTests(unittest.TestCase):
    @staticmethod
    def _oscillator_frame(phase=0.0, noise=0.0):
        t = np.linspace(0.0, 12.0, 500)
        rng = np.random.default_rng(123)
        return pd.DataFrame({
            "t": t,
            "x": np.cos(t + phase) + noise * rng.normal(size=len(t)),
            "v": -np.sin(t + phase) + noise * rng.normal(size=len(t)),
        })

    def test_multi_trajectory_scout_uses_blocked_validation(self):
        report = scout_hyperparameters(
            [self._oscillator_frame(0.0), self._oscillator_frame(0.8)],
            libraries=("Polynomial",),
            thresholds=(0.01, 0.05, 0.20),
        )

        self.assertEqual(report["candidate_count"], 9)
        self.assertEqual(report["profiles"]["balanced"]["degree"], 1)
        self.assertEqual(report["profiles"]["simplest"]["active_terms"], 2)
        self.assertGreater(report["profiles"]["balanced"]["val_r2"], 0.99)

    def test_noise_estimate_is_robust_and_dimensionless(self):
        _, _, clean_noise = _smooth_derivative_and_noise(
            self._oscillator_frame(noise=0.0))
        _, _, noisy_noise = _smooth_derivative_and_noise(
            self._oscillator_frame(noise=0.20))

        self.assertLess(clean_noise, 0.01)
        self.assertGreater(noisy_noise, clean_noise * 100)

    def test_legacy_tuple_api_fails_safely_for_short_data(self):
        tiny = pd.DataFrame({"t": [0.0, 1.0], "x": [1.0, 0.0]})
        library, degree, threshold, reason = analyze_data_linearity(tiny)

        self.assertEqual((library, degree, threshold), ("Polynomial", 1, 0.10))
        self.assertTrue(reason.startswith("Error analyzing data:"))


if __name__ == "__main__":
    unittest.main()
