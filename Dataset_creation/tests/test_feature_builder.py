"""Tests for feature matrix construction."""

import numpy as np
import pytest
from src.dataset.features import build_feature_matrix
from configs.parameters import N_FEATURES


class TestBuildFeatureMatrix:
    """Test X/Y matrix construction."""

    @pytest.fixture
    def dummy_data(self):
        n_cells = 100
        n_times = 5
        coords = np.random.rand(n_cells, 3).astype(np.float64)
        times = np.linspace(0, 10, n_times)
        T_array = np.random.uniform(300, 1000, (n_times, n_cells))
        cyl = {
            "T_set": 1000.0, "cx": 0.0, "cy": 0.18, "cz": 0.195,
            "radius": 0.05, "height": 0.1,
            "kappa": 80.0, "Cp": 450.0, "rho": 7800.0,
            "brick_heater_kappa": 8.0,
        }
        return coords, times, T_array, cyl

    def test_output_shapes(self, dummy_data):
        coords, times, T_array, cyl = dummy_data
        X, Y = build_feature_matrix(coords, times, T_array, cyl)
        n_cells = coords.shape[0]
        n_times = len(times)
        assert X.shape == (n_cells * n_times, N_FEATURES)
        assert Y.shape == (n_cells * n_times, 1)

    def test_n_features_is_14(self):
        assert N_FEATURES == 14

    def test_output_dtype(self, dummy_data):
        X, Y = build_feature_matrix(*dummy_data)
        assert X.dtype == np.float32
        assert Y.dtype == np.float32

    def test_nan_timesteps_skipped(self, dummy_data):
        coords, times, T_array, cyl = dummy_data
        T_array[2, :] = np.nan  # Corrupt one timestep
        X, Y = build_feature_matrix(coords, times, T_array, cyl)
        expected_rows = coords.shape[0] * (len(times) - 1)
        assert X.shape[0] == expected_rows

    def test_constant_features_are_constant(self, dummy_data):
        coords, times, T_array, cyl = dummy_data
        X, _ = build_feature_matrix(coords, times, T_array, cyl)
        # T_set (column 4) should be constant
        assert np.all(X[:, 4] == np.float32(cyl["T_set"]))

    def test_brick_heater_kappa_column(self, dummy_data):
        coords, times, T_array, cyl = dummy_data
        X, _ = build_feature_matrix(coords, times, T_array, cyl)
        # brick_heater_kappa is the last column (index 13)
        assert np.all(X[:, 13] == np.float32(cyl["brick_heater_kappa"]))

    def test_no_volume_or_mass_columns(self, dummy_data):
        """Verify volume and mass are no longer in features."""
        from configs.parameters import FEATURE_COLUMNS
        assert "volume" not in FEATURE_COLUMNS
        assert "mass" not in FEATURE_COLUMNS