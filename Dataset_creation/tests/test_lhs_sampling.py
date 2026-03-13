"""Tests for Latin Hypercube Sampling."""

import pytest
from src.sampling.lhs import latin_hypercube_samples, generate_valid_cases


class TestLHS:
    """Test LHS generation."""

    def test_correct_number_of_samples(self):
        ranges = {"a": [1, 2, 3], "b": [10, 20]}
        samples = latin_hypercube_samples(ranges, n_samples=5)
        assert len(samples) == 5

    def test_all_values_from_choices(self):
        ranges = {"x": [1.0, 2.0, 3.0]}
        samples = latin_hypercube_samples(ranges, n_samples=10)
        for s in samples:
            assert s["x"] in ranges["x"]

    def test_reproducible_with_seed(self):
        ranges = {"a": [1, 2, 3], "b": [10, 20, 30]}
        s1 = latin_hypercube_samples(ranges, 10, seed=42)
        s2 = latin_hypercube_samples(ranges, 10, seed=42)
        assert s1 == s2

    def test_different_seeds_differ(self):
        ranges = {"a": [1, 2, 3], "b": [10, 20, 30]}
        s1 = latin_hypercube_samples(ranges, 10, seed=42)
        s2 = latin_hypercube_samples(ranges, 10, seed=99)
        assert s1 != s2


class TestGenerateValidCases:
    """Test case generation with geometry validation."""

    def test_all_cases_have_required_keys(self):
        cases = generate_valid_cases(n_samples=5, seed=42)
        required = {"T_set", "cx", "cy", "cz", "radius", "height",
                     "kappa", "Cp", "rho", "volume", "mass", "mol_weight"}
        for c in cases:
            assert required.issubset(c.keys()), f"Missing: {required - set(c.keys())}"

    def test_cx_is_always_zero(self):
        cases = generate_valid_cases(n_samples=10, seed=42)
        for c in cases:
            assert c["cx"] == 0.0

    def test_volume_is_consistent(self):
        import math
        cases = generate_valid_cases(n_samples=5, seed=42)
        for c in cases:
            expected = math.pi * c["radius"] ** 2 * c["height"]
            assert abs(c["volume"] - expected) < 1e-12