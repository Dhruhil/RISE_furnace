"""Tests for cylinder geometry validation."""

import pytest
from src.geometry.validator import validate_cylinder_geometry
from configs.furnace import FURNACE_BOUNDS


class TestValidateCylinderGeometry:
    """Test cylinder-in-furnace geometry constraints."""

    def test_base_case_is_valid(self):
        params = {
            "cx": 0.0, "cy": 0.18, "cz": 0.195,
            "radius": 0.05, "height": 0.10,
        }
        assert validate_cylinder_geometry(params) is True

    def test_cylinder_too_tall_x(self):
        params = {
            "cx": 0.0, "cy": 0.18, "cz": 0.195,
            "radius": 0.05, "height": 0.20,  # exceeds x_max
        }
        assert validate_cylinder_geometry(params) is False

    def test_cylinder_too_close_to_y_min(self):
        params = {
            "cx": 0.0, "cy": 0.08, "cz": 0.195,  # cy - r = 0.03 < 0.07
            "radius": 0.05, "height": 0.10,
        }
        assert validate_cylinder_geometry(params) is False

    def test_cylinder_too_close_to_y_max(self):
        params = {
            "cx": 0.0, "cy": 0.30, "cz": 0.195,  # cy + r = 0.35 > 0.29
            "radius": 0.05, "height": 0.10,
        }
        assert validate_cylinder_geometry(params) is False

    def test_cylinder_at_z_boundary(self):
        params = {
            "cx": 0.0, "cy": 0.18, "cz": 0.04,  # cz - r = -0.01 < 0.01
            "radius": 0.05, "height": 0.10,
        }
        assert validate_cylinder_geometry(params) is False

    def test_small_cylinder_fits(self):
        params = {
            "cx": 0.0, "cy": 0.18, "cz": 0.195,
            "radius": 0.03, "height": 0.06,
        }
        assert validate_cylinder_geometry(params) is True