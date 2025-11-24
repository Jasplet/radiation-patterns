import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
import pygc
from unittest import mock
from radiation.patterns import (
    calc_station_dist_az_rad,
    model_ray_param_ak135,
)
from obspy.geodetics import kilometers2degrees


@pytest.mark.parametrize(
    "lat1, lon1, lat2, lon2",
    (
        [10, 0, 20, 10],
        [34.05, -118.25, 36.12, -115.17],
        [-15.78, -47.93, -22.90, -43.20],
    ),
)
def test_calc_station_dist_az_rad(lat1, lon1, lat2, lon2):
    fault = {"latitude": lat2, "longitude": lon2}
    expected = pygc.great_distance(
        end_latitude=lat1,
        end_longitude=lon1,
        start_latitude=fault["latitude"],
        start_longitude=fault["longitude"],
    )
    distance, azimuth_rad = calc_station_dist_az_rad(lat1, lon1, fault)
    assert distance == expected["distance"] * 1e-3
    assert azimuth_rad == np.deg2rad(expected["azimuth"])


class MockArrival:
    def __init__(self, ray_param, name):
        # convert from s/km to s/degree as this is the units TauPyModel returns
        self.ray_param_sec_degree = 1 / kilometers2degrees(1 / ray_param)
        self.ray_param_sec_km = ray_param
        self.name = name


@pytest.mark.parametrize("vp, vs", [(5, 3), (6, 2)])
def test_model_ray_param_ak135_known_values(vp, vs):
    """Test known values for ray parameters and takeoff angles."""

    # Mock TaupyModel to return known values
    with mock.patch("radiation.patterns.TauPyModel") as MockTaupyModel:
        mock_instance = MockTaupyModel.return_value
        mock_arrivals = [
            MockArrival(0.1, "P"),  # P wave
            MockArrival(0.2, "S"),  # S wave
        ]
        mock_instance.get_ray_paths.return_value = mock_arrivals
        lat1, lon1 = 0, 0
        lat2, lon2 = 10, 10
        takeoff_p, takeoff_s = model_ray_param_ak135(
            lat1, lon1, lat2, lon2, vp=vp, vs=vs
        )

        MockTaupyModel.assert_called_once_with(model="ak135")
        mock_instance.get_ray_paths.assert_called_once()
        # expected takeoffs assuming default velciites of 5km/s and 3km/s

        expected_takeoff_p = np.rad2deg(
            np.arcsin(mock_arrivals[0].ray_param_sec_km * vp)
        )  # ~30 degrees
        expected_takeoff_s = np.rad2deg(
            np.arcsin(mock_arrivals[1].ray_param_sec_km * vs)
        )  # ~36.87 degrees

        np.testing.assert_almost_equal(takeoff_p, expected_takeoff_p, decimal=3)
        np.testing.assert_almost_equal(takeoff_s, expected_takeoff_s, decimal=3)


def test_model_ray_param_ak135_different_distances():
    """Test that different distances produce different takeoff angles."""
    lat1, lon1 = 0, 0

    # Short distance
    takeoff_p_short, takeoff_s_short = model_ray_param_ak135(lat1, lon1, 1, 1)

    # Long distance
    takeoff_p_long, takeoff_s_long = model_ray_param_ak135(lat1, lon1, 30, 30)

    # Longer distances should have smaller takeoff angles (more horizontal rays)
    assert takeoff_p_long < takeoff_p_short
    assert takeoff_s_long < takeoff_s_short


def test_model_ray_param_ak135_returns_tuple():
    """Test that the function returns a tuple of two floats."""
    lat1, lon1 = 34.05, -118.25
    lat2, lon2 = 36.12, -115.17

    result = model_ray_param_ak135(lat1, lon1, lat2, lon2)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], (float, np.floating))
    assert isinstance(result[1], (float, np.floating))
