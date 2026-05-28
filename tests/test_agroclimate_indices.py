import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "agroclimate_indices"))

from agroclimate_indices import AgroClimateConfig, compute_agroclimate_indices, summarize_indices


def build_weather(days=120):
    time = np.datetime64("2026-01-01") + np.arange(days).astype("timedelta64[D]")
    base_p = np.zeros(days, dtype=float)
    base_p[10:13] = [8.0, 7.0, 8.0]
    base_p[45:48] = [10.0, 12.0, 5.0]
    base_p[48:95] = 3.0
    base_p[95:105] = 1.5
    scale = np.array([[1.0, 0.8], [1.2, 0.6]])
    p = base_p[:, None, None] * scale[None, :, :]

    trend = np.linspace(0.0, 2.0, days)[:, None, None]
    tmin = 17.0 + trend + np.zeros_like(p)
    tmax = 29.0 + trend + np.zeros_like(p)
    tmean = (tmin + tmax) / 2.0
    rh = np.full_like(p, 0.68)
    wind = np.full_like(p, 3.0)
    rad = np.full_like(p, 21.0)
    soil = np.full_like(p, 0.28)

    return {
        "time": time,
        "precipitation": p,
        "temperature_2m_min": tmin,
        "temperature_2m_max": tmax,
        "temperature_2m_mean": tmean,
        "relative_humidity_2m": rh,
        "wind_speed_10m": wind,
        "shortwave_radiation": rad,
        "soil_moisture": soil,
        "surface_pressure": np.full_like(p, 90.0),
        "elevation": np.array([[1200.0, 1100.0], [1000.0, 900.0]]),
    }


def test_core_indices_on_gridded_daily_data():
    weather = build_weather()
    clim = {
        "precipitation_total": np.array([[170.0, 170.0], [170.0, 170.0]]),
        "precipitation_p33": np.array([[130.0, 130.0], [130.0, 130.0]]),
        "precipitation_p66": np.array([[210.0, 210.0], [210.0, 210.0]]),
        "precipitation_1_mean": np.array([[75.0, 75.0], [75.0, 75.0]]),
        "precipitation_1_std": np.array([[20.0, 20.0], [20.0, 20.0]]),
        "precipitation_3_mean": np.array([[180.0, 180.0], [180.0, 180.0]]),
        "precipitation_3_std": np.array([[30.0, 30.0], [30.0, 30.0]]),
        "water_balance_1_mean": np.array([[30.0, 30.0], [30.0, 30.0]]),
        "water_balance_1_std": np.array([[15.0, 15.0], [15.0, 15.0]]),
        "water_balance_3_mean": np.array([[90.0, 90.0], [90.0, 90.0]]),
        "water_balance_3_std": np.array([[25.0, 25.0], [25.0, 25.0]]),
        "soil_moisture_mean": np.array([[0.25, 0.25], [0.25, 0.25]]),
        "shortwave_radiation_total": np.array([[2400.0, 2400.0], [2400.0, 2400.0]]),
    }
    cfg = AgroClimateConfig(latitude=-13.0, elevation_m=1100.0)

    result = compute_agroclimate_indices(weather, clim, cfg)
    summary = summarize_indices(result)

    assert result["crop"]["rainfall_total"].shape == (2, 2)
    assert result["crop"]["onset"]["first_candidate_index"][0, 0] == 10
    assert bool(result["crop"]["onset"]["false_start_risk"][0, 0])
    assert result["crop"]["onset"]["onset_index"][0, 0] == 45
    assert result["crop"]["cessation"]["cessation_index"][0, 0] >= 94
    assert np.isfinite(result["crop"]["reference_et0"]["total"]).all()
    assert result["crop"]["spi"][1]["available"]
    assert result["crop"]["spei"][1]["available"]
    assert result["livestock"]["temperature_humidity_index"]["max"].shape == (2, 2)
    assert result["integrated"]["seasonal_advisory_class"]["class"].shape == (2, 2)
    assert summary["rainfall_total"] > 0


if __name__ == "__main__":
    test_core_indices_on_gridded_daily_data()
    print("agroclimate index smoke test passed")
