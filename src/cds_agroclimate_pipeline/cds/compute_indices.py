#!/usr/bin/env python3
# cds/compute_indices.py
"""
Aggregate CDS seasonal forecast sub-daily NetCDF files to daily resolution
and compute the full suite of agro-climate indices for each ensemble member.

Author: Jemal Ahmed
Email: J.Ahmed@cgiar.org

Overview
--------
Reads raw + derived NetCDF files produced by
cds.download_surface, applies unit conversions,
aggregates 6-hourly data to daily, and calls
agroclimate_indices.compute_agroclimate_indices() for every ensemble member.

Per-member scalar index maps and ensemble statistics (mean, std, p10, p50,
p90) are saved as compressed NetCDF files under {model_dir}/indices/.

Daily resolution — all variables at 24-hour cadence
----------------------------------------------------
All 9 CDS variables are downloaded at step=24 h (215 daily steps).
This ensures full consistency, 4× smaller files than 6-hourly, and no
sub-daily aggregation overhead.

- accumulated (tp, ssrd, e): cumulative from forecast start.
  Daily totals: day_n = value[24n h] − value[24(n-1) h].
- daily_value (t2m, d2m, u10, v10): snapshot at each 24-h mark.
  Used directly as the daily value (end-of-day, ~00:00 UTC).
- daily_extreme (mx2t24, mn2t24): rolling 24-h max/min.
  Extracted directly at 24h, 48h, … steps.

Example
-------
python -m cds_agroclimate_pipeline.cds.compute_indices \\
    --year 2026 --month 5 --day 1 \\
    --models ecmwf ukmo \\
    --country ethiopia \\
    --coarsen 5

    --coarsen 5 reduces the 0.05° grid to 0.25° (native seasonal resolution).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import xarray as xr

from cds_agroclimate_pipeline.paths import DEFAULT_COUNTRY, PROJECT_ROOT, cds_root

# earthkit suite — ECMWF-standard tools integrated throughout this script:
#   earthkit.data       – field-level CDS/NetCDF reader; metadata validation
#   earthkit.transforms – temporal aggregation (6h→daily, daily→monthly)
import earthkit.data as ekd
import earthkit.transforms
from earthkit.transforms import temporal as ekt_temporal

# -------------------------------------------------------------------------
# Locate agroclimate_indices library
# -------------------------------------------------------------------------

_INDICES_DIR = PROJECT_ROOT / "agroclimate_indices"
if str(_INDICES_DIR) not in sys.path:
    sys.path.insert(0, str(_INDICES_DIR))

from agroclimate_indices import AgroClimateConfig, compute_agroclimate_indices  # noqa: E402


# -------------------------------------------------------------------------
# 1. Dataset and model definitions (mirrors download script)
# -------------------------------------------------------------------------

DATASET = "seasonal-original-single-levels"

MODELS: List[Dict[str, str]] = [
    {"originating_centre": "ukmo",         "system": "610"},
    {"originating_centre": "ecmwf",        "system": "51"},
    {"originating_centre": "meteo_france", "system": "9"},
    {"originating_centre": "dwd",          "system": "22"},
    {"originating_centre": "cmcc",         "system": "4"},
    {"originating_centre": "ncep",         "system": "2"},
    {"originating_centre": "jma",          "system": "4"},
    {"originating_centre": "eccc",         "system": "5"},
    {"originating_centre": "bom",          "system": "2"},
]


# -------------------------------------------------------------------------
# 2. Variable configuration — standardized units
# -------------------------------------------------------------------------

class VarSpec(NamedTuple):
    """Full specification for one CDS variable including unit pipeline."""
    agg_type:    str    # "accumulated" | "instantaneous" | "daily_extreme"
    scale:       float  # multiply raw value by this (applied before offset)
    offset:      float  # add this after scaling  (e.g. -273.15 for K → °C)
    raw_unit:    str    # CDS native unit
    unit:        str    # unit after scale+offset — what agroclimate_indices receives
    weather_key: str    # key in the weather dict; leading _ = internal only
    valid_range: Tuple[float, float]  # plausible daily value range for QC logging


# CDS file stem → VarSpec
#
# All variables are downloaded at 24-hour cadence (daily resolution).
# This ensures consistency, halves memory vs 6-hourly, and avoids
# any sub-daily aggregation step in the compute pipeline.
#
# aggregation types
#   accumulated  – running total from forecast start (tp, ssrd, e)
#                  → daily totals via prepend-zero + diff
#   daily_value  – snapshot at each 24-h step (t2m, d2m, u10, v10)
#                  → select values at 24 h, 48 h, 72 h, … (pass-through)
#   daily_extreme – rolling 24-h max/min embedded in field (mx2t24, mn2t24)
#                  → select values at 24 h, 48 h, 72 h, … (pass-through)
#
# unit conversions:
#   tp   : m → mm day⁻¹               × 1000
#   ssrd : J m⁻² → MJ m⁻² day⁻¹      × 1e-6
#   e    : m (negative=evap) → mm/day  × −1000
#   t2m/d2m/mx2t24/mn2t24 : K → °C    × 1, offset −273.15
#   u10/v10 : m s⁻¹                    no change
CDS_RAW_VARIABLE_CONFIG: Dict[str, VarSpec] = {
    "total_precipitation": VarSpec(
        agg_type="accumulated",  scale=1000.0,   offset=0.0,
        raw_unit="m",            unit="mm day⁻¹",
        weather_key="precipitation",
        valid_range=(0.0, 300.0),
    ),
    "surface_solar_radiation_downwards": VarSpec(
        agg_type="accumulated",  scale=1e-6,     offset=0.0,
        raw_unit="J m⁻²",        unit="MJ m⁻² day⁻¹",
        weather_key="shortwave_radiation",
        valid_range=(0.0, 40.0),
    ),
    "evaporation": VarSpec(
        agg_type="accumulated",  scale=-1000.0,  offset=0.0,
        raw_unit="m",            unit="mm day⁻¹",
        weather_key="evaporation",
        valid_range=(0.0, 20.0),
    ),
    "2m_temperature": VarSpec(
        agg_type="daily_value",  scale=1.0,      offset=-273.15,
        raw_unit="K",            unit="°C",
        weather_key="temperature_2m_mean",
        valid_range=(-40.0, 60.0),
    ),
    "2m_dewpoint_temperature": VarSpec(
        agg_type="daily_value",  scale=1.0,      offset=-273.15,
        raw_unit="K",            unit="°C",
        weather_key="_dew_point_degc",
        valid_range=(-50.0, 45.0),
    ),
    "10m_u_component_of_wind": VarSpec(
        agg_type="daily_value",  scale=1.0,      offset=0.0,
        raw_unit="m s⁻¹",        unit="m s⁻¹",
        weather_key="_u10",
        valid_range=(-60.0, 60.0),
    ),
    "10m_v_component_of_wind": VarSpec(
        agg_type="daily_value",  scale=1.0,      offset=0.0,
        raw_unit="m s⁻¹",        unit="m s⁻¹",
        weather_key="_v10",
        valid_range=(-60.0, 60.0),
    ),
    "maximum_2m_temperature_in_the_last_24_hours": VarSpec(
        agg_type="daily_extreme", scale=1.0,     offset=-273.15,
        raw_unit="K",             unit="°C",
        weather_key="temperature_2m_max",
        valid_range=(-30.0, 65.0),
    ),
    "minimum_2m_temperature_in_the_last_24_hours": VarSpec(
        agg_type="daily_extreme", scale=1.0,     offset=-273.15,
        raw_unit="K",             unit="°C",
        weather_key="temperature_2m_min",
        valid_range=(-50.0, 50.0),
    ),
}

# Derived variables (computed by download script from raw fields)
# stem → (weather_key, unit, valid_range)
class DerivedSpec(NamedTuple):
    weather_key: str
    unit:        str
    valid_range: Tuple[float, float]

CDS_DERIVED_VARIABLE_CONFIG: Dict[str, DerivedSpec] = {
    "relative_humidity_2m": DerivedSpec(
        weather_key="relative_humidity_2m",
        unit="%",
        valid_range=(0.0, 100.0),
    ),
    "wind_speed_10m": DerivedSpec(
        weather_key="wind_speed_10m",
        unit="m s⁻¹",
        valid_range=(0.0, 70.0),
    ),
}

# Canonical units for every weather key passed to compute_agroclimate_indices
WEATHER_UNITS: Dict[str, str] = {
    "precipitation":        "mm day⁻¹",
    "shortwave_radiation":  "MJ m⁻² day⁻¹",
    "evaporation":          "mm day⁻¹",
    "temperature_2m_mean":  "°C",
    "temperature_2m_max":   "°C",
    "temperature_2m_min":   "°C",
    "relative_humidity_2m": "%",
    "wind_speed_10m":       "m s⁻¹",
    # internal (not passed to index engine)
    "_dew_point_degc":      "°C",
    "_u10":                 "m s⁻¹",
    "_v10":                 "m s⁻¹",
}

# Output index units (used for NetCDF attributes)
INDEX_UNITS: Dict[str, str] = {
    "rainfall_total":               "mm",
    "growing_degree_days":          "degree-days (°C)",
    "length_of_growing_period_days":"days",
    "et0_total":                    "mm",
    "water_stress_index_mean":      "dimensionless [0–1]",
    "dry_spell_max_days":           "days",
    "wet_spell_max_days":           "days",
    "onset_day_of_forecast":        "day number from forecast start",
    "cessation_day_of_forecast":    "day number from forecast start",
    "false_start_fraction":         "dimensionless [0–1]",
    "thi_max":                      "THI (dimensionless)",
    "thi_mean":                     "THI (dimensionless)",
    "thi_heat_stress_days":         "days",
    "pasture_drought_score":        "dimensionless [0–1]",
    "vector_suitability_score":     "dimensionless [0–1]",
    "surface_water_stress_score":   "dimensionless [0–1]",
    "advisory_score":               "dimensionless [0–1]",
    "agro_pastoral_drought_score":  "dimensionless [0–1]",
    "feed_water_stress_score":      "dimensionless [0–1]",
    "crop_livelihood_stress_score": "dimensionless [0–1]",
    "resilience_score":             "dimensionless [0–1]",
}


# -------------------------------------------------------------------------
# 3. Directory helpers
# -------------------------------------------------------------------------

def model_folder_name(model: Dict[str, str]) -> str:
    return f"{model['originating_centre']}_system{model['system']}"


def build_model_dir(root: Path, model: Dict[str, str], year: int, month: int, day: int) -> Path:
    return (
        root
        / DATASET
        / f"{year:04d}"
        / f"{month:02d}"
        / f"{day:02d}"
        / model_folder_name(model)
    )


# -------------------------------------------------------------------------
# 4. CDS step-coordinate utilities
# -------------------------------------------------------------------------

def extract_step_hours(ds: xr.Dataset) -> np.ndarray:
    """
    Return lead-time values as a float64 array of hours.

    Handles the three naming conventions seen in CDS NetCDF output:
    - 'forecast_period'  timedelta64  (seasonal-original-single-levels)
    - 'step'             timedelta64  (ERA5 / cfgrib convention)
    - 'leadtime_hour'    int/float    (explicitly named in some requests)
    """
    for name in ("forecast_period", "step", "leadtime_hour"):
        if name in ds.coords or name in ds.dims:
            coord = ds[name].values
            if np.issubdtype(coord.dtype, np.timedelta64):
                return coord / np.timedelta64(1, "h")
            return coord.astype(float)

    raise ValueError(
        f"Cannot find a time-step coordinate (forecast_period / step / leadtime_hour). "
        f"Available coords: {list(ds.coords)}"
    )


def extract_member_ids(ds: xr.Dataset) -> np.ndarray:
    """Return ensemble member indices. Scalar (size-1) → [0]."""
    if "number" in ds.dims:
        return ds["number"].values.astype(int)
    return np.array([0], dtype=int)


# -------------------------------------------------------------------------
# 4b. earthkit helpers  (data reader · time coord · deaccumulation)
# -------------------------------------------------------------------------

def _ek_open_to_xarray(nc_path: Path) -> Optional[xr.Dataset]:
    """
    Open a CDS NetCDF file via earthkit.data for GRIB-style field metadata,
    then convert to an xarray Dataset.  Falls back to plain xr.open_dataset.

    earthkit.data gives us GRIB_shortName, paramId, units, grid-shape etc.
    on every field — useful for debugging mismatched downloads.
    """
    if not nc_path.exists() or nc_path.stat().st_size == 0:
        return None
    try:
        fl = ekd.from_source("file", str(nc_path))
        if len(fl) > 0:
            m = fl[0].metadata()
            logging.debug(
                "    ekd  %-44s  shortName=%-6s  paramId=%-5s  "
                "units=%-12s  Nx=%-4s Ny=%-4s  nfields=%d",
                nc_path.name,
                m.get("GRIB_shortName", "?"), m.get("GRIB_paramId", "?"),
                m.get("units", "?"),
                m.get("GRIB_Nx", "?"),  m.get("GRIB_Ny", "?"),
                len(fl),
            )
        return fl.to_xarray()
    except Exception as exc:
        logging.debug("earthkit-data open failed (%s); using xarray: %s", exc, nc_path.name)
        return _open_nc(nc_path)


def _add_time_coord(
    da: xr.DataArray,
    step_dim: str,
    init_date: str,
) -> xr.DataArray:
    """
    Replace the CDS timedelta step coordinate (forecast_period / step /
    leadtime_hour) with absolute datetime64 values.

    Required before earthkit.transforms.temporal functions, which expect
    a datetime64 'time' dimension.  Returns *da* unchanged if init_date is
    empty or the step dimension is not present.
    """
    if not init_date or step_dim not in da.dims:
        return da
    fp  = da[step_dim].values
    ref = np.datetime64(init_date, "ns")
    if np.issubdtype(fp.dtype, np.timedelta64):
        valid_times = (ref + fp).astype("datetime64[ns]")
    else:
        # CDS sometimes encodes step as integer hours
        valid_times = np.array(
            [ref + np.timedelta64(int(h), "h") for h in fp],
            dtype="datetime64[ns]",
        )
    return da.assign_coords({step_dim: valid_times}).rename({step_dim: "time"})


def _deaccumulate_and_sum(
    da: xr.DataArray,
    init_date: str,
) -> xr.DataArray:
    """
    Convert a CDS cumulative-from-start field to daily totals.

    Algorithm
    ---------
    1. Prepend a zero slice at t=init_date so day-1 total = first step value.
    2. diff() recovers per-step increments; clip(min=0) enforces non-negative.
    3. Sub-daily steps (6 h): ekt_temporal.daily_sum sums 4 increments→daily.
       Daily steps (24 h): increments already are daily totals, return directly.

    Dimension-order contract: output preserves da.dims exactly.
    The key pitfall is xr.expand_dims() inserting the new axis at position 0;
    we instead insert at the same position "time" occupies in da.
    """
    orig_dims = da.dims                              # e.g. (number, time, lat, lon)
    t0   = np.datetime64(init_date, "ns")
    zero = xr.zeros_like(da.isel(time=0)).assign_coords(time=t0)
    if "time" not in zero.dims:
        # Place "time" at the same axis position it occupies in da
        time_axis = list(da.dims).index("time")
        zero = zero.expand_dims("time", axis=time_axis)
    increments = xr.concat([zero, da], dim="time").diff("time").clip(min=0)
    step_ns  = float(da["time"].values[1] - da["time"].values[0]) if da.sizes["time"] > 1 else 86_400e9
    step_hrs = step_ns / 3_600_000_000_000.0
    result = ekt_temporal.daily_sum(increments) if step_hrs < 24.0 else increments
    # Restore original dim order — resample/diff/concat can silently reorder
    return result.transpose(*orig_dims)


# -------------------------------------------------------------------------
# 5. Daily aggregation (pure numpy — kept for daily_extreme and fallbacks)
# -------------------------------------------------------------------------

def aggregate_accumulated_daily(
    arr: np.ndarray,
    step_hours: np.ndarray,
) -> np.ndarray:
    """
    Convert field accumulated from forecast start to daily totals.

    Selects values at multiples of 24h, then takes differences:
        day_1 = arr[24h]
        day_n = arr[24n h] - arr[24(n-1) h]

    Input:  arr  shape (step,) or (member, step, lat, lon)
    Output: shape (n_days,) or (member, n_days, lat, lon)
    """
    daily_mask = (step_hours % 24 == 0) & (step_hours > 0)
    n_days = int(daily_mask.sum())

    if arr.ndim == 1:
        vals = arr[daily_mask]
        return np.concatenate([[vals[0]], np.diff(vals)])

    # arr: (member, step, lat, lon)
    vals = arr[:, daily_mask, :, :]          # (member, n_days, lat, lon)
    zeros = np.zeros((arr.shape[0], 1, arr.shape[2], arr.shape[3]), dtype=arr.dtype)
    with_zero = np.concatenate([zeros, vals], axis=1)
    daily = np.diff(with_zero, axis=1)
    return np.maximum(daily, 0.0)


def aggregate_instantaneous_daily(
    arr: np.ndarray,
    step_hours: np.ndarray,
) -> np.ndarray:
    """
    Compute daily mean from 6-hourly instantaneous values.

    Groups all 6h steps whose ceil(step/24) equals day index d:
        steps  6h, 12h, 18h, 24h  → day 1
        steps 30h, 36h, 42h, 48h  → day 2
        ...

    Input:  arr  shape (step,) or (member, step, lat, lon)
    Output: shape (n_days,) or (member, n_days, lat, lon)
    """
    day_index = np.ceil(step_hours / 24.0).astype(int)  # 1-based
    n_days = int(day_index.max())

    if arr.ndim == 1:
        daily = np.full(n_days, np.nan, dtype=float)
        for d in range(1, n_days + 1):
            mask = day_index == d
            if mask.any():
                daily[d - 1] = np.nanmean(arr[mask])
        return daily

    # arr: (member, step, lat, lon)
    daily = np.full(
        (arr.shape[0], n_days, arr.shape[2], arr.shape[3]),
        np.nan,
        dtype=arr.dtype,
    )
    for d in range(1, n_days + 1):
        mask = day_index == d
        if mask.any():
            daily[:, d - 1, :, :] = np.nanmean(arr[:, mask, :, :], axis=1)
    return daily


def aggregate_daily_extreme(
    arr: np.ndarray,
    step_hours: np.ndarray,
) -> np.ndarray:
    """
    Extract rolling 24h extreme fields at 24h step boundaries.

    mx2t24 and mn2t24 are already 24h rolling extremes; selecting values
    at steps 24h, 48h, 72h, ... gives the daily max/min for each day.

    Input:  arr  shape (step,) or (member, step, lat, lon)
    Output: shape (n_days,) or (member, n_days, lat, lon)
    """
    daily_mask = (step_hours % 24 == 0) & (step_hours > 0)
    if arr.ndim == 1:
        return arr[daily_mask]
    return arr[:, daily_mask, :, :]


# -------------------------------------------------------------------------
# 6. Bulk loader — all members in one disk read per variable
# -------------------------------------------------------------------------
# Key insight from benchmarking:
#   • da.isel(number=m).values  = 13.4 s / 306 MB  → 51× = 683 s per variable
#   • da.values (all members)   = 71.5 s / 15.6 GB → read ONCE, aggregate to
#     3.9 GB daily array, keep in memory, free raw array.
# With 9 variables loaded sequentially: ~12 min data load, <1 min compute.
# -------------------------------------------------------------------------

def _open_nc(nc_path: Path) -> Optional[xr.Dataset]:
    """Open a NetCDF4 file lazily; return None on failure."""
    if not nc_path.exists() or nc_path.stat().st_size == 0:
        return None
    try:
        return xr.open_dataset(nc_path, engine="netcdf4")
    except Exception:
        try:
            return xr.open_dataset(nc_path)
        except Exception as exc:
            logging.warning("Cannot open %s: %s", nc_path, exc)
            return None


def _probe_grid(model_dir: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """
    Read lat/lon arrays and member count from the first available raw file.
    Does NOT load any data — only inspects metadata.
    Returns (latitudes, longitudes, n_members).
    """
    for stem in CDS_RAW_VARIABLE_CONFIG:
        nc_path = model_dir / f"{stem}.nc"
        ds = _open_nc(nc_path)
        if ds is None:
            continue
        for dim in ("forecast_reference_time",):
            if dim in ds.dims and ds.sizes[dim] == 1:
                ds = ds.squeeze(dim, drop=True)
        lats = ds["latitude"].values.astype(np.float64)
        lons = ds["longitude"].values.astype(np.float64)
        n_members = int(ds.sizes.get("number", 1))
        ds.close()
        return lats, lons, n_members
    return None, None, 0


def load_variable_for_member(
    nc_path: Path,
    agg_type: str,
    scale: float,
    offset: float,
    member_idx: int,
    coarsen_factor: int,
) -> Optional[np.ndarray]:
    """
    Load and aggregate ONE member's data from a single CDS variable file.

    Peak memory per call: ~300 MB (one member × 860 steps × 321×277 float32).
    This is the core primitive used by build_weather_for_member_from_disk.

    Returns daily array of shape (n_days, lat, lon), or None.
    """
    ds = _open_nc(nc_path)
    if ds is None:
        return None

    for dim in ("forecast_reference_time",):
        if dim in ds.dims and ds.sizes[dim] == 1:
            ds = ds.squeeze(dim, drop=True)

    step_hours = extract_step_hours(ds)

    data_vars = [v for v in ds.data_vars]
    if not data_vars:
        ds.close()
        return None
    var_name = data_vars[0]
    da = ds[var_name]

    step_dim = next(
        (d for d in ("forecast_period", "step", "leadtime_hour") if d in da.dims), None
    )
    if step_dim is None:
        ds.close()
        return None

    # Select this member only before loading data
    if "number" in da.dims:
        da = da.isel(number=member_idx)       # (step, lat, lon)
    # else: no member dim — treat as single member

    if coarsen_factor > 1:
        da = da.coarsen(
            latitude=coarsen_factor, longitude=coarsen_factor, boundary="trim"
        ).mean()

    arr = da.values.astype(np.float32)        # (step, lat, lon)
    ds.close()

    if scale != 1.0:
        arr *= float(scale)
    if offset != 0.0:
        arr += float(offset)

    # Wrap in fake member dim so aggregation helpers work unchanged
    arr_m = arr[np.newaxis, ...]              # (1, step, lat, lon)
    if agg_type == "accumulated":
        daily = aggregate_accumulated_daily(arr_m, step_hours)
    elif agg_type == "instantaneous":
        daily = aggregate_instantaneous_daily(arr_m, step_hours)
    elif agg_type == "daily_extreme":
        daily = aggregate_daily_extreme(arr_m, step_hours)
    else:
        raise ValueError(f"Unknown aggregation type: {agg_type}")

    return daily[0].astype(np.float32)        # (n_days, lat, lon)


# -------------------------------------------------------------------------
# 7. Bulk load all variables for a model (one disk read per file)
# -------------------------------------------------------------------------

def load_all_members_daily(
    model_dir: Path,
    coarsen_factor: int,
    init_date: str = "",          # "YYYY-MM-DD" — used to build datetime time coordinate
) -> Optional[Dict[str, Any]]:
    """
    Load every CDS variable for ALL ensemble members in a single disk read
    per file, immediately aggregate to daily, free the raw array, then move
    to the next variable.

    Peak RAM ≈ one raw variable (≤15.6 GB) + one daily variable (≤3.9 GB)
            = ≤19.5 GB at any time.  Total stored ≈ 9 × 3.9 = 35 GB.

    Returns dict with keys:
        variables   {weather_key: (n_members, n_days, lat, lon) float32}
        latitudes   (lat,) float64
        longitudes  (lon,) float64
        n_members   int
        n_days      int
    """
    variables: Dict[str, np.ndarray] = {}
    lats = lons = None

    for stem, spec in CDS_RAW_VARIABLE_CONFIG.items():
        nc_path = model_dir / f"{stem}.nc"
        # ── earthkit-data open (GRIB metadata logged at DEBUG) ────────
        ds = _ek_open_to_xarray(nc_path)
        if ds is None:
            continue

        for dim in ("forecast_reference_time",):
            if dim in ds.dims and ds.sizes[dim] == 1:
                ds = ds.squeeze(dim, drop=True)

        step_hours = extract_step_hours(ds)          # kept for daily_extreme / fallback
        var_name   = list(ds.data_vars)[0]
        da         = ds[var_name]

        step_dim = next(
            (d for d in ("forecast_period", "step", "leadtime_hour") if d in da.dims), None
        )
        if step_dim is None:
            ds.close(); continue

        if "number" not in da.dims:
            da = da.expand_dims("number")
        da = da.transpose("number", step_dim, "latitude", "longitude")

        if coarsen_factor > 1:
            da = da.coarsen(
                latitude=coarsen_factor, longitude=coarsen_factor, boundary="trim"
            ).mean()

        if lats is None:
            lats = da["latitude"].values.astype(np.float64)
            lons = da["longitude"].values.astype(np.float64)

        # ── Extract / aggregate to daily resolution ───────────────────
        # All vars are now downloaded at 24-hour cadence (step=24 in CDS request).
        #
        # daily_value  (t2m, d2m, u10, v10) — snapshot at each 24-h mark
        # daily_extreme (mx2t24, mn2t24)     — rolling 24-h max/min at 24-h mark
        #   Both: select values at steps 24, 48, 72, … → 215-element array.
        #
        # accumulated  (tp, ssrd, evap)      — cumulative from forecast start
        #   Prepend zero at t=init_date, diff → 215 daily totals.
        #   Uses earthkit.transforms.temporal.daily_sum for sub-daily fallback.
        logging.info("  Loading %-48s …", stem)

        if spec.agg_type == "accumulated":
            da_t     = _add_time_coord(da, step_dim, init_date)
            has_time = "time" in da_t.dims
            if has_time and init_date:
                daily = _deaccumulate_and_sum(da_t, init_date).values.astype(np.float32)
            else:
                arr   = da.values.astype(np.float32)
                daily = aggregate_accumulated_daily(arr, step_hours)
                del arr

        else:  # daily_value OR daily_extreme — direct slice at 24-h marks
            raw   = da.values.astype(np.float32)
            daily = aggregate_daily_extreme(raw, step_hours)
            del raw

        ds.close(); gc.collect()

        if spec.scale != 1.0: daily *= float(spec.scale)
        if spec.offset != 0.0: daily += float(spec.offset)

        variables[spec.weather_key] = daily.astype(np.float32)
        logging.info("  → daily shape %s  [%s]", daily.shape, spec.unit)

    # Derived files (RH, VPD, wind_speed) written by download script
    # Open via earthkit-data → aggregate via earthkit-transforms.temporal
    derived_dir = model_dir / "derived"
    for stem, dspec in CDS_DERIVED_VARIABLE_CONFIG.items():
        nc_path = derived_dir / f"{stem}.nc"
        ds = _ek_open_to_xarray(nc_path)       # earthkit-data with xarray fallback
        if ds is None:
            logging.debug("Derived file absent, will compute from raw: %s", stem)
            continue
        for dim in ("forecast_reference_time",):
            if dim in ds.dims and ds.sizes[dim] == 1:
                ds = ds.squeeze(dim, drop=True)
        step_hours = extract_step_hours(ds)
        var_name   = list(ds.data_vars)[0]
        da         = ds[var_name]
        step_dim   = next(
            (d for d in ("forecast_period", "step", "leadtime_hour") if d in da.dims), None
        )
        if step_dim is None:
            ds.close(); continue
        if "number" not in da.dims:
            da = da.expand_dims("number")
        da  = da.transpose("number", step_dim, "latitude", "longitude")
        if coarsen_factor > 1:
            da = da.coarsen(
                latitude=coarsen_factor, longitude=coarsen_factor, boundary="trim"
            ).mean()
        # Derived vars (RH, VPD, wind_speed) were derived at 24-h cadence
        # by the download script — direct slice at 24-h marks.
        raw   = da.values.astype(np.float32)
        daily = aggregate_daily_extreme(raw, step_hours)
        del raw
        ds.close(); gc.collect()
        variables[dspec.weather_key] = daily
        logging.info("  → derived %s  shape %s", stem, daily.shape)

    if not variables or lats is None:
        return None

    n_members = max(
        arr.shape[0] for arr in variables.values() if arr.ndim == 4
    )
    n_days = min(arr.shape[1] for arr in variables.values() if arr.ndim == 4)

    # Trim all to consistent n_days
    for key in variables:
        if variables[key].ndim == 4 and variables[key].shape[1] > n_days:
            variables[key] = variables[key][:, :n_days, :, :]

    # ── Build datetime time coordinate ───────────────────────────────────────
    # Day 1 = init_date, day 2 = init_date + 1, …
    # Needed by earthkit-transforms monthly aggregation in save_monthly_weather_summaries.
    time_values: Optional[np.ndarray] = None
    if init_date:
        try:
            ref = np.datetime64(init_date, "ns")
            time_values = np.array(
                [ref + np.timedelta64(d - 1, "D") for d in range(1, n_days + 1)],
                dtype="datetime64[ns]",
            )
        except Exception as exc:
            logging.debug("Could not build time coordinate: %s", exc)

    return {
        "variables": variables,
        "time_values": time_values,
        "latitudes": lats, "longitudes": lons,
        "n_members": n_members, "n_days": n_days,
    }


# -------------------------------------------------------------------------
# 8. QC helper (kept for backward-compat; now called inside builder above)
# -------------------------------------------------------------------------

def build_weather_dict_for_member(
    variables: Dict[str, np.ndarray],
    member_idx: int,
    n_days: int,
) -> Dict[str, np.ndarray]:
    """Legacy helper — slices pre-loaded variables dict for one member."""
    """
    Extract one ensemble member's daily data as a weather dict for
    compute_agroclimate_indices.

    All unit conversions have already been applied during the load phase
    (scale × raw + offset).  This function only slices the member dimension
    and optionally derives composite variables (wind speed, RH from dew point).

    Weather dict units on exit
    --------------------------
    precipitation        mm day⁻¹
    temperature_2m_*     °C
    relative_humidity_2m %
    wind_speed_10m       m s⁻¹
    shortwave_radiation  MJ m⁻² day⁻¹
    evaporation          mm day⁻¹
    """
    weather: Dict[str, np.ndarray] = {}

    for key, arr in variables.items():
        if key.startswith("_"):
            continue                      # internal keys handled below
        data = (
            arr[member_idx, :n_days, :, :] if arr.ndim == 4
            else arr[:n_days, :, :]
        )
        weather[key] = data.astype(np.float64, copy=False)

    # --- Derive wind_speed_10m from u/v if not already present ---
    if "wind_speed_10m" not in weather:
        u = variables.get("_u10")
        v = variables.get("_v10")
        if u is not None and v is not None:
            u_d = u[member_idx, :n_days, :, :].astype(float)
            v_d = v[member_idx, :n_days, :, :].astype(float)
            weather["wind_speed_10m"] = np.sqrt(u_d ** 2 + v_d ** 2)

    # --- Derive relative_humidity_2m from dew point (°C) if not present ---
    if "relative_humidity_2m" not in weather and "_dew_point_degc" in variables:
        t_c = weather.get("temperature_2m_mean")
        d_c = variables["_dew_point_degc"][member_idx, :n_days, :, :].astype(float)
        if t_c is not None:
            # FAO-56 SVP formula (inputs already in °C after load-phase conversion)
            es = 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))
            ea = 0.6108 * np.exp(17.27 * d_c / (d_c + 237.3))
            weather["relative_humidity_2m"] = np.clip(100.0 * ea / es, 0.0, 100.0)

    # --- QC: log out-of-range fractions for each variable (DEBUG level) ---
    all_specs = {s.weather_key: s for s in CDS_RAW_VARIABLE_CONFIG.values()}
    for key, data in weather.items():
        spec = all_specs.get(key)
        if spec is None:
            continue
        lo, hi = spec.valid_range
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            continue
        oob = np.sum((finite < lo) | (finite > hi)) / finite.size
        if oob > 0.01:
            logging.warning(
                "QC  %-26s  %.1f%% pixels outside expected range [%g, %g %s]"
                "  (member %d, min=%.2f, max=%.2f)",
                key, oob * 100, lo, hi, spec.unit,
                member_idx, float(finite.min()), float(finite.max()),
            )

    return weather


# -------------------------------------------------------------------------
# 9. Extract scalar index maps from compute_agroclimate_indices output
# -------------------------------------------------------------------------

def extract_scalar_indices(result: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Pull scalar (lat, lon) arrays from the nested compute_agroclimate_indices
    output dict into a flat dict keyed by the names in CROP/LIVESTOCK/
    INTEGRATED_SCALAR_INDICES.
    """
    out: Dict[str, np.ndarray] = {}
    crop = result["crop"]
    livestock = result["livestock"]
    integrated = result["integrated"]

    # --- Crop ---
    out["rainfall_total"] = _to_f32(crop["rainfall_total"])
    out["growing_degree_days"] = _to_f32(crop["growing_degree_days"])
    out["length_of_growing_period_days"] = _to_f32(crop["length_of_growing_period_days"])
    out["et0_total"] = _to_f32(
        crop["reference_et0"]["total"] if crop["reference_et0"]["total"] is not None else None
    )
    out["water_stress_index_mean"] = _to_f32(
        crop["water_stress_index"]["mean"] if crop["water_stress_index"]["mean"] is not None else None
    )
    out["dry_spell_max_days"] = _to_f32(crop["dry_spell"]["max_length"])
    out["wet_spell_max_days"] = _to_f32(crop["wet_spell"]["max_length"])
    out["onset_day_of_forecast"] = _to_f32(
        crop["onset"]["onset_index"],
        fill_for_missing=-1,
    )
    out["cessation_day_of_forecast"] = _to_f32(
        crop["cessation"]["cessation_index"],
        fill_for_missing=-1,
    )
    out["false_start_fraction"] = _to_f32(
        crop["onset"]["false_start_risk"].astype(float)
        if crop["onset"]["false_start_risk"] is not None
        else None
    )

    # Extreme rainfall days
    extreme = crop.get("extreme_rainfall_days", {})
    for key, val in extreme.items():
        if val is not None:
            out[f"extreme_rain_{key}"] = _to_f32(val)

    # --- Livestock ---
    thi_summary = livestock["temperature_humidity_index"]
    if thi_summary is not None:
        out["thi_max"] = _to_f32(thi_summary["max"])
        out["thi_mean"] = _to_f32(thi_summary["mean"])
        heat_dur = livestock.get("heat_stress_duration")
        if heat_dur is not None and heat_dur.get("available") is not False:
            out["thi_heat_stress_days"] = _to_f32(heat_dur.get("total_days"))
    out["pasture_drought_score"] = _to_f32(livestock["pasture_drought_index"]["score"])
    out["vector_suitability_score"] = _to_f32(livestock["vector_suitability_proxy"]["score"])
    out["surface_water_stress_score"] = _to_f32(livestock["surface_water_stress_proxy"]["score"])

    # --- Integrated ---
    out["advisory_score"] = _to_f32(integrated["seasonal_advisory_class"]["score"])
    out["agro_pastoral_drought_score"] = _to_f32(integrated["agro_pastoral_drought_risk"]["score"])
    out["feed_water_stress_score"] = _to_f32(integrated["feed_water_stress_index"]["score"])
    out["crop_livelihood_stress_score"] = _to_f32(
        integrated["crop_livestock_livelihood_stress_score"]["score"]
    )
    out["resilience_score"] = _to_f32(integrated["integrated_resilience_opportunity"]["score"])

    return out


def _to_f32(value: Any, fill_for_missing: float = np.nan) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    # Replace np.datetime64 or object arrays with NaN
    if arr.dtype.kind in ("M", "O"):
        return None
    result = arr.astype(np.float32)
    if not np.isfinite(fill_for_missing):
        return result
    result = np.where(result < 0, fill_for_missing, result)
    return result


# -------------------------------------------------------------------------
# 10. NetCDF output utilities
# -------------------------------------------------------------------------

def _nc_encoding(var_names, dtype="float32"):
    return {
        name: {
            "zlib": True,
            "complevel": 4,
            "dtype": dtype,
            "_FillValue": np.float32(-9999.0),
        }
        for name in var_names
    }


def save_member_indices(
    index_data: Dict[int, Dict[str, np.ndarray]],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    member_ids: np.ndarray,
    indices_dir: Path,
    init_date: str,
    model_name: str,
) -> None:
    """
    Save per-member scalar index maps and ensemble statistics as NetCDF.

    Outputs
    -------
    {indices_dir}/indices_per_member.nc     – (number, lat, lon) for each index
    {indices_dir}/ensemble_statistics.nc    – mean/std/p10/p50/p90 per index
    """
    indices_dir.mkdir(parents=True, exist_ok=True)

    # Collect all index names
    all_names = sorted({name for member_dict in index_data.values() for name in member_dict})

    # Stack member results: (n_members, lat, lon) per index
    n_members = len(member_ids)
    n_lat = latitudes.size
    n_lon = longitudes.size
    stacked: Dict[str, np.ndarray] = {}

    for name in all_names:
        arrays = []
        for m_id in member_ids:
            arr = index_data[int(m_id)].get(name)
            if arr is not None and arr.shape == (n_lat, n_lon):
                arrays.append(arr)
            else:
                arrays.append(np.full((n_lat, n_lon), np.nan, dtype=np.float32))
        stacked[name] = np.stack(arrays, axis=0)  # (n_members, lat, lon)

    # --- Per-member NetCDF ---
    per_member_ds = xr.Dataset(
        {
            name: xr.DataArray(
                stacked[name],
                dims=["number", "latitude", "longitude"],
                coords={"number": member_ids, "latitude": latitudes, "longitude": longitudes},
                attrs={"units": _index_units(name)},
            )
            for name in all_names
        },
        attrs={
            "title": "CDS seasonal forecast agro-climate indices per ensemble member",
            "source_dataset": DATASET,
            "model": model_name,
            "forecast_initialization_date": init_date,
            "n_members": n_members,
            "input_weather_units": str(WEATHER_UNITS),
        },
    )
    per_member_path = indices_dir / "indices_per_member.nc"
    per_member_ds.to_netcdf(
        per_member_path,
        encoding=_nc_encoding(all_names),
    )
    logging.info("Saved per-member indices: %s", per_member_path)

    # --- Ensemble statistics NetCDF ---
    ens_vars: Dict[str, xr.DataArray] = {}
    coords = {"latitude": latitudes, "longitude": longitudes}

    for name in all_names:
        arr = stacked[name]
        with np.errstate(all="ignore"):
            ens_vars[f"{name}_mean"] = xr.DataArray(
                np.nanmean(arr, axis=0).astype(np.float32),
                dims=["latitude", "longitude"], coords=coords,
                attrs={"units": _index_units(name), "statistic": "ensemble_mean"},
            )
            ens_vars[f"{name}_std"] = xr.DataArray(
                np.nanstd(arr, axis=0).astype(np.float32),
                dims=["latitude", "longitude"], coords=coords,
                attrs={"units": _index_units(name), "statistic": "ensemble_std"},
            )
            for pct in (10, 25, 50, 75, 90):
                ens_vars[f"{name}_p{pct}"] = xr.DataArray(
                    np.nanpercentile(arr, pct, axis=0).astype(np.float32),
                    dims=["latitude", "longitude"], coords=coords,
                    attrs={"units": _index_units(name), "statistic": f"p{pct}"},
                )

    ens_ds = xr.Dataset(
        ens_vars,
        attrs={
            "title": "CDS seasonal forecast agro-climate indices ensemble statistics",
            "source_dataset": DATASET,
            "model": model_name,
            "forecast_initialization_date": init_date,
            "n_members": n_members,
        },
    )
    ens_path = indices_dir / "ensemble_statistics.nc"
    ens_ds.to_netcdf(
        ens_path,
        encoding=_nc_encoding(list(ens_vars.keys())),
    )
    logging.info("Saved ensemble statistics: %s", ens_path)


def save_monthly_weather_summaries(
    variables: Dict[str, np.ndarray],
    time_values: np.ndarray,          # datetime64[ns], shape (n_days,)
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    member_ids: np.ndarray,
    indices_dir: Path,
    init_date: str,
    model_name: str,
) -> None:
    """
    Aggregate daily weather arrays to monthly using earthkit.transforms.temporal
    and save as ``monthly_weather_summaries.nc``.

    Aggregation rules
    -----------------
    Accumulated vars (precipitation, shortwave_radiation, evaporation)
        → monthly_sum  (total energy / water per calendar month)
    Instantaneous vars (temperature, RH, VPD, wind speed)
        → monthly_mean

    Output dimensions: (number, time, latitude, longitude)
    where *time* is the first timestamp of each calendar month.
    """
    # Variables that represent totals and should be summed over each month
    MONTHLY_SUM_KEYS = {"precipitation", "shortwave_radiation", "evaporation"}

    # Public weather keys with their (long_name, monthly_unit)
    VAR_META: Dict[str, Tuple[str, str]] = {
        "precipitation":        ("Total precipitation",          "mm month⁻¹"),
        "shortwave_radiation":  ("Surface solar radiation down", "MJ m⁻² month⁻¹"),
        "evaporation":          ("Evaporation",                  "mm month⁻¹"),
        "temperature_2m_mean":  ("2m temperature (mean)",        "°C"),
        "temperature_2m_max":   ("2m temperature (max)",         "°C"),
        "temperature_2m_min":   ("2m temperature (min)",         "°C"),
        "relative_humidity_2m": ("Relative humidity 2m",         "%"),
        "vapor_pressure_deficit": ("Vapour pressure deficit",    "kPa"),
        "wind_speed_10m":       ("10m wind speed",               "m s⁻¹"),
    }

    monthly_das: Dict[str, xr.DataArray] = {}
    n_members = len(member_ids)

    for key, arr in variables.items():
        if key.startswith("_") or key not in VAR_META:
            continue
        if arr.ndim != 4 or arr.shape[0] != n_members:
            continue

        long_name, unit = VAR_META[key]

        # Wrap in xarray with a proper datetime time axis
        da = xr.DataArray(
            arr.astype(np.float32),
            dims=["number", "time", "latitude", "longitude"],
            coords={
                "number":    member_ids,
                "time":      time_values,
                "latitude":  latitudes,
                "longitude": longitudes,
            },
            attrs={"long_name": long_name, "units": unit},
        )

        try:
            if key in MONTHLY_SUM_KEYS:
                monthly_da = ekt_temporal.monthly_sum(da)
            else:
                monthly_da = ekt_temporal.monthly_mean(da)
            monthly_da.attrs.update({"long_name": long_name, "units": unit})
            monthly_das[key] = monthly_da
        except Exception as exc:
            logging.warning("Monthly aggregation failed for %s: %s", key, exc)

    if not monthly_das:
        logging.warning("No variables aggregated to monthly — skipping monthly output.")
        return

    monthly_ds = xr.Dataset(
        monthly_das,
        attrs={
            "title":                     "Monthly weather summaries from CDS seasonal forecast",
            "source_dataset":            DATASET,
            "model":                     model_name,
            "forecast_initialization_date": init_date,
            "n_members":                 int(n_members),
            "aggregation_tool":          "earthkit.transforms.temporal",
            "earthkit_transforms_version": earthkit.transforms.__version__,
        },
    )

    out_path = indices_dir / "monthly_weather_summaries.nc"
    indices_dir.mkdir(parents=True, exist_ok=True)
    monthly_ds.to_netcdf(
        out_path,
        encoding=_nc_encoding(list(monthly_das.keys())),
    )
    logging.info("Saved monthly weather summaries (%d months): %s",
                 monthly_ds.sizes.get("time", 0), out_path)


def _index_units(name: str) -> str:
    """Return the CF-compliant unit string for a scalar index name."""
    for key, unit in INDEX_UNITS.items():
        if name.startswith(key):
            return unit
    return "1"  # CF convention for dimensionless when unknown


# -------------------------------------------------------------------------
# 11. Main per-model computation driver
# -------------------------------------------------------------------------

def compute_indices_for_model(
    model_dir: Path,
    year: int,
    month: int,
    day: int,
    model: Dict[str, str],
    coarsen_factor: int,
    overwrite: bool,
    cfg_workers: int = 4,
) -> None:
    """
    Load CDS data for one model, compute agro-climate indices for every
    ensemble member, and save outputs to {model_dir}/indices/.
    """
    indices_dir = model_dir / "indices"
    done_flag = indices_dir / "ensemble_statistics.nc"

    if done_flag.exists() and not overwrite:
        logging.info("SKIP (already computed): %s", indices_dir)
        return

    logging.info("Processing model: %s", model_folder_name(model))

    # --- Bulk-load all variables (one disk read per file) ---
    # Each file is read once for all members, aggregated to daily,
    # then the raw array is freed. Peak RAM ≤ 19.5 GB; total stored ≈ 35 GB.
    init_date = f"{year:04d}-{month:02d}-{day:02d}"
    data = load_all_members_daily(model_dir, coarsen_factor, init_date=init_date)
    if data is None:
        logging.warning("No data loaded for %s — skipping.", model_folder_name(model))
        return

    variables   = data["variables"]
    time_values = data.get("time_values")       # datetime64[ns] (n_days,); may be None
    latitudes   = data["latitudes"]
    longitudes  = data["longitudes"]
    n_members   = data["n_members"]
    n_days      = data["n_days"]

    logging.info(
        "All variables loaded: %d members × %d days × %d×%d grid",
        n_members, n_days, latitudes.size, longitudes.size,
    )

    lat_grid_2d = np.tile(latitudes[:, np.newaxis], (1, longitudes.size))
    cfg = AgroClimateConfig(
        latitude=lat_grid_2d, elevation_m=0.0, shortwave_radiation_unit="mj_m2_day"
    )

    # --- Thread pool: slice member from in-memory arrays, run indices ---
    # Each thread gets a (n_days, lat, lon) view — no disk I/O, just RAM slicing.
    index_data: Dict[int, Dict[str, np.ndarray]] = {}
    n_workers = min(cfg_workers, n_members)

    def _member_task(m_id: int) -> Tuple[int, Dict[str, np.ndarray]]:
        weather = build_weather_dict_for_member(variables, m_id, n_days)
        result  = compute_agroclimate_indices(weather, config=cfg)
        return m_id, extract_scalar_indices(result)

    logging.info("Computing %d members with %d threads …", n_members, n_workers)
    done_count = 0
    futures_map: Dict = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for m_id in range(n_members):
            futures_map[pool.submit(_member_task, m_id)] = m_id

        for fut in as_completed(futures_map):
            m_id = futures_map[fut]
            try:
                member_id, scalars = fut.result()
                index_data[member_id] = scalars
                done_count += 1
                logging.info("  ✓ %d/%d  member=%d", done_count, n_members, m_id)
            except Exception as exc:
                logging.exception("  ✗ FAILED member=%d: %s", m_id, exc)

    if not index_data:
        logging.error("No indices computed for model %s.", model_folder_name(model))
        return

    # --- Save outputs ---
    member_ids = np.arange(n_members)           # 0-based member index array
    save_member_indices(
        index_data=index_data,
        latitudes=latitudes,
        longitudes=longitudes,
        member_ids=member_ids,
        indices_dir=indices_dir,
        init_date=init_date,
        model_name=model_folder_name(model),
    )

    # Monthly weather summaries (earthkit-transforms: daily → monthly aggregation)
    if time_values is not None:
        save_monthly_weather_summaries(
            variables=variables,
            time_values=time_values,
            latitudes=latitudes,
            longitudes=longitudes,
            member_ids=member_ids,
            indices_dir=indices_dir,
            init_date=init_date,
            model_name=model_folder_name(model),
        )

    # Save run metadata
    meta = {
        "model": model_folder_name(model),
        "forecast_initialization_date": init_date,
        "n_members": int(n_members),
        "coarsen_factor": coarsen_factor,
        "grid_resolution_deg": round(abs(float(latitudes[1]) - float(latitudes[0])), 4)
        if latitudes.size > 1 else None,
        "index_names": sorted(next(iter(index_data.values())).keys()),
    }
    (indices_dir / "compute_metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    logging.info("DONE: %s", model_folder_name(model))


# -------------------------------------------------------------------------
# 12. Logging setup and CLI
# -------------------------------------------------------------------------

def setup_logging(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    log_file = root / "cds_agroclimate_indices.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        ],
    )


def _model_process_worker(kwargs: Dict[str, Any]) -> str:
    """
    Top-level function for ProcessPoolExecutor workers.
    Must be module-level so it can be pickled by 'spawn' start method.
    Runs compute_indices_for_model for one model and returns its name.
    """
    # Re-insert the library path in each worker process (spawn start method)
    _lib = str(Path(__file__).resolve().parent / "agroclimate_indices")
    if _lib not in sys.path:
        sys.path.insert(0, _lib)
    setup_logging(Path(kwargs["root"]))
    compute_indices_for_model(**{k: v for k, v in kwargs.items() if k != "root"})
    return kwargs["model_name"]


def parse_args() -> argparse.Namespace:
    _default_workers = max(1, os.cpu_count() // 2)
    parser = argparse.ArgumentParser(
        description=(
            "Compute agro-climate indices from CDS seasonal forecast NetCDF files "
            "downloaded by cds_agroclimate_pipeline.cds.download_surface."
        )
    )
    parser.add_argument("--year",  type=int, required=True, help="Forecast init year, e.g. 2026")
    parser.add_argument("--month", type=int, required=True, help="Forecast init month, e.g. 5")
    parser.add_argument("--day",   type=int, default=1,     help="Forecast init day. Default: 1")
    parser.add_argument("--country", type=str, default=DEFAULT_COUNTRY,
                        help="Country slug in config/countries. Default: ethiopia.")
    parser.add_argument(
        "--root",
        default=None,
        help="Root directory (same as used for downloading). Default: data/countries/<country>/seasonal/cds.",
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Model centres to process, e.g. --models ecmwf ukmo. Default: all.",
    )
    parser.add_argument(
        "--coarsen", type=int, default=1,
        help=(
            "Spatial coarsening factor. Default 1 = no coarsening (correct for 0.25° data). "
            "Use 2 to halve resolution (0.25°→0.5°)."
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=_default_workers,
        help=(
            "Total parallel workers. Split evenly between model-level processes "
            f"and per-model member threads. Default: {_default_workers} "
            "(half of logical CPU count)."
        ),
    )
    parser.add_argument(
        "--model-workers", type=int, default=None,
        help=(
            "Number of models to process simultaneously (ProcessPoolExecutor). "
            "Default: 1 if workers≤4, else 2. Set higher only if RAM allows "
            "(each model process loads its own copy of the daily data)."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute and overwrite existing index files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root) if args.root else cds_root(args.country)
    setup_logging(root)

    if args.models:
        selected = [m for m in MODELS if m["originating_centre"] in args.models]
    else:
        selected = MODELS

    if not selected:
        raise ValueError(f"No matching models for filter: {args.models}")

    # ── Worker budget allocation ──────────────────────────────────────
    # model_workers: how many models run simultaneously (process pool)
    # member_workers: threads per model for member parallelism
    total_workers   = args.workers
    model_workers   = args.model_workers or (1 if total_workers <= 4 else 2)
    model_workers   = min(model_workers, len(selected))
    member_workers  = max(1, total_workers // model_workers)

    logging.info("CDS agro-climate index computation")
    logging.info("Date         : %04d-%02d-%02d", args.year, args.month, args.day)
    logging.info("Country      : %s", args.country)
    logging.info("Models       : %s", [model_folder_name(m) for m in selected])
    logging.info("Coarsen      : %d", args.coarsen)
    logging.info("Root         : %s", root)
    logging.info(
        "Workers      : %d total | %d model processes × %d member threads",
        total_workers, model_workers, member_workers,
    )

    # ── Model-level parallelism (ProcessPoolExecutor) ─────────────────
    # Each process is fully independent: loads its own data, saves its own output.
    task_kwargs = []
    for model in selected:
        model_dir = build_model_dir(root, model, args.year, args.month, args.day)
        if not model_dir.exists():
            logging.warning("Model directory not found, skipping: %s", model_dir)
            continue
        task_kwargs.append({
            "model_dir":    model_dir,
            "year":         args.year,
            "month":        args.month,
            "day":          args.day,
            "model":        model,
            "coarsen_factor": args.coarsen,
            "overwrite":    args.overwrite,
            "cfg_workers":  member_workers,
            # Extra keys for the worker wrapper (not passed to compute_indices_for_model)
            "root":         root,
            "model_name":   model_folder_name(model),
        })

    if model_workers == 1 or len(task_kwargs) == 1:
        # Run in-process — avoids spawn overhead for single-model runs
        for kw in task_kwargs:
            compute_indices_for_model(
                model_dir=kw["model_dir"],
                year=kw["year"], month=kw["month"], day=kw["day"],
                model=kw["model"], coarsen_factor=kw["coarsen_factor"],
                overwrite=kw["overwrite"], cfg_workers=kw["cfg_workers"],
            )
    else:
        with ProcessPoolExecutor(max_workers=model_workers) as pool:
            futures = {pool.submit(_model_process_worker, kw): kw["model_name"]
                       for kw in task_kwargs}
            for fut in as_completed(futures):
                mname = futures[fut]
                exc = fut.exception()
                if exc:
                    logging.error("✗ Model %s FAILED: %s", mname, exc)
                else:
                    logging.info("✓ Model %s complete", fut.result())


if __name__ == "__main__":
    main()
