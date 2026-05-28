#!/usr/bin/env python3
# cds/download_surface.py
"""
Download CDS seasonal agro-surface variables for Africa and derive agro-climate variables.

Author: Jemal Ahmed
Email: J.Ahmed@cgiar.org

Purpose
-------
This script downloads selected agro-surface variables from the CDS seasonal-original-single-levels
dataset for multiple seasonal forecast systems and automatically derives:

    1. relative_humidity_2m (%)
    2. vapor_pressure_deficit (kPa)
    3. wind_speed_10m (m s-1)

Raw CDS variables are downloaded as one NetCDF file per variable to support restartable workflows.

Example
-------
python -m cds_agroclimate_pipeline.cds.download_surface \
    --year 2026 \
    --month 5 \
    --day 1 \
    --models ecmwf \
    --country ethiopia
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Dict, List, Set

import cdsapi
import numpy as np
import xarray as xr

from cds_agroclimate_pipeline.paths import DEFAULT_COUNTRY, cds_root, country_cds_area

# earthkit-data: field-level CDS/NetCDF reader — used post-download for
# metadata inspection (shortName, paramId, grid dims, n_fields).
try:
    import earthkit.data as ekd
    _EARTHKIT_DATA = True
except ImportError:
    _EARTHKIT_DATA = False

# earthkit-meteo: ECMWF-standard vectorised meteorological formulas
# All thermo functions expect temperature in Kelvin; SVP returned in Pa.
try:
    import earthkit.meteo.thermo as _ek_thermo
    import earthkit.meteo.wind as _ek_wind
    _EARTHKIT_METEO = True
except ImportError:
    _EARTHKIT_METEO = False
    logging.warning("earthkit.meteo not available — falling back to FAO-56 formulas.")

# Unified flag for backward compat
_EARTHKIT = _EARTHKIT_METEO

# Thread-safe lock for derive step (only one derive per model at a time)
_derive_lock = Lock()


# ---------------------------------------------------------------------
# 1. CDS dataset
# ---------------------------------------------------------------------

DATASET = "seasonal-original-single-levels"


# ---------------------------------------------------------------------
# 2. Seasonal forecast systems
# ---------------------------------------------------------------------

MODELS: List[Dict[str, str]] = [
    {"originating_centre": "ukmo", "system": "610"},
    {"originating_centre": "ecmwf", "system": "51"},
    {"originating_centre": "meteo_france", "system": "9"},
    {"originating_centre": "dwd", "system": "22"},
    {"originating_centre": "cmcc", "system": "4"},
    {"originating_centre": "ncep", "system": "2"},
    {"originating_centre": "jma", "system": "4"},
    {"originating_centre": "eccc", "system": "5"},
    {"originating_centre": "bom", "system": "2"},
]


# ---------------------------------------------------------------------
# 3. Core agro-surface variables to download from CDS
# ---------------------------------------------------------------------

AGRO_SURFACE_CDS_VARIABLES: List[str] = [
    "total_precipitation",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "evaporation",
    "maximum_2m_temperature_in_the_last_24_hours",
    "minimum_2m_temperature_in_the_last_24_hours",
    "surface_solar_radiation_downwards",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind"
]


# ---------------------------------------------------------------------
# 4. Spatial and temporal settings
# ---------------------------------------------------------------------

# CDS order: [North, West, South, East]
# CDS order: [North, West, South, East].
# Country-specific areas are loaded from config/countries/<country>.yaml.

# 0.25° grid — matches native ECMWF seasonal forecast resolution.
# Ethiopia at 0.25° = 49 lat × 61 lon = 2,989 grid cells.
GRID = [0.25, 0.25]


# ---------------------------------------------------------------------
# 5. Download utilities
# ---------------------------------------------------------------------

def generate_leadtime_hours(max_lead_hour: int = 5160, step: int = 24) -> List[str]:
    """
    Generate CDS leadtime_hour values at daily resolution.

    Default (step=24):
        24, 48, 72, ..., 5160  →  215 daily steps

    All variables — instantaneous (t2m, d2m, u10, v10) and accumulated
    (tp, ssrd, e) — are requested at the same 24-hour cadence for:
      • Consistency: one value per calendar day across all fields.
      • Memory:      4× smaller files than 6-hourly (860 → 215 steps).
      • Speed:       load + compute 4× faster per model.

    Note: instantaneous vars at step N represent the field value at
    T_init + N hours (end-of-day snapshot, ~00:00 UTC each day).
    For seasonal forecasting purposes this is equivalent to a daily value.
    """
    return [str(hour) for hour in range(step, max_lead_hour + step, step)]


def build_request(
    model: Dict[str, str],
    variable: str,
    year: int,
    month: int,
    day: int,
    max_lead_hour: int,
    area: List[float],
    grid: List[float],
) -> Dict:
    """
    Build CDS request for one model and one variable.
    """

    return {
        "originating_centre": model["originating_centre"],
        "system": model["system"],
        "variable": [variable],
        "year": [f"{year:04d}"],
        "month": [f"{month:02d}"],
        "day": [f"{day:02d}"],
        "leadtime_hour": generate_leadtime_hours(max_lead_hour=max_lead_hour),
        "data_format": "netcdf",
        "grid": grid,
        "area": area,
    }


def model_folder_name(model: Dict[str, str]) -> str:
    return f"{model['originating_centre']}_system{model['system']}"


def build_model_dir(
    root: Path,
    model: Dict[str, str],
    year: int,
    month: int,
    day: int,
) -> Path:
    """
    Example output folder:
        data/countries/ethiopia/seasonal/cds/
            seasonal-original-single-levels/
            2026/05/01/ecmwf_system51/
    """

    out_dir = (
        root
        / DATASET
        / f"{year:04d}"
        / f"{month:02d}"
        / f"{day:02d}"
        / model_folder_name(model)
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir


def build_output_path(
    root: Path,
    model: Dict[str, str],
    variable: str,
    year: int,
    month: int,
    day: int,
) -> Path:
    model_dir = build_model_dir(root, model, year, month, day)
    return model_dir / f"{variable}.nc"


def write_request_json(target_nc: Path, request: Dict) -> None:
    """
    Save exact CDS request beside each NetCDF file.
    """
    request_path = target_nc.with_suffix(".request.json")

    with request_path.open("w", encoding="utf-8") as f:
        json.dump(request, f, indent=2)


def download_with_retry(
    client: cdsapi.Client,
    request: Dict,
    target_nc: Path,
    overwrite: bool = False,
    max_retries: int = 3,
    sleep_seconds: int = 60,
) -> None:
    """
    Download one CDS request with retry logic.

    Uses temporary .part file and renames only after successful download.
    """

    if target_nc.exists() and target_nc.stat().st_size > 0 and not overwrite:
        logging.info("SKIP existing raw file: %s", target_nc)
        return

    tmp_file = target_nc.with_suffix(target_nc.suffix + ".part")

    if tmp_file.exists():
        tmp_file.unlink()

    write_request_json(target_nc, request)

    for attempt in range(1, max_retries + 1):
        try:
            logging.info("Downloading attempt %s/%s -> %s", attempt, max_retries, target_nc)

            result = client.retrieve(DATASET, request)
            result.download(str(tmp_file))

            if not tmp_file.exists() or tmp_file.stat().st_size == 0:
                raise RuntimeError(f"Downloaded file is empty: {tmp_file}")

            tmp_file.rename(target_nc)
            _ek_inspect_download(target_nc)   # earthkit-data field validation
            logging.info("DONE raw file: %s", target_nc)
            return

        except Exception as exc:
            logging.exception("FAILED attempt %s for %s", attempt, target_nc)

            if attempt == max_retries:
                if tmp_file.exists():
                    tmp_file.unlink()
                raise exc

            time.sleep(sleep_seconds * attempt)


# ---------------------------------------------------------------------
# 5b. Post-download earthkit-data field inspection
# ---------------------------------------------------------------------

def _ek_inspect_download(nc_path: Path) -> None:
    """
    After a successful CDS download, open the file with earthkit.data and
    log GRIB-style field metadata: shortName, paramId, units, grid shape,
    and total field count.  A quick sanity-check that CDS returned the
    right variable before the expensive derive / compute steps begin.
    """
    if not _EARTHKIT_DATA:
        return
    try:
        fl = ekd.from_source("file", str(nc_path))
        if len(fl) == 0:
            logging.warning("earthkit-data: no fields found in %s", nc_path.name)
            return
        m = fl[0].metadata()
        logging.info(
            "  ekd-validate  %-38s  shortName=%-6s  paramId=%-5s  "
            "units=%-12s  Nx=%-4s Ny=%-4s  nfields=%d",
            nc_path.name,
            m.get("GRIB_shortName",  "?"),
            m.get("GRIB_paramId",    "?"),
            m.get("units",           "?"),
            m.get("GRIB_Nx",         "?"),
            m.get("GRIB_Ny",         "?"),
            len(fl),
        )
    except Exception as exc:
        logging.debug("earthkit-data inspect skipped for %s: %s", nc_path.name, exc)


# ---------------------------------------------------------------------
# 6. Derived variable calculation
# ---------------------------------------------------------------------

def get_single_dataarray(ds: xr.Dataset, expected_name: str | None = None) -> xr.DataArray:
    """
    Return the main data variable from a one-variable CDS NetCDF file.

    CDS NetCDF variable names may be short names such as:
        t2m, d2m, u10, v10, tp, ssrd

    Because we save one CDS variable per file, the safest approach is to select
    the only data variable in the file.
    """

    data_vars = list(ds.data_vars)

    if expected_name and expected_name in data_vars:
        return ds[expected_name]

    if len(data_vars) == 1:
        return ds[data_vars[0]]

    raise ValueError(
        f"Expected one data variable, found {data_vars}. "
        "Open the file and inspect variable names."
    )


def saturation_vapor_pressure_kpa(temp_c: xr.DataArray) -> xr.DataArray:
    """
    FAO-56 saturation vapor pressure equation.

    Input:
        temperature in °C

    Output:
        saturation vapor pressure in kPa
    """
    return 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))


def derive_relative_humidity_and_vpd(
    t2m_k: xr.DataArray,
    d2m_k: xr.DataArray,
) -> xr.Dataset:
    """
    Derive relative humidity and vapor pressure deficit from:
        - 2m temperature
        - 2m dewpoint temperature

    Inputs are expected in Kelvin.
    """

    t2m_c = t2m_k - 273.15
    d2m_c = d2m_k - 273.15

    es = saturation_vapor_pressure_kpa(t2m_c)
    ea = saturation_vapor_pressure_kpa(d2m_c)

    rh = 100.0 * ea / es
    rh = rh.clip(min=0.0, max=100.0)

    vpd = es - ea
    vpd = vpd.clip(min=0.0)

    rh.name = "relative_humidity_2m"
    vpd.name = "vapor_pressure_deficit"

    rh.attrs = {
        "long_name": "2m relative humidity derived from 2m temperature and 2m dewpoint temperature",
        "units": "%",
        "method": "RH = 100 * e(Td) / es(T), where vapor pressure is calculated using FAO-56 saturation vapor pressure equation",
        "source_variables": "2m_temperature, 2m_dewpoint_temperature",
    }

    vpd.attrs = {
        "long_name": "Vapor pressure deficit derived from 2m temperature and 2m dewpoint temperature",
        "units": "kPa",
        "method": "VPD = es(T) - e(Td), where vapor pressure is calculated using FAO-56 saturation vapor pressure equation",
        "source_variables": "2m_temperature, 2m_dewpoint_temperature",
    }

    return xr.Dataset(
        {
            "relative_humidity_2m": rh,
            "vapor_pressure_deficit": vpd,
        }
    )


def derive_wind_speed(
    u10: xr.DataArray,
    v10: xr.DataArray,
) -> xr.Dataset:
    """
    Derive 10m wind speed from u and v wind components.
    """

    wind_speed = np.sqrt(u10**2 + v10**2)
    wind_speed.name = "wind_speed_10m"

    wind_speed.attrs = {
        "long_name": "10m wind speed derived from 10m u and v wind components",
        "units": "m s-1",
        "method": "wind_speed_10m = sqrt(u10^2 + v10^2)",
        "source_variables": "10m_u_component_of_wind, 10m_v_component_of_wind",
    }

    return xr.Dataset({"wind_speed_10m": wind_speed})


def compression_encoding(ds: xr.Dataset) -> Dict:
    """
    Compression settings for NetCDF output.
    Requires netCDF4 backend.
    """
    return {
        var: {
            "zlib": True,
            "complevel": 4,
            "_FillValue": -9999.0,
        }
        for var in ds.data_vars
    }


def derive_variables_for_model_dir(
    model_dir: Path,
    overwrite: bool = False,
) -> None:
    """
    Derive RH, VPD, and wind speed for one model folder.

    Expected files:
        2m_temperature.nc
        2m_dewpoint_temperature.nc
        10m_u_component_of_wind.nc
        10m_v_component_of_wind.nc

    Output:
        derived/relative_humidity_2m.nc
        derived/vapor_pressure_deficit.nc
        derived/wind_speed_10m.nc
        derived/agro_derived_variables.nc
    """

    required_files = {
        "t2m": model_dir / "2m_temperature.nc",
        "d2m": model_dir / "2m_dewpoint_temperature.nc",
        "u10": model_dir / "10m_u_component_of_wind.nc",
        "v10": model_dir / "10m_v_component_of_wind.nc",
    }

    missing = [str(path) for path in required_files.values() if not path.exists()]

    if missing:
        logging.warning("Cannot derive variables for %s. Missing files: %s", model_dir, missing)
        return

    derived_dir = model_dir / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)

    rh_file = derived_dir / "relative_humidity_2m.nc"
    vpd_file = derived_dir / "vapor_pressure_deficit.nc"
    wind_file = derived_dir / "wind_speed_10m.nc"
    combined_file = derived_dir / "agro_derived_variables.nc"

    if (
        rh_file.exists()
        and vpd_file.exists()
        and wind_file.exists()
        and combined_file.exists()
        and not overwrite
    ):
        logging.info("SKIP existing derived files: %s", derived_dir)
        return

    logging.info("Deriving agro-climate variables for: %s", model_dir)
    logging.info("  Using %s", "earthkit.meteo" if _EARTHKIT else "FAO-56 numpy fallback")

    # ------------------------------------------------------------------ #
    # Strategy: bulk-load ALL members at once per file (one disk read =   #
    # ~72 s per file), compute derived variables vectorised across all     #
    # members simultaneously, write, then free before loading next pair.  #
    # Peak RAM ≈ 2 × 15.6 GB = ~31 GB (fits comfortably in 64 GB).       #
    # ------------------------------------------------------------------ #

    def _load_all(ds: xr.Dataset) -> np.ndarray:
        """Load entire DataArray into a float32 ndarray (all members at once)."""
        da = get_single_dataarray(ds)
        arr = da.values.astype(np.float32)
        dims = da.dims
        coords = {k: da.coords[k].values for k in dims}
        return arr, dims, coords

    def _make_da(arr, dims, coords, name):
        return xr.DataArray(arr, dims=dims,
                            coords={k: coords[k] for k in dims},
                            name=name)

    # ── Humidity (RH + VPD) ─────────────────────────────────────────── #
    logging.info("  Loading t2m + d2m (all members) …")
    with xr.open_dataset(required_files["t2m"]) as ds_t2m, \
         xr.open_dataset(required_files["d2m"]) as ds_d2m:
        t2m_k, dims, coords = _load_all(ds_t2m)   # K, (n, step, lat, lon)
        d2m_k, _,    _      = _load_all(ds_d2m)   # K

    logging.info("  Computing RH and VPD …")
    if _EARTHKIT:
        # earthkit: inputs in K
        # relative_humidity_from_dewpoint returns % directly (NOT fraction)
        # saturation_vapour_pressure returns Pa → ×0.001 for kPa
        rh  = np.clip(_ek_thermo.relative_humidity_from_dewpoint(t2m_k, d2m_k),
                      0.0, 100.0).astype(np.float32)
        es  = _ek_thermo.saturation_vapour_pressure(t2m_k).astype(np.float64)
        ea  = _ek_thermo.saturation_vapour_pressure(d2m_k).astype(np.float64)
        vpd = np.maximum(es - ea, 0.0).astype(np.float32) * 0.001  # Pa → kPa
        del es, ea
    else:
        t2m_c = t2m_k - 273.15
        d2m_c = d2m_k - 273.15
        es  = 0.6108 * np.exp(17.27 * t2m_c / (t2m_c + 237.3))
        ea  = 0.6108 * np.exp(17.27 * d2m_c / (d2m_c + 237.3))
        rh  = np.clip(100.0 * ea / es, 0.0, 100.0).astype(np.float32)
        vpd = np.maximum(es - ea, 0.0).astype(np.float32)
        del t2m_c, d2m_c, es, ea
    del t2m_k, d2m_k; gc.collect()

    rh_da  = _make_da(rh,  dims, coords, "relative_humidity_2m")
    vpd_da = _make_da(vpd, dims, coords, "vapor_pressure_deficit")
    del rh, vpd; gc.collect()

    rh_da.attrs  = {"long_name": "2m relative humidity", "units": "%",
                    "source": "earthkit.meteo" if _EARTHKIT else "FAO-56"}
    vpd_da.attrs = {"long_name": "Vapor pressure deficit", "units": "kPa",
                    "source": "earthkit.meteo" if _EARTHKIT else "FAO-56"}

    hum_ds = xr.Dataset({"relative_humidity_2m": rh_da, "vapor_pressure_deficit": vpd_da})
    logging.info("  Writing RH and VPD …")
    hum_ds[["relative_humidity_2m"]].to_netcdf(
        rh_file,  encoding=compression_encoding(hum_ds[["relative_humidity_2m"]]))
    hum_ds[["vapor_pressure_deficit"]].to_netcdf(
        vpd_file, encoding=compression_encoding(hum_ds[["vapor_pressure_deficit"]]))
    del rh_da, vpd_da, hum_ds; gc.collect()

    # ── Wind speed ───────────────────────────────────────────────────── #
    logging.info("  Loading u10 + v10 (all members) …")
    with xr.open_dataset(required_files["u10"]) as ds_u10, \
         xr.open_dataset(required_files["v10"]) as ds_v10:
        u10, dims_w, coords_w = _load_all(ds_u10)
        v10, _,      _        = _load_all(ds_v10)

    logging.info("  Computing wind speed …")
    if _EARTHKIT:
        wind_spd = _ek_wind.speed(u10, v10).astype(np.float32)
    else:
        wind_spd = np.sqrt(u10 ** 2 + v10 ** 2).astype(np.float32)
    del u10, v10; gc.collect()

    wind_da = _make_da(wind_spd, dims_w, coords_w, "wind_speed_10m")
    wind_da.attrs = {"long_name": "10m wind speed", "units": "m s-1",
                     "source": "earthkit.meteo" if _EARTHKIT else "numpy"}
    del wind_spd; gc.collect()

    wind_ds = xr.Dataset({"wind_speed_10m": wind_da})
    logging.info("  Writing wind speed …")
    wind_ds.to_netcdf(wind_file, encoding=compression_encoding(wind_ds))

    # ── Combined file ────────────────────────────────────────────────── #
    logging.info("  Writing combined file …")
    combined_ds = xr.merge(
        [xr.open_dataset(rh_file), xr.open_dataset(vpd_file), wind_ds],
        compat="override",
    )
    combined_ds.attrs = {
        "title": "Derived agro-climate variables from CDS seasonal forecast",
        "source_dataset": DATASET,
        "source_model_folder": str(model_dir),
        "derived_variables": "relative_humidity_2m, vapor_pressure_deficit, wind_speed_10m",
        "computation": "earthkit.meteo" if _EARTHKIT else "FAO-56 numpy",
    }
    combined_ds.to_netcdf(combined_file, encoding=compression_encoding(combined_ds))
    del wind_da, wind_ds, combined_ds; gc.collect()

    logging.info("DONE derived files: %s", derived_dir)


# (setup_logging defined in section 9 below)


# ---------------------------------------------------------------------
# 8. Parallel download worker
# ---------------------------------------------------------------------

def _download_one_variable(
    model: Dict[str, str],
    variable: str,
    year: int,
    month: int,
    day: int,
    root: Path,
    max_lead_hour: int,
    area: List[float],
    overwrite: bool,
) -> str:
    """
    Download one (model, variable) pair. Creates its own cdsapi.Client so
    this function is safe to call from multiple threads simultaneously.

    Returns a label string on success; raises on failure.
    """
    client = cdsapi.Client(quiet=True)
    target_nc = build_output_path(root, model, variable, year, month, day)
    request = build_request(
        model=model,
        variable=variable,
        year=year,
        month=month,
        day=day,
        max_lead_hour=max_lead_hour,
        area=area,
        grid=GRID,
    )
    download_with_retry(
        client=client,
        request=request,
        target_nc=target_nc,
        overwrite=overwrite,
        max_retries=3,
        sleep_seconds=60,
    )
    return f"{model_folder_name(model)}/{variable}"


# ---------------------------------------------------------------------
# 9. Logging and CLI
# ---------------------------------------------------------------------

def setup_logging(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)

    log_file = root / "cds_agro_download_and_derivation.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CDS seasonal agro-surface variables and derive RH, VPD, and wind speed."
    )

    parser.add_argument("--year",  type=int, required=True,
                        help="Forecast initialization year, e.g. 2026")
    parser.add_argument("--month", type=int, required=True,
                        help="Forecast initialization month, e.g. 5")
    parser.add_argument("--day",   type=int, default=1,
                        help="Forecast initialization day. Default: 1")
    parser.add_argument("--country", type=str, default=DEFAULT_COUNTRY,
                        help="Country slug in config/countries. Default: ethiopia.")
    parser.add_argument("--root",  type=str,
                        default=None,
                        help="Root output directory. Default: data/countries/<country>/seasonal/cds.")
    parser.add_argument("--area-nwse", type=float, nargs=4, default=None,
                        help="Override CDS area as: north west south east.")
    parser.add_argument("--max-lead-hour", type=int, default=5160,
                        help="Maximum lead time in hours. Default: 5160")
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Model centres to download, e.g. --models ecmwf ukmo. Default: all.",
    )
    parser.add_argument(
        "--workers", type=int,
        default=min(9, len(AGRO_SURFACE_CDS_VARIABLES)),
        help=(
            "Number of parallel CDS download threads. Each thread submits an "
            "independent request and owns its cdsapi.Client. "
            f"Default: {min(9, len(AGRO_SURFACE_CDS_VARIABLES))} "
            "(one per variable). CDS queues requests beyond its own concurrency limit."
        ),
    )
    parser.add_argument("--overwrite",    action="store_true",
                        help="Overwrite existing raw and derived files.")
    parser.add_argument("--derive-only",  action="store_true",
                        help="Skip CDS download; only derive variables from existing raw files.")
    parser.add_argument("--no-derive",    action="store_true",
                        help="Only download raw variables; skip derivation.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root) if args.root else cds_root(args.country)
    area = list(args.area_nwse) if args.area_nwse else country_cds_area(args.country)
    setup_logging(root)

    if args.models:
        selected_models = [
            m for m in MODELS if m["originating_centre"] in args.models
        ]
    else:
        selected_models = MODELS

    if not selected_models:
        raise ValueError(f"No matching models selected from: {args.models}")

    n_tasks = len(selected_models) * len(AGRO_SURFACE_CDS_VARIABLES)
    n_workers = min(args.workers, n_tasks)

    logging.info("Dataset        : %s", DATASET)
    logging.info("Date           : %04d-%02d-%02d", args.year, args.month, args.day)
    logging.info("Country        : %s", args.country)
    logging.info("Root           : %s", root)
    logging.info("Models         : %s", [m["originating_centre"] for m in selected_models])
    logging.info("Variables      : %s", AGRO_SURFACE_CDS_VARIABLES)
    logging.info("Area [N,W,S,E] : %s", area)
    logging.info("Grid           : %s", GRID)
    logging.info("Max lead hour  : %s", args.max_lead_hour)
    logging.info("Download workers: %d  (tasks: %d)", n_workers, n_tasks)

    # ------------------------------------------------------------------
    # Parallel downloads — all (model, variable) pairs simultaneously.
    # Each thread creates its own cdsapi.Client; CDS queues excess requests.
    # ------------------------------------------------------------------
    if not args.derive_only:
        # Track how many variables each model has completed (thread-safe counter)
        completed: Dict[str, Set[str]] = {
            model_folder_name(m): set() for m in selected_models
        }
        completed_lock = Lock()

        def _task(model: Dict, variable: str) -> str:
            label = _download_one_variable(
                model=model, variable=variable,
                year=args.year, month=args.month, day=args.day,
                root=root, max_lead_hour=args.max_lead_hour,
                area=area,
                overwrite=args.overwrite,
            )
            mname = model_folder_name(model)
            with completed_lock:
                completed[mname].add(variable)
                n_done = len(completed[mname])
            logging.info(
                "✓ [%d/%d] %s", n_done, len(AGRO_SURFACE_CDS_VARIABLES), label
            )
            # Trigger derive for this model as soon as ALL its variables land
            if not args.no_derive and n_done == len(AGRO_SURFACE_CDS_VARIABLES):
                model_dir = build_model_dir(root, model, args.year, args.month, args.day)
                logging.info("→ All variables ready for %s — deriving …", mname)
                with _derive_lock:
                    derive_variables_for_model_dir(model_dir, args.overwrite)
                logging.info("✓ Derivation complete: %s", mname)
            return label

        futures_map = {}
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for model in selected_models:
                for variable in AGRO_SURFACE_CDS_VARIABLES:
                    fut = pool.submit(_task, model, variable)
                    futures_map[fut] = (model_folder_name(model), variable)

            failed = []
            for fut in as_completed(futures_map):
                mname, variable = futures_map[fut]
                exc = fut.exception()
                if exc:
                    logging.error("✗ FAILED %s/%s: %s", mname, variable, exc)
                    failed.append(f"{mname}/{variable}")

        if failed:
            logging.warning(
                "%d download(s) failed: %s", len(failed), failed
            )

    # ------------------------------------------------------------------
    # Derive-only mode — no downloads, just compute derived variables
    # ------------------------------------------------------------------
    elif args.derive_only and not args.no_derive:
        for model in selected_models:
            model_dir = build_model_dir(root, model, args.year, args.month, args.day)
            derive_variables_for_model_dir(model_dir, args.overwrite)


if __name__ == "__main__":
    main()
