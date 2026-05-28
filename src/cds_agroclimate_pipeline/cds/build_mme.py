#!/usr/bin/env python3
"""
cds/build_mme.py
==============================
Merge per-model ensemble_statistics.nc files into a single
Multi-Model Ensemble (MME) dataset.

For each of the 23 agro-climate indices the MME computes:
  • mme_mean      — equal-weighted mean across all available model means
  • mme_spread    — inter-model std (uncertainty in the signal)
  • mme_min / mme_max — range across models
  • mme_p10/p50/p90   — mean of each model's percentile (conservative range)
  • mme_agreement — fraction of models with mean > reference threshold
  • mme_n_models  — number of models that contributed

All model grids are regridded to the first available model grid using
bilinear interpolation.

Output: {root}/seasonal-original-single-levels/{Y}/{M}/{D}/
         multimodel_ensemble/mme_statistics.nc

Usage:
    python -m cds_agroclimate_pipeline.cds.build_mme --year 2026 --month 5 [--day 1]
    python -m cds_agroclimate_pipeline.cds.build_mme --year 2026 --month 5 --models ecmwf ukmo ncep
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import xarray as xr

from cds_agroclimate_pipeline.paths import DEFAULT_COUNTRY, cds_root

# ─────────────────────────────────────────────────────────────────────────────
# 1. Model registry (originating_centre → system number)
# ─────────────────────────────────────────────────────────────────────────────
ALL_MODELS: List[Dict[str, str]] = [
    {"originating_centre": "ecmwf",        "system": "51"},
    {"originating_centre": "ukmo",         "system": "610"},
    {"originating_centre": "meteo_france", "system": "9"},
    {"originating_centre": "dwd",          "system": "22"},
    {"originating_centre": "cmcc",         "system": "4"},
    {"originating_centre": "ncep",         "system": "2"},
    {"originating_centre": "jma",          "system": "4"},
    {"originating_centre": "eccc",         "system": "5"},
    {"originating_centre": "bom",          "system": "2"},
]

# Long names for display
MODEL_LABELS: Dict[str, str] = {
    "ecmwf_system51":        "ECMWF S51",
    "ukmo_system610":        "UKMO S610",
    "meteo_france_system9":  "Météo-France S9",
    "dwd_system22":          "DWD S22",
    "cmcc_system4":          "CMCC S4",
    "ncep_system2":          "NCEP CFSv2",
    "jma_system4":           "JMA S4",
    "eccc_system5":          "ECCC S5",
    "bom_system2":           "BoM S2",
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. Helpers
# ─────────────────────────────────────────────────────────────────────────────
def model_folder_name(m: Dict[str, str]) -> str:
    return f"{m['originating_centre']}_system{m['system']}"


def build_model_dir(root: Path, model_name: str, year: int, month: int, day: int) -> Path:
    return (root / "seasonal-original-single-levels"
            / f"{year}" / f"{month:02d}" / f"{day:02d}" / model_name)


def stats_path(model_dir: Path) -> Path:
    return model_dir / "indices" / "ensemble_statistics.nc"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Multi-Model Ensemble statistics")
    p.add_argument("--year",   type=int, default=2026)
    p.add_argument("--month",  type=int, default=5)
    p.add_argument("--day",    type=int, default=1)
    p.add_argument("--country", type=str, default=DEFAULT_COUNTRY,
                   help="Country slug in config/countries. Default: ethiopia.")
    p.add_argument("--root",   type=Path,
                   default=None,
                   help="CDS root. Default: data/countries/<country>/seasonal/cds.")
    p.add_argument("--models", nargs="*",
                   help="Subset of model originating_centre names. Default: all available.")
    p.add_argument("--min-models", type=int, default=2,
                   help="Minimum number of models required to produce MME (default 2).")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Reference grid helpers
# ─────────────────────────────────────────────────────────────────────────────
def reference_grid_from_dataset(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    return np.asarray(ds[lat_name].values, dtype=np.float32), np.asarray(ds[lon_name].values, dtype=np.float32)


def regrid_to_reference(ds: xr.Dataset, ref_lat: np.ndarray, ref_lon: np.ndarray) -> xr.Dataset:
    """Bilinearly interpolate ds onto the selected country/model reference grid."""
    # Normalise coordinate names
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"

    ds = ds.rename({lat_name: "latitude", lon_name: "longitude"})

    # xarray interpolation is happiest with ascending source/target coordinates.
    if float(ds.latitude[0]) > float(ds.latitude[-1]):
        ds = ds.isel(latitude=slice(None, None, -1))
    if float(ds.longitude[0]) > float(ds.longitude[-1]):
        ds = ds.isel(longitude=slice(None, None, -1))

    interp_lat = np.sort(ref_lat)
    interp_lon = np.sort(ref_lon)

    ds_out = ds.interp(
        latitude=xr.DataArray(interp_lat, dims="latitude"),
        longitude=xr.DataArray(interp_lon, dims="longitude"),
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )
    return ds_out.reindex(latitude=ref_lat, longitude=ref_lon)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Discover unique index base names (strip _mean / _p10 / _std etc.)
# ─────────────────────────────────────────────────────────────────────────────
STAT_SUFFIXES = ("_mean", "_std", "_p10", "_p25", "_p50", "_p75", "_p90")

def index_base_names(ds: xr.Dataset) -> List[str]:
    bases = set()
    for v in ds.data_vars:
        for suf in STAT_SUFFIXES:
            if v.endswith(suf):
                bases.add(v[: -len(suf)])
                break
    return sorted(bases)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main MME builder
# ─────────────────────────────────────────────────────────────────────────────
def build_mme(
    datasets: Dict[str, xr.Dataset],   # {folder_name: ds}
    model_names: List[str],
    init_date: str,
    ref_lat: np.ndarray,
    ref_lon: np.ndarray,
) -> xr.Dataset:
    """
    Compute MME statistics from a dict of per-model datasets.
    Returns an xarray.Dataset with variables:
        {index}_mme_mean, {index}_mme_spread, {index}_mme_min,
        {index}_mme_max, {index}_mme_p10, {index}_mme_p50,
        {index}_mme_p90, {index}_mme_agreement, {index}_mme_n_models
    """
    # Determine common index bases across all models
    all_bases: Optional[set] = None
    for ds in datasets.values():
        b = set(index_base_names(ds))
        all_bases = b if all_bases is None else (all_bases & b)
    if not all_bases:
        raise ValueError("No common index bases found across models.")
    bases = sorted(all_bases)

    logging.info("  Common indices: %d  (%s … %s)", len(bases), bases[0], bases[-1])

    n_models = len(model_names)

    mme_vars: Dict[str, xr.DataArray] = {}

    for base in bases:
        # Stack model means → shape (n_models, lat, lon)
        means_list, p10_list, p50_list, p90_list = [], [], [], []

        for mn in model_names:
            ds = datasets[mn]
            means_list.append(ds[f"{base}_mean"].values)
            if f"{base}_p10" in ds:
                p10_list.append(ds[f"{base}_p10"].values)
            if f"{base}_p50" in ds:
                p50_list.append(ds[f"{base}_p50"].values)
            if f"{base}_p90" in ds:
                p90_list.append(ds[f"{base}_p90"].values)

        means = np.array(means_list)                 # (n_models, lat, lon)
        n_valid = np.sum(np.isfinite(means), axis=0) # (lat, lon)

        def _da(arr: np.ndarray, name: str) -> xr.DataArray:
            return xr.DataArray(
                arr.astype(np.float32),
                dims=["latitude", "longitude"],
                coords={"latitude": ref_lat, "longitude": ref_lon},
                name=name,
            )

        # MME mean (nanmean across models)
        mme_mean = np.nanmean(means, axis=0)
        mme_vars[f"{base}_mme_mean"] = _da(mme_mean, f"{base}_mme_mean")

        # Inter-model spread (nanstd)
        mme_spread = np.nanstd(means, axis=0, ddof=0)
        mme_vars[f"{base}_mme_spread"] = _da(mme_spread, f"{base}_mme_spread")

        # Min / max across models
        mme_vars[f"{base}_mme_min"] = _da(np.nanmin(means, axis=0), f"{base}_mme_min")
        mme_vars[f"{base}_mme_max"] = _da(np.nanmax(means, axis=0), f"{base}_mme_max")

        # Agreement: fraction of models > MME mean  (directional agreement)
        agreement = np.sum(means > mme_mean[np.newaxis], axis=0) / n_valid.clip(1)
        # Remap to 0–1: 0.5 = no agreement, 1.0 = all above, 0.0 = all below
        mme_vars[f"{base}_mme_agreement"] = _da(
            agreement.astype(np.float32), f"{base}_mme_agreement")

        # Number of models contributing
        mme_vars[f"{base}_mme_n_models"] = _da(
            n_valid.astype(np.float32), f"{base}_mme_n_models")

        # Percentile averages
        if p10_list:
            mme_vars[f"{base}_mme_p10"] = _da(
                np.nanmean(p10_list, axis=0), f"{base}_mme_p10")
        if p50_list:
            mme_vars[f"{base}_mme_p50"] = _da(
                np.nanmean(p50_list, axis=0), f"{base}_mme_p50")
        if p90_list:
            mme_vars[f"{base}_mme_p90"] = _da(
                np.nanmean(p90_list, axis=0), f"{base}_mme_p90")

    ds_mme = xr.Dataset(mme_vars)
    ds_mme.attrs = {
        "title":                       "CDS Multi-Model Ensemble (MME) agro-climate indices",
        "source_dataset":              "seasonal-original-single-levels",
        "forecast_initialization_date": init_date,
        "models_included":             ", ".join(model_names),
        "n_models":                    n_models,
        "common_indices":              len(bases),
        "grid":                        f"{len(ref_lat)} lat × {len(ref_lon)} lon — first available model reference",
        "mme_method":                  "Equal-weight mean across model ensemble means",
        "agreement_definition":        "Fraction of models with mean > MME mean (0=all below, 1=all above)",
    }
    return ds_mme


# ─────────────────────────────────────────────────────────────────────────────
# 6. Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    root = args.root or cds_root(args.country)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    init_date = f"{args.year}-{args.month:02d}-{args.day:02d}"
    print()
    print("=" * 70)
    print(f"  Multi-Model Ensemble (MME)  |  Init: {init_date}")
    print(f"  Country: {args.country}  |  Root: {root}")
    print("=" * 70)

    # Resolve requested models
    if args.models:
        selected = [m for m in ALL_MODELS if m["originating_centre"] in args.models]
    else:
        selected = ALL_MODELS

    # Check which models have ensemble_statistics.nc ready
    available: Dict[str, xr.Dataset] = {}
    missing: List[str] = []
    ref_lat: Optional[np.ndarray] = None
    ref_lon: Optional[np.ndarray] = None
    for m in selected:
        folder = model_folder_name(m)
        mdir = build_model_dir(root, folder, args.year, args.month, args.day)
        sp = stats_path(mdir)
        if sp.exists():
            try:
                ds = xr.open_dataset(sp)
                if ref_lat is None or ref_lon is None:
                    ref_lat, ref_lon = reference_grid_from_dataset(ds)
                    logging.info("  reference grid: %d lat × %d lon from %s",
                                 len(ref_lat), len(ref_lon), folder)
                ds_regrid = regrid_to_reference(ds, ref_lat, ref_lon)
                available[folder] = ds_regrid
                logging.info("  ✓ loaded  %s  (%d vars)", folder, len(ds.data_vars))
            except Exception as exc:
                logging.warning("  ✗ failed to load %s: %s", folder, exc)
                missing.append(folder)
        else:
            missing.append(folder)
            logging.warning("  ✗ missing  %s  (no ensemble_statistics.nc)", folder)

    if missing:
        print(f"\n  Models NOT ready ({len(missing)}): {', '.join(missing)}")
    print(f"  Models ready    ({len(available)}): {', '.join(available.keys())}")

    if len(available) < args.min_models:
        print(f"\n  ERROR: only {len(available)} model(s) available "
              f"(need at least {args.min_models}). Aborting.")
        sys.exit(1)
    if ref_lat is None or ref_lon is None:
        raise RuntimeError("No reference grid could be inferred from available models.")

    # Output directory
    base_date_dir = (root / "seasonal-original-single-levels"
                     / f"{args.year}" / f"{args.month:02d}" / f"{args.day:02d}")
    out_dir = base_date_dir / "multimodel_ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mme_statistics.nc"

    print(f"\n  Building MME from {len(available)} models …")
    model_names = list(available.keys())
    ds_mme = build_mme(available, model_names, init_date, ref_lat, ref_lon)

    # Save
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds_mme.data_vars}
    ds_mme.to_netcdf(out_path, encoding=encoding)

    size_mb = out_path.stat().st_size / 1e6
    print()
    print("  " + "=" * 60)
    print(f"  MME saved: mme_statistics.nc")
    print(f"  Location:  {out_path}")
    print(f"  Size:      {size_mb:.1f} MB  |  Indices: {len(ds_mme.data_vars)} vars")
    print(f"  Models:    {', '.join(model_names)}")
    print("  " + "=" * 60)
    print()


if __name__ == "__main__":
    main()
