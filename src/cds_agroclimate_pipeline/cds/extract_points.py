#!/usr/bin/env python3
# cds/extract_points.py
"""
Extract CDS seasonal forecast values at arbitrary station / site coordinates
using earthkit.geo.GeoKDTree for fast nearest-neighbour grid-to-point mapping.

Author: Jemal Ahmed
Email:  J.Ahmed@cgiar.org

Usage
-----
# Extract at a list of lat/lon points from a CSV
uv run python -m cds_agroclimate_pipeline.cds.extract_points \\
    --year 2026 --month 5 --model ecmwf \\
    --sites sites.csv \\
    --output forecasts_at_sites.csv

# Quick test with built-in demo sites
uv run python -m cds_agroclimate_pipeline.cds.extract_points \\
    --year 2026 --month 5 --model ecmwf --demo

Sites CSV format (no header required beyond column names):
    name,lat,lon
    Addis Ababa,9.03,38.74
    Nairobi,-1.29,36.82
    ...

Outputs
-------
One CSV row per (site × index) with ensemble statistics (mean / std / p10–p90).
Also writes a GeoJSON file with the same data for GIS inspection.
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from cds_agroclimate_pipeline.paths import DEFAULT_COUNTRY, cds_root

# earthkit-geo: geographic nearest-point lookups via KD-tree on a sphere
import earthkit.geo as ekg
from earthkit.geo import GeoKDTree

warnings.filterwarnings("ignore", category=UserWarning, module="gribapi")
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Optional imports ───────────────────────────────────────────────────────
try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

# ─────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────

DATASET = "seasonal-original-single-levels"

DEMO_SITES = [
    # Major Ethiopian cities and agricultural zones
    {"name": "Addis Ababa",    "lat":  9.03,  "lon": 38.74},
    {"name": "Dire Dawa",      "lat":  9.60,  "lon": 41.86},
    {"name": "Mekelle",        "lat": 13.50,  "lon": 39.48},
    {"name": "Gondar",         "lat": 12.60,  "lon": 37.47},
    {"name": "Bahir Dar",      "lat": 11.59,  "lon": 37.39},
    {"name": "Jimma",          "lat":  7.67,  "lon": 36.83},
    {"name": "Hawassa",        "lat":  7.05,  "lon": 38.48},
    {"name": "Arba Minch",     "lat":  6.04,  "lon": 37.55},
    {"name": "Jijiga",         "lat":  9.35,  "lon": 42.79},
    {"name": "Gambela",        "lat":  8.25,  "lon": 34.58},
    {"name": "Assosa",         "lat": 10.07,  "lon": 34.53},
    {"name": "Nekemte",        "lat":  9.08,  "lon": 36.55},
]

# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def model_folder_name(model: str) -> str:
    SYSTEMS = {
        "ecmwf": "ecmwf_system51", "ukmo": "ukmo_system610",
        "ncep":  "ncep_system2",   "jma":  "jma_system4",
        "cmcc":  "cmcc_system4",  "dwd":  "dwd_system22",
        "meteo_france": "meteo_france_system9",
        "eccc":  "eccc_system5",   "bom":  "bom_system2",
    }
    return SYSTEMS.get(model.lower(), model)


def build_model_dir(root: Path, model: str, year: int, month: int, day: int) -> Path:
    return (
        root / DATASET
        / f"{year:04d}" / f"{month:02d}" / f"{day:02d}"
        / model_folder_name(model)
    )


def load_sites_csv(csv_path: Path) -> List[Dict]:
    """Load sites from CSV with columns: name, lat, lon."""
    sites = []
    with open(csv_path) as f:
        header = None
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if header is None:
                header = [p.lower() for p in parts]
                continue
            row = dict(zip(header, parts))
            sites.append({
                "name": row.get("name", f"site_{len(sites)}"),
                "lat":  float(row["lat"]),
                "lon":  float(row["lon"]),
            })
    return sites


# ─────────────────────────────────────────────────────────────────────────
# earthkit-geo: build KDTree from grid, extract nearest point per site
# ─────────────────────────────────────────────────────────────────────────

def build_grid_kdtree(latitudes: np.ndarray, longitudes: np.ndarray) -> GeoKDTree:
    """
    Build an earthkit.geo.GeoKDTree from a 2-D lat/lon grid.

    GeoKDTree operates on a sphere (great-circle distance) so nearest-point
    queries return the true closest grid cell even near the poles.
    """
    lat_2d = np.repeat(latitudes[:, np.newaxis],  longitudes.size, axis=1)  # (nlat, nlon)
    lon_2d = np.repeat(longitudes[np.newaxis, :], latitudes.size,  axis=0)  # (nlat, nlon)
    return GeoKDTree(lat_2d.ravel(), lon_2d.ravel())


def nearest_grid_indices(
    kdtree: GeoKDTree,
    site_lat: float,
    site_lon: float,
    n_lat: int,
    n_lon: int,
) -> Tuple[int, int]:
    """
    Return (lat_idx, lon_idx) of the nearest grid cell to (site_lat, site_lon).

    earthkit.geo.GeoKDTree.nearest_point returns a flat index into the
    ravelled (n_lat × n_lon) grid; we unravel to (row, col).
    """
    # GeoKDTree.nearest_point expects a single (lats, lons) tuple
    result   = kdtree.nearest_point(
        (np.array([site_lat]), np.array([site_lon]))
    )
    flat_idx = int(np.asarray(result).flat[0])
    return int(flat_idx // n_lon), int(flat_idx % n_lon)


# ─────────────────────────────────────────────────────────────────────────
# Main extraction logic
# ─────────────────────────────────────────────────────────────────────────

def extract_at_sites(
    model_dir: Path,
    sites: List[Dict],
    year: int,
    month: int,
    day: int,
    model: str,
) -> List[Dict]:
    """
    For every site, find the nearest grid cell using earthkit.geo.GeoKDTree,
    then extract all index variables from ensemble_statistics.nc.

    Returns a list of records (one per site × index) ready for CSV / GeoJSON.
    """
    ens_path = model_dir / "indices" / "ensemble_statistics.nc"
    monthly_path = model_dir / "indices" / "monthly_weather_summaries.nc"

    if not ens_path.exists():
        logging.error("ensemble_statistics.nc not found: %s", ens_path)
        return []

    ds_ens = xr.open_dataset(ens_path)
    latitudes  = ds_ens["latitude"].values
    longitudes = ds_ens["longitude"].values
    n_lat, n_lon = latitudes.size, longitudes.size

    # ── Build GeoKDTree once for the whole grid ────────────────────────
    logging.info(
        "Building earthkit.geo.GeoKDTree for %d×%d grid …", n_lat, n_lon
    )
    kdtree = build_grid_kdtree(latitudes, longitudes)
    logging.info("  KDTree built. Extracting %d sites …", len(sites))

    init_date = f"{year:04d}-{month:02d}-{day:02d}"
    records: List[Dict] = []

    for site in sites:
        s_lat, s_lon = site["lat"], site["lon"]

        # ── earthkit-geo nearest point ─────────────────────────────────
        row, col = nearest_grid_indices(kdtree, s_lat, s_lon, n_lat, n_lon)
        grid_lat = float(latitudes[row])
        grid_lon = float(longitudes[col])
        # haversine_distance(points1, points2) where each is (lats, lons) tuple
        dist_km  = float(
            ekg.haversine_distance(
                (np.array([s_lat]),    np.array([s_lon])),
                (np.array([grid_lat]), np.array([grid_lon])),
            ).flat[0]
        ) / 1000.0

        logging.info(
            "  %-22s  query=(%.2f,%.2f)  nearest=(%.2f,%.2f)  dist=%.1f km",
            site["name"], s_lat, s_lon, grid_lat, grid_lon, dist_km,
        )

        base = {
            "site":      site["name"],
            "site_lat":  s_lat,
            "site_lon":  s_lon,
            "grid_lat":  grid_lat,
            "grid_lon":  grid_lon,
            "dist_km":   round(dist_km, 2),
            "model":     model,
            "init_date": init_date,
        }

        # ── Extract all index statistics at this grid cell ─────────────
        for var in ds_ens.data_vars:
            val = float(ds_ens[var].values[row, col])
            records.append({**base, "variable": var, "value": round(val, 4)})

    ds_ens.close()

    # ── Also extract monthly summaries if available ────────────────────
    if monthly_path.exists():
        ds_mon = xr.open_dataset(monthly_path)
        # Ensemble mean across members
        if "number" in ds_mon.dims:
            ds_mon_mean = ds_mon.mean(dim="number")
        else:
            ds_mon_mean = ds_mon

        for site in sites:
            row, col = nearest_grid_indices(
                kdtree, site["lat"], site["lon"], n_lat, n_lon
            )
            base = {
                "site":      site["name"],
                "site_lat":  site["lat"],
                "site_lon":  site["lon"],
                "grid_lat":  float(latitudes[row]),
                "grid_lon":  float(longitudes[col]),
                "model":     model,
                "init_date": init_date,
            }
            for var in ds_mon_mean.data_vars:
                if "time" in ds_mon_mean[var].dims:
                    for t_idx, t_val in enumerate(ds_mon_mean["time"].values):
                        val = float(ds_mon_mean[var].isel(time=t_idx).values[row, col])
                        month_str = str(t_val)[:7]
                        records.append({
                            **base,
                            "variable": f"monthly_{var}_{month_str}",
                            "value": round(val, 4),
                        })
        ds_mon.close()

    return records


# ─────────────────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────────────────

def write_csv(records: List[Dict], out_path: Path) -> None:
    if not records:
        logging.warning("No records to write.")
        return
    if _HAS_PANDAS:
        import pandas as pd
        pd.DataFrame(records).to_csv(out_path, index=False)
    else:
        keys = list(records[0].keys())
        with open(out_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for r in records:
                f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    logging.info("Saved CSV: %s  (%d rows)", out_path, len(records))


def write_geojson(records: List[Dict], out_path: Path) -> None:
    """
    Write records as a GeoJSON FeatureCollection.
    One feature per (site × variable), geometry = site location.
    """
    # Group by site name to create one feature per site with all values
    sites_seen: Dict[str, Dict] = {}
    for r in records:
        name = r["site"]
        if name not in sites_seen:
            sites_seen[name] = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [r["site_lon"], r["site_lat"]],
                },
                "properties": {
                    "name":      name,
                    "site_lat":  r["site_lat"],
                    "site_lon":  r["site_lon"],
                    "grid_lat":  r.get("grid_lat"),
                    "grid_lon":  r.get("grid_lon"),
                    "dist_km":   r.get("dist_km"),
                    "model":     r["model"],
                    "init_date": r["init_date"],
                },
            }
        sites_seen[name]["properties"][r["variable"]] = r["value"]

    fc = {"type": "FeatureCollection", "features": list(sites_seen.values())}
    with open(out_path, "w") as f:
        json.dump(fc, f, indent=2)
    logging.info("Saved GeoJSON: %s  (%d sites)", out_path, len(sites_seen))


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract CDS seasonal forecast indices at station coordinates "
            "using earthkit.geo.GeoKDTree for nearest-grid-cell lookup."
        )
    )
    parser.add_argument("--year",  type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--day",   type=int, default=1)
    parser.add_argument(
        "--model", default="ecmwf",
        help="Forecast model centre (ecmwf, ukmo, ncep, …).",
    )
    parser.add_argument(
        "--sites",
        type=Path, default=None,
        help="CSV file with columns: name, lat, lon.  "
             "If omitted, --demo uses built-in African city list.",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Use built-in demo sites (10 African cities).",
    )
    parser.add_argument(
        "--output",
        type=Path, default=None,
        help="Output CSV path.  Default: <model_dir>/indices/point_forecasts.csv",
    )
    parser.add_argument(
        "--country",
        default=DEFAULT_COUNTRY,
        help="Country slug in config/countries. Default: ethiopia.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Pipeline root directory. Default: data/countries/<country>/seasonal/cds.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    root = Path(args.root) if args.root else cds_root(args.country)

    # ── Load sites ─────────────────────────────────────────────────────
    if args.sites:
        sites = load_sites_csv(args.sites)
        logging.info("Loaded %d sites from %s", len(sites), args.sites)
    elif args.demo:
        sites = DEMO_SITES
        logging.info("Using %d built-in demo sites.", len(sites))
    else:
        logging.error("Provide --sites CSV or use --demo."); return

    # ── Locate model directory ─────────────────────────────────────────
    model_dir = build_model_dir(root, args.model, args.year, args.month, args.day)
    if not model_dir.exists():
        logging.error("Model directory not found: %s", model_dir); return

    # ── Extract ────────────────────────────────────────────────────────
    records = extract_at_sites(model_dir, sites, args.year, args.month, args.day, args.model)
    if not records:
        logging.warning("No data extracted."); return

    # ── Write outputs ──────────────────────────────────────────────────
    out_dir = model_dir / "indices"
    csv_path  = args.output or out_dir / "point_forecasts.csv"
    json_path = csv_path.with_suffix(".geojson")

    write_csv(records, csv_path)
    write_geojson(records, json_path)
    logging.info("Done.  %d site×variable records written.", len(records))


if __name__ == "__main__":
    main()
