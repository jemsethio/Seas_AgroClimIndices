#!/usr/bin/env python3
# visualization/maps.py
"""
Map seasonal agro-climate indices and monthly weather summaries
using earthkit.plots.

Author: Jemal Ahmed
Email:  J.Ahmed@cgiar.org

Usage
-----
# Seasonal index maps for one model
uv run python -m cds_agroclimate_pipeline.visualization.maps --year 2026 --month 5 --model ecmwf

# Monthly precipitation / temperature panels
uv run python -m cds_agroclimate_pipeline.visualization.maps --year 2026 --month 5 --model ecmwf --monthly

# All models (saves one figure per model)
uv run python -m cds_agroclimate_pipeline.visualization.maps --year 2026 --month 5 --all-models

Outputs are saved as PNG under {model_dir}/indices/plots/.
"""
from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from cds_agroclimate_pipeline.paths import (
    DEFAULT_COUNTRY,
    cds_root,
    configure_matplotlib_cache,
    country_bbox_lonlat,
)

configure_matplotlib_cache()

import matplotlib
matplotlib.use("Agg")                    # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import earthkit.plots as ekp             # maps, quickplot, Figure

warnings.filterwarnings("ignore", category=UserWarning, module="gribapi")
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

DATASET  = "seasonal-original-single-levels"


def _earthkit_domain(country: str) -> list[float]:
    lon_min, lon_max, lat_min, lat_max = country_bbox_lonlat(country)
    return [lon_min, lat_min, lon_max, lat_max]


ETHIOPIA_DOMAIN = _earthkit_domain(DEFAULT_COUNTRY)

# Indices to map: (variable_name_in_nc, title, colormap, vmin, vmax, unit_label)
SEASONAL_PANELS: List[Tuple[str, str, str, float, float, str]] = [
    ("rainfall_total_mean",            "Seasonal Total Rainfall",         "Blues",         0,   1200, "mm"),
    ("et0_total_mean",                 "Seasonal Total ET₀",              "YlOrRd",        0,   1800, "mm"),
    ("water_stress_index_mean_mean",   "Mean Water Stress Index",         "RdYlGn_r",      0,    1.0, "0–1"),
    ("growing_degree_days_mean",       "Growing Degree Days",             "YlOrBr",        0,   3500, "°C·d"),
    ("dry_spell_max_days_mean",        "Longest Dry Spell",               "hot_r",         0,     90, "days"),
    ("advisory_score_mean",            "Composite Advisory Score",        "RdYlGn",        0,    100, "score"),
]

MONTHLY_PANELS: List[Tuple[str, str, str, float, float, str]] = [
    ("precipitation",       "Monthly Precipitation",        "Blues",   0,   400, "mm month⁻¹"),
    ("temperature_2m_mean", "Monthly Mean Temperature",     "RdBu_r", 10,    40, "°C"),
    ("shortwave_radiation", "Monthly Solar Radiation",      "YlOrRd",  0,   800, "MJ m⁻² month⁻¹"),
    ("relative_humidity_2m","Monthly Mean RH",              "BuGn",    0,   100, "%"),
    ("vapor_pressure_deficit","Monthly Mean VPD",           "YlOrRd",  0,     3, "kPa"),
    ("wind_speed_10m",      "Monthly Mean Wind Speed",      "PuBu",    0,     8, "m s⁻¹"),
]


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def model_folder_name(model: str) -> str:
    """Convert 'ecmwf' → 'ecmwf_system51', etc."""
    SYSTEMS = {
        "ecmwf": "ecmwf_system51",
        "ukmo":  "ukmo_system610",
        "ncep":  "ncep_system2",
        "jma":   "jma_system4",
        "cmcc":  "cmcc_system4",
        "dwd":   "dwd_system22",
        "meteo_france": "meteo_france_system9",
        "eccc":  "eccc_system5",
        "bom":   "bom_system2",
    }
    return SYSTEMS.get(model.lower(), model)


def build_model_dir(root: Path, model: str, year: int, month: int, day: int) -> Path:
    folder = model_folder_name(model)
    return root / DATASET / f"{year:04d}" / f"{month:02d}" / f"{day:02d}" / folder


def _load_nc(path: Path) -> Optional[xr.Dataset]:
    if not path.exists() or path.stat().st_size == 0:
        logging.warning("File not found: %s", path)
        return None
    try:
        return xr.open_dataset(path)
    except Exception as exc:
        logging.error("Cannot open %s: %s", path, exc)
        return None


def _safe_title(model: str, year: int, month: int) -> str:
    import calendar
    return f"{model.upper()}  |  Init: {year}-{month:02d}-01  |  {calendar.month_abbr[month]} {year} Forecast"


# -------------------------------------------------------------------------
# Seasonal index panel
# -------------------------------------------------------------------------

def plot_seasonal_indices(
    model_dir: Path,
    model: str,
    year: int,
    month: int,
    out_dir: Path,
) -> None:
    """
    Create a 2×3 panel of key seasonal index maps using earthkit.plots.
    Reads  indices/ensemble_statistics.nc  (written by compute script).
    """
    nc_path = model_dir / "indices" / "ensemble_statistics.nc"
    ds = _load_nc(nc_path)
    if ds is None:
        return

    available = [p for p in SEASONAL_PANELS if p[0] in ds]
    if not available:
        logging.warning("No matching index variables found in %s", nc_path)
        return

    ncols = 3
    nrows = (len(available) + ncols - 1) // ncols

    try:
        # ── earthkit-plots Figure ──────────────────────────────────────────
        fig = ekp.Figure(
            rows=nrows, columns=ncols,
            domain=ETHIOPIA_DOMAIN,
        )

        for idx, (var, title, cmap, vmin, vmax, unit) in enumerate(available):
            da = ds[var]
            row, col = divmod(idx, ncols)

            sub = fig.subplot(row=row, column=col)
            sub.coastlines()
            sub.gridlines()
            sub.plot(
                da,
                style=ekp.Style(colors=cmap, levels=10, vmin=vmin, vmax=vmax),
                title=f"{title}\n({unit})",
            )

        suptitle = _safe_title(model, year, month)
        fig.title(suptitle)
        out_path = out_dir / "seasonal_indices.png"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.save(str(out_path), dpi=150)
        logging.info("Saved seasonal index map: %s", out_path)

    except Exception as ekp_exc:
        logging.warning(
            "earthkit.plots render failed (%s); falling back to matplotlib.", ekp_exc
        )
        _plot_seasonal_matplotlib(ds, available, model, year, month, out_dir)
    finally:
        ds.close()


def _plot_seasonal_matplotlib(
    ds: xr.Dataset,
    panels: list,
    model: str,
    year: int,
    month: int,
    out_dir: Path,
) -> None:
    """Plain matplotlib fallback used when earthkit.plots raises."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        has_cartopy = True
    except ImportError:
        has_cartopy = False

    ncols, nrows = 3, (len(panels) + 2) // 3
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 4 * nrows),
        subplot_kw={"projection": ccrs.PlateCarree()} if has_cartopy else {},
        squeeze=False,
    )

    lats = ds["latitude"].values
    lons = ds["longitude"].values

    for idx, (var, title, cmap, vmin, vmax, unit) in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        arr = ds[var].values
        im = ax.pcolormesh(lons, lats, arr, cmap=cmap, vmin=vmin, vmax=vmax)
        if has_cartopy:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            ax.set_extent(ETHIOPIA_DOMAIN, crs=ccrs.PlateCarree())
        plt.colorbar(im, ax=ax, fraction=0.03, label=unit)
        ax.set_title(f"{title}", fontsize=9)

    # Hide unused subplots
    for idx in range(len(panels), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(_safe_title(model, year, month), fontsize=11, y=1.01)
    fig.tight_layout()
    out_path = out_dir / "seasonal_indices.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved seasonal index map (matplotlib): %s", out_path)


# -------------------------------------------------------------------------
# Monthly weather panel
# -------------------------------------------------------------------------

def plot_monthly_summaries(
    model_dir: Path,
    model: str,
    year: int,
    month: int,
    out_dir: Path,
) -> None:
    """
    Create per-variable monthly time-series panels from
    indices/monthly_weather_summaries.nc (ensemble mean across members).
    """
    nc_path = model_dir / "indices" / "monthly_weather_summaries.nc"
    ds = _load_nc(nc_path)
    if ds is None:
        return

    # Compute ensemble mean across members (dim "number")
    if "number" in ds.dims:
        ds_mean = ds.mean(dim="number")
    else:
        ds_mean = ds

    available = [(v, t, c, lo, hi, u) for v, t, c, lo, hi, u in MONTHLY_PANELS if v in ds_mean]
    if not available:
        logging.warning("No monthly variables found in %s", nc_path)
        ds.close()
        return

    n_months = ds_mean.sizes.get("time", 0)
    ncols, nrows = 3, (len(available) + 2) // 3

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        has_cartopy = True
    except ImportError:
        has_cartopy = False

    lats = ds_mean["latitude"].values
    lons = ds_mean["longitude"].values
    times = ds_mean["time"].values

    # One figure per month (first 3 months shown as a 2-row panel each)
    months_to_show = min(n_months, 6)  # cap at 6 for readability

    for t_idx in range(months_to_show):
        month_label = str(times[t_idx])[:10]
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(6 * ncols, 4 * nrows),
            subplot_kw={"projection": ccrs.PlateCarree()} if has_cartopy else {},
            squeeze=False,
        )

        for idx, (var, title, cmap, vmin, vmax, unit) in enumerate(available):
            ax = axes[idx // ncols][idx % ncols]
            arr = ds_mean[var].isel(time=t_idx).values
            im = ax.pcolormesh(lons, lats, arr, cmap=cmap, vmin=vmin, vmax=vmax)
            if has_cartopy:
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, linewidth=0.3)
                ax.set_extent(ETHIOPIA_DOMAIN, crs=ccrs.PlateCarree())
            plt.colorbar(im, ax=ax, fraction=0.03, label=unit)
            ax.set_title(f"{title}", fontsize=9)

        for idx in range(len(available), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        suptitle = f"Monthly Weather Summary — {month_label}  |  {_safe_title(model, year, month)}"
        fig.suptitle(suptitle, fontsize=10, y=1.01)
        fig.tight_layout()
        out_path = out_dir / f"monthly_{month_label}.png"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logging.info("Saved monthly map (%s): %s", month_label, out_path)

    ds.close()


# -------------------------------------------------------------------------
# Ensemble spread map (reliability)
# -------------------------------------------------------------------------

def plot_ensemble_spread(
    model_dir: Path,
    model: str,
    year: int,
    month: int,
    out_dir: Path,
) -> None:
    """
    Map ensemble spread (std / mean coefficient of variation) for key indices.
    Helps identify high-uncertainty regions in the forecast.
    """
    nc_path = model_dir / "indices" / "ensemble_statistics.nc"
    ds = _load_nc(nc_path)
    if ds is None:
        return

    spread_pairs = [
        ("rainfall_total_mean",   "rainfall_total_std",   "Rainfall CV",      "Oranges"),
        ("et0_total_mean",        "et0_total_std",        "ET₀ CV",           "Purples"),
        ("advisory_score_mean",   "advisory_score_std",   "Advisory Score CV","Reds"),
    ]
    available = [
        (m, s, t, c) for m, s, t, c in spread_pairs
        if m in ds and s in ds
    ]
    if not available:
        ds.close()
        return

    ncols = min(3, len(available))
    fig, axes = plt.subplots(
        1, ncols, figsize=(6 * ncols, 4),
        squeeze=False,
    )
    lats = ds["latitude"].values
    lons = ds["longitude"].values

    for idx, (mean_var, std_var, title, cmap) in enumerate(available):
        ax = axes[0][idx]
        mean_arr = ds[mean_var].values
        std_arr  = ds[std_var].values
        with np.errstate(invalid="ignore", divide="ignore"):
            cv = np.where(np.abs(mean_arr) > 1e-6, std_arr / np.abs(mean_arr), np.nan)
        im = ax.pcolormesh(lons, lats, cv, cmap=cmap, vmin=0, vmax=0.5)
        plt.colorbar(im, ax=ax, fraction=0.03, label="CV (std/mean)")
        ax.set_title(title, fontsize=9)

    fig.suptitle(f"Ensemble Uncertainty  |  {_safe_title(model, year, month)}", fontsize=10)
    fig.tight_layout()
    out_path = out_dir / "ensemble_spread.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved ensemble spread map: %s", out_path)
    ds.close()


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise CDS seasonal agro-climate indices using earthkit.plots."
    )
    parser.add_argument("--year",  type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--day",   type=int, default=1)
    parser.add_argument(
        "--model", default="ecmwf",
        help="Model centre name, e.g. ecmwf, ukmo, ncep. Ignored if --all-models.",
    )
    parser.add_argument(
        "--all-models", action="store_true",
        help="Visualise all 9 models.",
    )
    parser.add_argument(
        "--monthly", action="store_true",
        help="Also produce monthly weather summary maps.",
    )
    parser.add_argument(
        "--spread", action="store_true",
        help="Also produce ensemble spread (CV) maps.",
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


ALL_MODELS = ["ecmwf", "ukmo", "ncep", "jma", "cmcc", "dwd", "meteo_france", "eccc", "bom"]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    global ETHIOPIA_DOMAIN
    ETHIOPIA_DOMAIN = _earthkit_domain(args.country)
    root = Path(args.root) if args.root else cds_root(args.country)
    models = ALL_MODELS if args.all_models else [args.model]

    for model in models:
        model_dir = build_model_dir(root, model, args.year, args.month, args.day)
        if not model_dir.exists():
            logging.warning("Model directory not found: %s", model_dir)
            continue

        out_dir = model_dir / "indices" / "plots"
        logging.info("Visualising %s → %s", model_dir.name, out_dir)

        plot_seasonal_indices(model_dir, model, args.year, args.month, out_dir)
        plot_ensemble_spread(model_dir, model, args.year, args.month, out_dir)

        if args.monthly:
            plot_monthly_summaries(model_dir, model, args.year, args.month, out_dir)

    logging.info("Done.")


if __name__ == "__main__":
    main()
