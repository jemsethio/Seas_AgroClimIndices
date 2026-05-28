#!/usr/bin/env python3
# publication/figures.py
"""
Publication-ready agro-climate forecast maps for Ethiopia.
ECMWF System-51  |  51-member ensemble  |  Kiremt season focus (JJAS)

Design principles
-----------------
- Data clipped to Ethiopia; neighbours grey, ocean blue
- contourf: smooth interpolated fill — no pixelated cell rows
- Gridlines invisible (labels only at axes edges)
- No city markers on maps
- Expert colour schemes anchored to agronomic / livestock / WMO thresholds
- Categorical vertical colorbars with meaningful bin labels (policy-ready)
- Onset/cessation: calendar-date labels, weekly tick steps
- Monthly figure: all 7 weather variables for June (1-month lead)

Figures
-------
  Fig 1  Core Water & Crop Indices            (6 panels, 2×3)
  Fig 2  Growing Season Timing & Extremes     (6 panels, 2×3)
  Fig 3  Livestock & Thermal Stress           (6 panels, 2×3)
  Fig 4  Composite Advisory & Risk Scores     (5 panels, 2×3)
  Fig 5  Ensemble Uncertainty P90–P10         (6 panels, 2×3)
  Fig 6  June Forecast — all weather vars     (2×4 panels)
  Fig 7  City Ensemble Box-plots              (4 panels)

Usage
-----
  uv run python -m cds_agroclimate_pipeline.publication.figures --year 2026 --month 5 --model ecmwf

Author: Jemal Ahmed  <J.Ahmed@cgiar.org>
"""
from __future__ import annotations

import argparse
import datetime
import warnings
from pathlib import Path
from typing import Optional

from cds_agroclimate_pipeline.paths import (
    DEFAULT_COUNTRY,
    boundary_path,
    cds_root,
    configure_matplotlib_cache,
    country_bbox_lonlat,
)

configure_matplotlib_cache()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import ShapelyFeature
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# ── Typography & style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":          "DejaVu Sans",
    "font.size":             9,
    "axes.titlesize":       11,
    "axes.labelsize":        8.5,
    "xtick.labelsize":       6.5,
    "ytick.labelsize":       6.5,
    "figure.dpi":           130,
    "figure.facecolor":     "#FAFAF8",
    "axes.facecolor":       "#FAFAF8",
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.12,
    "savefig.facecolor":    "#FAFAF8",
    "axes.linewidth":        0.7,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
})

# ── Country extent: [lon_min, lon_max, lat_min, lat_max] ───────────────────
EXTENT = country_bbox_lonlat(DEFAULT_COUNTRY)

# ─────────────────────────────────────────────────────────────────────────────
# Expert colour schemes (domain-science anchored)
# ─────────────────────────────────────────────────────────────────────────────

def _scheme(colors, bounds):
    """Create (cmap, norm) from a list of colors and boundary values."""
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)
    ticks = [(bounds[i] + bounds[i + 1]) / 2 for i in range(len(bounds) - 1)]
    return cmap, norm, ticks


# ── Categorical risk scale (0–1, higher = worse) ───────────────────────────
# Colours: IPC-phase inspired — green / amber / orange / red
_RISK = _scheme(
    ['#1a9641', '#fee391', '#ec7014', '#b30000'],
    [0.00, 0.25, 0.50, 0.75, 1.00],
)
RISK_CMAP, RISK_NORM, RISK_TICKS = _RISK
RISK_LABELS = ['Low\n(0–0.25)', 'Moderate\n(0.25–0.5)', 'High\n(0.5–0.75)', 'Critical\n(0.75–1)']

# ── Opportunity scale (0–1, higher = better): inverted risk ────────────────
_OPP = _scheme(
    ['#b30000', '#ec7014', '#fee391', '#1a9641'],
    [0.00, 0.25, 0.50, 0.75, 1.00],
)
OPP_CMAP, OPP_NORM, OPP_TICKS = _OPP
OPP_LABELS = ['Low\n(0–0.25)', 'Moderate\n(0.25–0.5)', 'High\n(0.5–0.75)', 'Very High\n(0.75–1)']

# ── Seasonal rainfall (mm): 7 WMO-inspired classes ─────────────────────────
_RAIN = _scheme(
    ['#a50026', '#f46d43', '#fee090', '#e0f3f8', '#74add1', '#4575b4', '#313695'],
    [0, 100, 250, 500, 750, 1000, 1500, 2000],
)
RAIN_CMAP, RAIN_NORM, RAIN_TICKS = _RAIN
RAIN_LABELS = ['<100', '100–250', '250–500', '500–750', '750–1000', '1000–1500', '>1500 mm']

# ── ET₀ total (mm): 5 arid-to-humid classes ────────────────────────────────
_ET0 = _scheme(
    ['#ffffb2', '#fecc5c', '#fd8d3c', '#e31a1c', '#800026'],
    [500, 700, 900, 1100, 1400, 1700],
)
ET0_CMAP, ET0_NORM, ET0_TICKS = _ET0
ET0_LABELS = ['500–700', '700–900', '900–1100', '1100–1400', '>1400 mm']

# ── Growing Degree Days (°C·d): 5 crop-phenology classes ───────────────────
_GDD = _scheme(
    ['#ffffcc', '#c2e699', '#78c679', '#31a354', '#006837'],
    [1000, 2000, 3000, 4000, 4500, 5500],
)
GDD_CMAP, GDD_NORM, GDD_TICKS = _GDD
GDD_LABELS = ['1000–2000', '2000–3000', '3000–4000', '4000–4500', '>4500 °C·d']

# ── Dry spell (days): agronomic stress thresholds ──────────────────────────
_DS = _scheme(
    ['#1a9641', '#a6d96a', '#ffffbf', '#fdae61', '#d7191c', '#7f0000'],
    [0, 10, 20, 30, 45, 90, 160],
)
DS_CMAP, DS_NORM, DS_TICKS = _DS
DS_LABELS = ['<10 d\n(normal)', '10–20 d', '20–30 d\n(stress)', '30–45 d', '45–90 d\n(severe)', '>90 d\n(extreme)']

# ── Wet spell (days): 4 classes ────────────────────────────────────────────
_WS = _scheme(
    ['#eff3ff', '#bdd7e7', '#6baed6', '#08519c'],
    [0, 10, 20, 45, 110],
)
WS_CMAP, WS_NORM, WS_TICKS = _WS
WS_LABELS = ['<10 d', '10–20 d', '20–45 d', '>45 d']

# ── Growing period / LGP (days): crop-season classes ──────────────────────
_LGP = _scheme(
    ['#ffffcc', '#d9f0a3', '#78c679', '#238443', '#005a32'],
    [0, 30, 60, 90, 150, 210],
)
LGP_CMAP, LGP_NORM, LGP_TICKS = _LGP
LGP_LABELS = ['<30 d\n(short)', '30–60 d', '60–90 d', '90–150 d', '>150 d\n(long)']

# ── Onset (forecast day from init): kiremt phenology classes ───────────────
# May 1 init: day 31=Jun 1, day 61=Jul 1, day 92=Aug 1, day 122=Sep 1
_ON = _scheme(
    ['#005824', '#31a354', '#fee08b', '#f46d43', '#a50026'],
    [0, 31, 61, 92, 122, 170],
)
ONSET_CMAP, ONSET_NORM, ONSET_TICKS = _ON
ONSET_LABELS = ['May\n(very early)', 'Jun\n(early/ideal)', 'Jul\n(normal)', 'Aug\n(late)', 'Sep+\n(very late)']

# ── Cessation (forecast day from init): end-of-season classes ──────────────
# Longer season (later cessation) = better
_CE = _scheme(
    ['#a50026', '#f46d43', '#fee08b', '#31a354', '#005824'],
    [50, 153, 170, 185, 200, 210],
)
CESS_CMAP, CESS_NORM, CESS_TICKS = _CE
CESS_LABELS = ['<Oct\n(very short)', 'Oct 1–20\n(short)', 'Oct 20–Nov 5\n(normal)', 'Nov 5–20\n(good)', '>Nov 20\n(long)']

# ── THI (Temperature-Humidity Index): WMO livestock stress thresholds ──────
_THI = _scheme(
    ['#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c'],
    [55, 68, 72, 78, 84, 92],
)
THI_CMAP, THI_NORM, THI_TICKS = _THI
THI_LABELS = ['Normal\n(<68)', 'Mild\n(68–72)', 'Moderate\n(72–78)', 'Severe\n(78–84)', 'Emergency\n(>84)']

# ── THI heat stress days (0–215 d): 5 classes ──────────────────────────────
_THIDAYS = _scheme(
    ['#1a9641', '#a6d96a', '#fdae61', '#d7191c', '#7f0000'],
    [0, 30, 90, 150, 180, 215],
)
THIDAYS_CMAP, THIDAYS_NORM, THIDAYS_TICKS = _THIDAYS
THIDAYS_LABELS = ['<30 d', '30–90 d', '90–150 d', '150–180 d', '>180 d']

# ── False-start risk (fraction 0–1): 4 classes ─────────────────────────────
_FS = _scheme(
    ['#1a9641', '#fee391', '#ec7014', '#b30000'],
    [0.0, 0.2, 0.4, 0.6, 1.0],
)
FS_CMAP, FS_NORM, FS_TICKS = _FS
FS_LABELS = ['Low\n(<0.2)', 'Moderate\n(0.2–0.4)', 'High\n(0.4–0.6)', 'Very High\n(>0.6)']

# ── Extreme rain days: 5 classes ───────────────────────────────────────────
_XRD = _scheme(
    ['#f7fbff', '#c6dbef', '#6baed6', '#2171b5', '#08306b'],
    [0, 1, 3, 7, 14, 22],
)
XRD_CMAP, XRD_NORM, XRD_TICKS = _XRD
XRD_LABELS = ['<1 d', '1–3 d', '3–7 d', '7–14 d', '>14 d']


# ── Style registry: variable stem → (cmap, norm, ticks, labels, cbar_title)
def get_coloring(stem: str) -> tuple:
    """Return (cmap, norm, tick_positions, tick_labels, colorbar_title)."""
    M = {
        "rainfall_total":          (RAIN_CMAP,    RAIN_NORM,    RAIN_TICKS,    RAIN_LABELS,    "Seasonal Total Rainfall (mm)"),
        "et0_total":               (ET0_CMAP,     ET0_NORM,     ET0_TICKS,     ET0_LABELS,     "Seasonal Total ET₀ (mm)"),
        "water_stress_index_mean": (RISK_CMAP,    RISK_NORM,    RISK_TICKS,    RISK_LABELS,    "Water Stress Index (0–1)"),
        "growing_degree_days":     (GDD_CMAP,     GDD_NORM,     GDD_TICKS,     GDD_LABELS,     "Growing Degree Days (°C·d)"),
        "dry_spell_max_days":      (DS_CMAP,      DS_NORM,      DS_TICKS,      DS_LABELS,      "Longest Dry Spell (days)"),
        "wet_spell_max_days":      (WS_CMAP,      WS_NORM,      WS_TICKS,      WS_LABELS,      "Longest Wet Spell (days)"),
        "onset_day_of_forecast":   (ONSET_CMAP,   ONSET_NORM,   ONSET_TICKS,   ONSET_LABELS,   "Onset Month (kiremt)"),
        "cessation_day_of_forecast":(CESS_CMAP,   CESS_NORM,    CESS_TICKS,    CESS_LABELS,    "Cessation Period"),
        "length_of_growing_period_days":(LGP_CMAP, LGP_NORM,   LGP_TICKS,    LGP_LABELS,     "Growing Period Length (days)"),
        "false_start_fraction":    (FS_CMAP,      FS_NORM,      FS_TICKS,      FS_LABELS,      "False Start Risk"),
        "extreme_rain_days_ge_20mm":(XRD_CMAP,   XRD_NORM,    XRD_TICKS,    XRD_LABELS,    "Heavy Rain Days ≥20 mm"),
        "extreme_rain_days_ge_50mm":(XRD_CMAP,   XRD_NORM,    XRD_TICKS,    XRD_LABELS,    "Extreme Rain Days ≥50 mm"),
        "thi_max":                 (THI_CMAP,     THI_NORM,     THI_TICKS,     THI_LABELS,     "Peak THI (livestock stress)"),
        "thi_mean":                (THI_CMAP,     THI_NORM,     THI_TICKS,     THI_LABELS,     "Mean THI (livestock stress)"),
        "thi_heat_stress_days":    (THIDAYS_CMAP, THIDAYS_NORM, THIDAYS_TICKS, THIDAYS_LABELS, "THI Heat Stress Days"),
        "pasture_drought_score":   (RISK_CMAP,    RISK_NORM,    RISK_TICKS,    RISK_LABELS,    "Pasture Drought Score"),
        "feed_water_stress_score": (RISK_CMAP,    RISK_NORM,    RISK_TICKS,    RISK_LABELS,    "Feed & Water Stress Score"),
        "vector_suitability_score":(RISK_CMAP,    RISK_NORM,    RISK_TICKS,    RISK_LABELS,    "Disease Vector Suitability"),
        "advisory_score":          (RISK_CMAP,    RISK_NORM,    RISK_TICKS,    RISK_LABELS,    "Composite Advisory Score"),
        "agro_pastoral_drought_score":(RISK_CMAP, RISK_NORM,   RISK_TICKS,   RISK_LABELS,    "Agro-Pastoral Drought Risk"),
        "crop_livelihood_stress_score":(RISK_CMAP,RISK_NORM,   RISK_TICKS,   RISK_LABELS,    "Crop–Livelihood Stress Score"),
        "resilience_score":        (OPP_CMAP,     OPP_NORM,     OPP_TICKS,     OPP_LABELS,     "Integrated Resilience Score"),
        "surface_water_stress_score":(RISK_CMAP,  RISK_NORM,   RISK_TICKS,   RISK_LABELS,    "Surface Water Stress Score"),
    }
    return M.get(stem, (plt.cm.viridis, None, None, None, stem.replace("_", " ").title()))


# ─────────────────────────────────────────────────────────────────────────────
# Geometry cache (loaded once)
# ─────────────────────────────────────────────────────────────────────────────

_ETHIOPIA_GEOM  = None
_NEIGHBOR_GEOMS = None
_ETH_MASK_CACHE: dict = {}


def _load_geometries() -> None:
    global _ETHIOPIA_GEOM, _NEIGHBOR_GEOMS
    if _ETHIOPIA_GEOM is not None:
        return
    shp = shapereader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries"
    )
    lon_min, lon_max, lat_min, lat_max = EXTENT
    nbrs: list = []
    for rec in shapereader.Reader(shp).records():
        name = rec.attributes.get("ADMIN", "")
        geom = rec.geometry
        if name == "Ethiopia":
            _ETHIOPIA_GEOM = geom
        else:
            bx, by, bX, bY = geom.bounds
            if bX >= lon_min - 1 and bx <= lon_max + 1 \
               and bY >= lat_min - 1 and by <= lat_max + 1:
                nbrs.append(geom)
    _NEIGHBOR_GEOMS = nbrs


def _make_mask(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    key = (lats.size, lons.size)
    if key in _ETH_MASK_CACHE:
        return _ETH_MASK_CACHE[key]
    from shapely import contains_xy
    _load_geometries()
    LONS_2d, LATS_2d = np.meshgrid(lons, lats)
    inside = contains_xy(
        _ETHIOPIA_GEOM,
        LONS_2d.ravel().astype(float),
        LATS_2d.ravel().astype(float),
    ).reshape(LATS_2d.shape)
    _ETH_MASK_CACHE[key] = inside
    return inside


# ─────────────────────────────────────────────────────────────────────────────
# Map infrastructure
# ─────────────────────────────────────────────────────────────────────────────

def _add_map_features(ax) -> None:
    """
    Country borders, rivers, ocean fill, and edge-only lat/lon labels.
    No gridlines are drawn ON the coloured surface.
    """
    proj = ccrs.PlateCarree()
    ax.set_extent(EXTENT, crs=proj)

    _load_geometries()

    ax.add_feature(cfeature.OCEAN, facecolor="#d0e8f2", zorder=3)

    if _NEIGHBOR_GEOMS:
        ax.add_feature(
            ShapelyFeature(_NEIGHBOR_GEOMS, proj,
                           facecolor="#e8e4d9", edgecolor="#bbbbbb", linewidth=0.5),
            zorder=3,
        )
    if _ETHIOPIA_GEOM:
        ax.add_feature(
            ShapelyFeature([_ETHIOPIA_GEOM], proj,
                           facecolor="none", edgecolor="#111111", linewidth=1.4),
            zorder=4,
        )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "cultural", "admin_1_states_provinces", "10m",
            edgecolor="#aaaaaa", facecolor="none", linewidth=0.22),
        zorder=4,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "rivers_lake_centerlines", "10m",
            edgecolor="#6ea6cc", facecolor="none", linewidth=0.35),
        zorder=4,
    )
    # Invisible gridlines — labels only at edges
    gl = ax.gridlines(crs=proj, draw_labels=True,
                      linewidth=0, color="none",
                      x_inline=False, y_inline=False)
    gl.top_labels   = False
    gl.right_labels = False
    gl.xformatter   = LONGITUDE_FORMATTER
    gl.yformatter   = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 5.5, "color": "#444"}
    gl.ylabel_style = {"size": 5.5, "color": "#444"}
    gl.xlocator     = mticker.FixedLocator([34, 36, 38, 40, 42, 44, 46, 48])
    gl.ylocator     = mticker.FixedLocator([4, 6, 8, 10, 12, 14])


def _panel_letter(ax, letter: str) -> None:
    ax.text(0.03, 0.97, f"({letter})",
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="top", ha="left", zorder=10,
            bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", pad=1.0))


# ── City reference markers ───────────────────────────────────────────────────
# (name, lon, lat, is_capital, text_dx, text_dy, ha)
CITY_MARKERS = [
    ("Addis Ababa",  38.74,  9.03, True,   0.30,  0.28, "left"),
    ("Dire Dawa",    41.86,  9.60, False,  0.28, -0.30, "left"),
    ("Mekelle",      39.48, 13.50, False,  0.28,  0.22, "left"),
    ("Gondar",       37.47, 12.60, False, -0.30,  0.22, "right"),
    ("Bahir Dar",    37.39, 11.59, False, -0.30, -0.22, "right"),
    ("Jimma",        36.83,  7.67, False, -0.30,  0.22, "right"),
    ("Hawassa",      38.48,  7.05, False,  0.28,  0.22, "left"),
    ("Jijiga",       42.79,  9.35, False, -0.28,  0.25, "right"),
    ("Gambela",      34.58,  8.25, False,  0.28,  0.22, "left"),
]


def _add_city_markers(ax, show_labels: bool = True, fontsize: float = 5.2) -> None:
    """Overlay major Ethiopian city dots and labels on a PlateCarree axis."""
    proj = ccrs.PlateCarree()
    for name, lon, lat, is_cap, dx, dy, ha in CITY_MARKERS:
        ax.plot(lon, lat, transform=proj, zorder=9,
                marker="*" if is_cap else "o",
                markersize=7.0 if is_cap else 3.8,
                color="#FFD700" if is_cap else "#FFFFFF",
                markeredgecolor="#111", markeredgewidth=0.9 if is_cap else 0.6,
                clip_on=True)
        if show_labels:
            ax.text(lon + dx, lat + dy, name,
                    transform=proj, fontsize=fontsize,
                    ha=ha, va="center", zorder=10,
                    fontweight="bold" if is_cap else "normal",
                    color="#111",
                    bbox=dict(facecolor="white", alpha=0.72,
                              edgecolor="none", pad=0.7,
                              boxstyle="round,pad=0.15"))


def _panel_stats(ax, data: np.ndarray, mask: np.ndarray | None = None) -> None:
    """Add a compact statistics badge (median + IQR) inside bottom-left of panel."""
    vals = data[mask & np.isfinite(data)] if mask is not None else data[np.isfinite(data)]
    if len(vals) < 5:
        return
    med = float(np.nanmedian(vals))
    p25, p75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
    fmt = ".2f" if abs(med) < 5 else (".1f" if abs(med) < 50 else ".0f")
    txt = f"ETH median: {med:{fmt}}\nIQR: {p25:{fmt}} – {p75:{fmt}}"
    ax.text(0.02, 0.03, txt,
            transform=ax.transAxes, fontsize=4.8, va="bottom", ha="left",
            zorder=11, color="#222",
            bbox=dict(facecolor="white", alpha=0.88, edgecolor="#bbb",
                      linewidth=0.4, pad=1.8, boxstyle="round,pad=0.28"))


def _north_arrow(ax, x: float = 0.935, y: float = 0.06, length: float = 0.065) -> None:
    """Add a minimalist N-arrow to a map axis (axes-fraction coordinates)."""
    ax.annotate("", xy=(x, y + length), xytext=(x, y),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#333",
                                lw=1.5, mutation_scale=9),
                zorder=12)
    ax.text(x, y + length + 0.022, "N",
            transform=ax.transAxes, fontsize=6.5, ha="center", va="bottom",
            fontweight="bold", color="#333", zorder=12)


def _source_note(fig, model: str, init_date: str) -> None:
    """Add a bottom-edge attribution line to a figure."""
    fig.text(0.01, 0.005,
             f"Source: ECMWF CDS Seasonal Forecast | System 51 | 51-member ensemble | "
             f"Init: {init_date} | EMI/ICPC El Niño Advisory 2026",
             ha="left", va="bottom", fontsize=5.5, color="#777",
             fontstyle="italic")


def draw_panel(
    ax, fig,
    lats: np.ndarray, lons: np.ndarray,
    data: np.ndarray,
    stem: str,
    title: str,
    letter: Optional[str] = None,
) -> None:
    """
    Render one Ethiopia agro-climate index panel.

    Key rendering strategy
    ----------------------
    • Feed the FULL unmasked grid to contourf so it fills edge-to-edge
      with no transparency holes at the Ethiopia border.
    • Sort lats ascending (grid is stored 15→3°N; contourf needs ascending).
    • Neighbour grey fills are drawn at zorder=3 on top of the colour field,
      clipping any colour that spills outside Ethiopia cleanly.
    • Colorbar: wider fraction, drawedges, larger text — policy-readable.
    """
    proj = ccrs.PlateCarree()
    cmap, norm, tick_pos, tick_lbs, cbar_title = get_coloring(stem)

    # ── 1. Sort lats ascending; use FULL data (no NaN mask) ──────────────
    si = np.argsort(lats)
    LONS_2d, LATS_2d = np.meshgrid(lons, lats[si])
    d_sorted = data[si, :]

    im = ax.pcolormesh(
        LONS_2d, LATS_2d, d_sorted,
        cmap=cmap, norm=norm,
        transform=proj, zorder=2,
        shading='nearest',
    )

    # ── 2. Map overlays, city markers, north arrow ───────────────────────
    _add_map_features(ax)
    _add_city_markers(ax, show_labels=True, fontsize=5.0)
    _north_arrow(ax)

    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=5,
                 color="#1a1a2e")
    if letter:
        _panel_letter(ax, letter)

    # ── 3. Colorbar — wider, edges between bins, readable labels ─────────
    cb = fig.colorbar(im, ax=ax,
                      orientation="vertical",
                      fraction=0.048, pad=0.025,
                      shrink=0.88, aspect=18,
                      extend='both',
                      drawedges=True)
    cb.dividers.set_linewidth(0.4)
    cb.dividers.set_color("#555")
    if tick_pos is not None:
        cb.set_ticks(tick_pos)
        cb.set_ticklabels(tick_lbs, fontsize=6.5)
    cb.ax.tick_params(labelsize=6.5, length=0)
    cb.ax.set_title(cbar_title, fontsize=6.2, pad=5,
                    ha="center", va="bottom", wrap=True,
                    fontweight="semibold")
    cb.outline.set_linewidth(0.6)

    # ── 4. Per-panel statistics badge ─────────────────────────────────────
    _panel_stats(ax, data)


# ─────────────────────────────────────────────────────────────────────────────
# Panel group definitions  (stem, display_title)
# ─────────────────────────────────────────────────────────────────────────────

FIG1_PANELS = [
    ("rainfall_total",          "Seasonal Total Rainfall"),
    ("et0_total",               "Seasonal Reference ET₀"),
    ("water_stress_index_mean", "Mean Water Stress Index"),
    ("growing_degree_days",     "Growing Degree Days"),
    ("dry_spell_max_days",      "Longest Dry Spell"),
    ("wet_spell_max_days",      "Longest Wet Spell"),
]

FIG2_PANELS = [
    ("onset_day_of_forecast",          "Rainy Season Onset (kiremt)"),
    ("cessation_day_of_forecast",      "Season Cessation"),
    ("length_of_growing_period_days",  "Growing Period Length"),
    ("false_start_fraction",           "False Start Risk"),
    ("extreme_rain_days_ge_20mm",      "Heavy Rain Days  (≥ 20 mm)"),
    ("extreme_rain_days_ge_50mm",      "Extreme Rain Days  (≥ 50 mm)"),
]

FIG3_PANELS = [
    ("thi_max",                "Peak THI  — Livestock Heat Stress"),
    ("thi_mean",               "Seasonal Mean THI"),
    ("thi_heat_stress_days",   "THI Heat Stress Days"),
    ("pasture_drought_score",  "Pasture Drought Score"),
    ("feed_water_stress_score","Feed & Water Stress Score"),
    ("vector_suitability_score","Disease Vector Suitability"),
]

FIG4_PANELS = [
    ("advisory_score",               "Composite Advisory Score"),
    ("agro_pastoral_drought_score",  "Agro-Pastoral Drought Risk"),
    ("crop_livelihood_stress_score", "Crop–Livelihood Stress"),
    ("resilience_score",             "Integrated Resilience"),
    ("surface_water_stress_score",   "Surface Water Stress"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Generic multi-panel factory (2 rows × 3 cols, landscape)
# ─────────────────────────────────────────────────────────────────────────────

def _make_figure(ds, panels, fig_title, out_path, lats, lons,
                 nrows=2, ncols=3, init_date="", model=""):
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6.5 * ncols, 5.5 * nrows),
        subplot_kw={"projection": proj},
        gridspec_kw={"hspace": 0.38, "wspace": 0.32},
    )
    fig.patch.set_facecolor("#FAFAF8")
    ax_flat = np.asarray(axes).ravel()
    letters  = "abcdefghijklmnopqrstuvwxyz"
    n_drawn  = 0

    for stem, title in panels:
        var = f"{stem}_mean"
        if var not in ds:
            continue
        data = ds[var].values.copy()
        if not np.any(np.isfinite(data)):
            continue
        if "onset" in stem or "cessation" in stem:
            data = np.where(data < 0, np.nan, data)
        if n_drawn >= len(ax_flat):
            break
        draw_panel(ax_flat[n_drawn], fig, lats, lons, data, stem, title,
                   letter=letters[n_drawn])
        n_drawn += 1

    for i in range(n_drawn, len(ax_flat)):
        ax_flat[i].set_visible(False)

    # ── Styled suptitle with El Niño notice ───────────────────────────────
    fig.suptitle(fig_title, fontsize=12, fontweight="bold", y=1.015,
                 color="#1a1a2e")
    if init_date:
        _source_note(fig, model, init_date)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}  ({n_drawn} panels)")


def _suptitle(model, suffix, init_date):
    return (f"{model.upper()} Seasonal Forecast  ·  Ethiopia Kiremt 2026 (JJAS)  ·  "
            f"Init: {init_date}  ·  51-member Ensemble\n"
            f"⚠  EMI & ICPC El Niño Advisory — {suffix}")


def figure1_crop_water(ds, out_dir, init_date, model):
    _make_figure(ds, FIG1_PANELS,
                 _suptitle(model, "Core Water & Crop Indices", init_date),
                 out_dir / "Fig1_crop_water_indices.png",
                 ds.latitude.values, ds.longitude.values,
                 init_date=init_date, model=model)


def figure2_season_timing(ds, out_dir, init_date, model):
    _make_figure(ds, FIG2_PANELS,
                 _suptitle(model, "Growing Season Timing & Extreme Rainfall", init_date),
                 out_dir / "Fig2_season_timing.png",
                 ds.latitude.values, ds.longitude.values,
                 init_date=init_date, model=model)


def figure3_livestock(ds, out_dir, init_date, model):
    _make_figure(ds, FIG3_PANELS,
                 _suptitle(model, "Livestock & Thermal Stress Indices", init_date),
                 out_dir / "Fig3_livestock_thermal.png",
                 ds.latitude.values, ds.longitude.values,
                 init_date=init_date, model=model)


def figure4_advisory(ds, out_dir, init_date, model):
    _make_figure(ds, FIG4_PANELS,
                 _suptitle(model, "Composite Advisory & Risk Scores", init_date),
                 out_dir / "Fig4_advisory_scores.png",
                 ds.latitude.values, ds.longitude.values,
                 nrows=2, ncols=3, init_date=init_date, model=model)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Ensemble uncertainty  (P90 – P10)
# ─────────────────────────────────────────────────────────────────────────────

def figure5_uncertainty(ds, out_dir, init_date, model):
    # Use continuous colormaps for spread (not categorical bins)
    SPREAD = [
        ("rainfall_total",          "Rainfall P90–P10",         "PuBu",    0, 800),
        ("et0_total",               "ET₀ P90–P10",         "OrRd",    0, 400),
        ("water_stress_index_mean", "Water Stress P90–P10",     "Oranges", 0, 0.6),
        ("growing_degree_days",     "GDD P90–P10",         "YlOrBr",  0, 2000),
        ("dry_spell_max_days",      "Dry Spell P90–P10",        "Reds",    0, 100),
        ("advisory_score",          "Advisory Score P90–P10",   "Purples", 0, 0.5),
    ]
    lats = ds.latitude.values
    lons = ds.longitude.values
    mask = _make_mask(lats, lons)
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        2, 3, figsize=(18, 10),
        subplot_kw={"projection": proj},
        gridspec_kw={"hspace": 0.35, "wspace": 0.30},
    )
    ax_flat = axes.ravel()
    letters = "abcdef"
    n = 0
    for stem, title, cmap, vmin, vmax in SPREAD:
        p10v, p90v = f"{stem}_p10", f"{stem}_p90"
        if p10v not in ds or p90v not in ds:
            continue
        spread = np.where(mask, ds[p90v].values - ds[p10v].values, np.nan)
        si = np.argsort(lats)
        LONS_2d, LATS_2d = np.meshgrid(lons, lats[si])
        ax = ax_flat[n]
        im = ax.pcolormesh(LONS_2d, LATS_2d, spread[si, :],
                           cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=proj, zorder=2,
                           shading='nearest')
        _add_map_features(ax)
        _add_city_markers(ax, show_labels=True, fontsize=4.8)
        _north_arrow(ax)
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=5,
                     color="#1a1a2e")
        _panel_letter(ax, letters[n])
        cb = fig.colorbar(im, ax=ax, orientation="vertical",
                          fraction=0.038, pad=0.022, shrink=0.92, aspect=22,
                          extend='max')
        cb.ax.tick_params(labelsize=6.5)
        cb.outline.set_linewidth(0.5)
        _panel_stats(ax, spread[si, :])
        n += 1
    for i in range(n, 6):
        ax_flat[i].set_visible(False)
    fig.suptitle(
        _suptitle(model, "Ensemble Uncertainty — P90 minus P10 Spread", init_date),
        fontsize=11, fontweight="bold", y=1.01)
    out_path = out_dir / "Fig5_ensemble_uncertainty.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}  ({n} panels)")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: June (1-month lead) — all weather variables
# ─────────────────────────────────────────────────────────────────────────────

JUNE_VARS = [
    # (varname, title, cmap, bins, labels, cbar_title)
    ("precipitation",
     "June Precipitation",
     *_scheme(['#fff7bc','#fec44f','#d95f0e','#7f2704'][::-1]+['#c6dbef','#6baed6','#2171b5'],
              [0,20,50,100,150,200,300,500]),
     "Precipitation (mm/month)"),
    ("temperature_2m_max",
     "June Maximum Temperature",
     *_scheme(['#4575b4','#91bfdb','#fee090','#fc8d59','#d73027'],
              [22,26,30,34,38,42]),
     "Tmax (°C)"),
    ("temperature_2m_min",
     "June Minimum Temperature",
     *_scheme(['#313695','#4575b4','#74add1','#abd9e9','#e0f3f8'],
              [10,14,18,22,26,30]),
     "Tmin (°C)"),
    ("temperature_2m_mean",
     "June Mean Temperature",
     *_scheme(['#4575b4','#91bfdb','#fee090','#fc8d59','#d73027'],
              [14,18,22,26,30,36]),
     "Tmean (°C)"),
    ("relative_humidity_2m",
     "June Relative Humidity",
     *_scheme(['#ffffd9','#c7e9b4','#7fcdbb','#41b6c4','#225ea8'],
              [30,50,65,75,85,100]),
     "Relative Humidity (%)"),
    ("shortwave_radiation",
     "June Solar Radiation",
     *_scheme(['#fff7bc','#fee391','#fec44f','#fe9929','#cc4c02'],
              [300,400,500,600,700,800]),
     "Solar Radiation (MJ m⁻² month⁻¹)"),
    ("wind_speed_10m",
     "June Mean Wind Speed",
     *_scheme(['#f7fcfd','#ccece6','#99d8c9','#66c2a4','#2ca25f','#006d2c'],
              [0,1,2,3,4,5,7]),
     "Wind Speed (m s⁻¹)"),
]


def figure6_june(ds_mon, out_dir, init_date, model):
    import pandas as pd
    dm = ds_mon.mean(dim="number") if "number" in ds_mon.dims else ds_mon
    june_idx = next(
        (i for i, t in enumerate(dm.time.values) if pd.Timestamp(t).month == 6),
        None,
    )
    if june_idx is None:
        print("  [skip] Fig 6: June not found")
        return

    lats = dm.latitude.values
    lons = dm.longitude.values
    mask = _make_mask(lats, lons)
    proj = ccrs.PlateCarree()

    n_panels = sum(1 for v, *_ in JUNE_VARS if v in dm)
    ncols = 4
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6.0 * ncols, 5.0 * nrows),
        subplot_kw={"projection": proj},
        gridspec_kw={"hspace": 0.35, "wspace": 0.30},
    )
    ax_flat = np.asarray(axes).ravel()
    letters = "abcdefgh"
    n = 0

    for var, title, cmap, norm, ticks, cbar_title in JUNE_VARS:
        if var not in dm:
            continue
        data = dm[var].isel(time=june_idx).values.copy()
        data_eth = np.where(mask, data, np.nan)
        si = np.argsort(lats)
        LONS_2d, LATS_2d = np.meshgrid(lons, lats[si])
        ax = ax_flat[n]
        im = ax.pcolormesh(LONS_2d, LATS_2d, data_eth[si, :],
                           cmap=cmap, norm=norm,
                           transform=proj, zorder=2,
                           shading='nearest')
        _add_map_features(ax)
        _add_city_markers(ax, show_labels=True, fontsize=4.8)
        _north_arrow(ax)
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=5,
                     color="#1a1a2e")
        _panel_letter(ax, letters[n])
        cb = fig.colorbar(im, ax=ax, orientation="vertical",
                          fraction=0.038, pad=0.022, shrink=0.92, aspect=22,
                          extend='both')
        _panel_stats(ax, data_eth[si, :])
        if ticks is not None:
            cb.set_ticks(ticks)
            # Format numeric tick labels from the boundary midpoints
            bounds = norm.boundaries
            tick_labels = []
            for i, b in enumerate(ticks):
                lo, hi = bounds[i], bounds[i + 1]
                tick_labels.append(f"{lo:.0f}–{hi:.0f}")
            cb.set_ticklabels(tick_labels, fontsize=5.8)
        cb.ax.set_title(cbar_title, fontsize=5.5, pad=4, ha="center")
        cb.outline.set_linewidth(0.5)
        n += 1

    for i in range(n, len(ax_flat)):
        ax_flat[i].set_visible(False)

    fig.suptitle(
        _suptitle(model,
                  "June Forecast — 1-Month Lead Time (Kiremt Onset Month)", init_date),
        fontsize=11, fontweight="bold", y=1.01)
    out_path = out_dir / "Fig6_june_weather_forecast.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}  ({n} panels)")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: City ensemble box-plots
# ─────────────────────────────────────────────────────────────────────────────

CITIES = [
    ("Addis Ababa", 38.74,  9.03), ("Dire Dawa",  41.86,  9.60),
    ("Mekelle",     39.48, 13.50), ("Gondar",     37.47, 12.60),
    ("Bahir Dar",   37.39, 11.59), ("Jimma",      36.83,  7.67),
    ("Hawassa",     38.48,  7.05), ("Jijiga",     42.79,  9.35),
    ("Gambela",     34.58,  8.25),
]

BOX_STYLES = [
    ("rainfall_total",     "Total Rainfall\n(mm)",        "Blues"),
    ("et0_total",          "Total ET₀\n(mm)",        "YlOrRd"),
    ("growing_degree_days","Growing Degree Days\n(°C·d)", "YlOrBr"),
    ("dry_spell_max_days", "Max Dry Spell\n(days)",        "Reds"),
]

def figure7_city_boxplots(ds_mem, out_dir, init_date, model):
    lats_g, lons_g = ds_mem.latitude.values, ds_mem.longitude.values

    def nearest(lat_q, lon_q):
        d = (lats_g[:, None] - lat_q) ** 2 + (lons_g[None, :] - lon_q) ** 2
        return np.unravel_index(d.argmin(), d.shape)

    available = [(v, l, c) for v, l, c in BOX_STYLES if v in ds_mem]
    if not available:
        print("  [skip] Fig 7: no per-member variables")
        return

    city_colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, len(CITIES)))

    fig, axes = plt.subplots(
        1, len(available),
        figsize=(4.0 * len(available), 5.0),
        gridspec_kw={"wspace": 0.42},
    )
    if len(available) == 1:
        axes = [axes]

    for col_idx, (var, label, cmap_name) in enumerate(available):
        ax  = axes[col_idx]
        data = ds_mem[var].values   # (n_members, lat, lon)

        box_data, colors = [], []
        for i, (_name, lon, lat) in enumerate(CITIES):
            ri, ci = nearest(lat, lon)
            vals = data[:, ri, ci]
            box_data.append(vals[np.isfinite(vals)])
            colors.append(city_colors[i])

        bp = ax.boxplot(
            box_data, vert=True, patch_artist=True, widths=0.6, notch=False,
            medianprops=dict(color="black", linewidth=2.0),
            whiskerprops=dict(linewidth=0.9, linestyle="-"),
            capprops=dict(linewidth=0.9),
            flierprops=dict(marker=".", ms=3, alpha=0.35),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.80)

        ax.set_xticks(range(1, len(CITIES) + 1))
        ax.set_xticklabels(
            [c[0].replace(" ", "\n") for c in CITIES], fontsize=6.5
        )
        ax.set_ylabel(label, fontsize=8.5)
        ax.set_title(label.split("\n")[0], fontsize=9, fontweight="semibold")
        ax.yaxis.grid(True, linewidth=0.5, alpha=0.65, color="#999")
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        _panel_letter(ax, "abcd"[col_idx])

    fig.suptitle(
        f"Ensemble Distribution at Ethiopian Cities  |  "
        f"Kiremt Season  |  Init: {init_date}  |  51 members",
        fontsize=10, fontweight="bold",
    )
    out_path = out_dir / "Fig7_city_ensemble_boxplots.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8: Location Advisory Choropleth
# ─────────────────────────────────────────────────────────────────────────────
#
#  Ethiopia's 11 regional states each receive a 4-tier advisory:
#    ● Normal  — favourable / no action
#    ▲ Watch   — monitor closely
#    ■ Alert   — targeted intervention
#    ★ Emergency — immediate response
#
#  Four panels (one per advisory theme):
#    (a) Crop & Water Advisory  — advisory_score
#    (b) Season Onset Advisory  — onset_day_of_forecast (kiremt timing)
#    (c) Livestock Heat Stress  — thi_max (WMO thresholds)
#    (d) Resilience Outlook     — resilience_score (inverted: higher=better)
#
#  A single shared legend at the bottom explains the tier colours.
# ─────────────────────────────────────────────────────────────────────────────

# Advisory tier design
_ADV_COLORS  = ['#1a9641', '#fee391', '#ec7014', '#b30000']
_ADV_LABELS  = ['Normal', 'Watch', 'Alert', 'Emergency']
_ADV_DESCS   = [
    'Favourable conditions',
    'Monitor closely',
    'Targeted intervention',
    'Immediate response',
]
_ADV_SYMBOLS = ['●', '▲', '■', '★']

# GADM ADM-3 (woreda) path — cached locally on first run
_GADM_ADM3_PATH = boundary_path(DEFAULT_COUNTRY, 3)
_ADM3_GDF       = None


def _load_admin2():            # kept as _load_admin2 for API compatibility
    """
    Load Ethiopia's 690 woredas from GADM 4.1 ADM-3.
    Falls back to ADM-2 zones if ADM-3 is unavailable.
    """
    global _ADM3_GDF
    if _ADM3_GDF is not None:
        return _ADM3_GDF
    try:
        import geopandas as gpd
    except ImportError:
        print("  [warn] geopandas not available — Fig 8 needs it"); return None

    if not _GADM_ADM3_PATH.exists():
        print("  Downloading GADM 4.1 ETH ADM-3 …", end=" ", flush=True)
        gdf = gpd.read_file(
            "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_ETH_shp.zip",
            layer="gadm41_ETH_3",
        )
        _GADM_ADM3_PATH.parent.mkdir(parents=True, exist_ok=True)
        gdf[["NAME_1", "NAME_2", "NAME_3", "geometry"]].to_file(
            _GADM_ADM3_PATH, driver="GPKG")
        print("done")
    else:
        import geopandas as gpd
        gdf = gpd.read_file(_GADM_ADM3_PATH)

    _ADM3_GDF = gdf
    return gdf


def _regional_mean(geom, lats, lons, data) -> float:
    """Spatial mean of grid-cell values whose centroids lie inside geom."""
    from shapely import contains_xy
    LONS_2d, LATS_2d = np.meshgrid(lons, lats)
    inside = contains_xy(geom,
                         LONS_2d.ravel().astype(float),
                         LATS_2d.ravel().astype(float))
    vals = data.ravel()[inside]
    vals = vals[np.isfinite(vals)]
    return float(np.nanmean(vals)) if len(vals) > 0 else np.nan


def _doy_label(doy: float, init_month: int = 5, init_year: int = 2026) -> str:
    """Convert forecast day-of-forecast to 'Mon DD' calendar label."""
    if np.isnan(doy) or doy < 0:
        return "n/a"
    d = datetime.date(init_year, init_month, 1) + datetime.timedelta(days=int(doy))
    return d.strftime("%b %d")


def _tier(value: float, thresholds: list, invert: bool = False) -> int:
    """
    Assign advisory tier 0-3 (green→red) from sorted thresholds.
    invert=True: higher values are BETTER (resilience), reverse mapping.
    """
    if np.isnan(value):
        return -1
    tier = sum(value >= t for t in thresholds)
    if invert:
        tier = len(thresholds) - tier
    return max(0, min(len(thresholds), tier))


def figure8_location_advisory(ds, out_dir, init_date, model):
    """
    4-panel zonal advisory choropleth — Ethiopia's 79 administrative zones (GADM ADM-2).
    Each panel classifies zones into 4 advisory tiers via traffic-light colours.
    Emergency zones are labelled; Alert zones get a bold border highlight.
    """
    lats = ds.latitude.values
    lons = ds.longitude.values
    proj = ccrs.PlateCarree()

    gdf = _load_admin2()
    if gdf is None or gdf.empty:
        print("  [skip] Fig 8: GADM ADM-2 zones not available")
        return

    # ── Advisory panel specs ─────────────────────────────────────────────
    PANELS = [
        ("advisory_score_mean",
         "(a)  Crop & Water Advisory",
         "Composite risk — water stress, dry spells, rainfall deficit",
         [0.25, 0.50, 0.75], False,
         lambda v: f"{v:.2f}"),
        ("onset_day_of_forecast_mean",
         "(b)  Kiremt Onset Advisory",
         "Forecast rainy-season start (Jun ideal → Sep+ crisis)",
         [61, 92, 122], False,
         lambda v: _doy_label(v)),
        ("thi_max_mean",
         "(c)  Livestock Heat Stress",
         "Peak THI — WMO thresholds: mild 68 / moderate 72 / severe 78",
         [68, 72, 78], False,
         lambda v: f"THI {v:.0f}"),
        ("resilience_score_mean",
         "(d)  Resilience Outlook",
         "Integrated livelihood resilience (higher = better)",
         [0.25, 0.50, 0.75], True,
         lambda v: f"{v:.2f}"),
    ]

    # ── Pre-compute all zonal values (shared across panels for efficiency) ──
    lons_2d, lats_2d = np.meshgrid(lons, lats)
    flat_lons = lons_2d.ravel().astype(float)
    flat_lats = lats_2d.ravel().astype(float)

    from shapely import contains_xy

    def _zone_mean(geom, data_flat):
        inside = contains_xy(geom, flat_lons, flat_lats)
        vals   = data_flat[inside]
        vals   = vals[np.isfinite(vals)]
        return float(np.nanmean(vals)) if len(vals) else np.nan

    # ── Layout: 2×2 maps + shared legend ────────────────────────────────
    fig = plt.figure(figsize=(18, 16), facecolor="#FAFAF8")
    gs  = fig.add_gridspec(
        3, 2,
        height_ratios=[1, 1, 0.11],
        hspace=0.28, wspace=0.14,
        top=0.90, bottom=0.04, left=0.03, right=0.97,
    )
    ax_grid = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for panel_idx, (var, ptitle, subtitle, thresh, inv, fmt) in enumerate(PANELS):
        if var not in ds:
            print(f"  [skip] Fig 8 panel {panel_idx+1}: {var} not found")
            continue

        data_flat = ds[var].values.ravel()
        r, c = ax_grid[panel_idx]
        ax   = fig.add_subplot(gs[r, c], projection=proj)

        # ── Count tiers across all woredas ───────────────────────────
        tier_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        # ── Draw each woreda polygon ──────────────────────────────────
        for _, row in gdf.iterrows():
            geom      = row.geometry
            zone_name = row.get("NAME_3", row.get("NAME_2", ""))
            val       = _zone_mean(geom, data_flat)
            t         = _tier(val, thresh, invert=inv)
            if t >= 0:
                tier_counts[t] += 1
            fill    = _ADV_COLORS[t] if t >= 0 else "#d0d0d0"
            edgeclr = {3: "#7f0000", 2: "#7f3000", 1: "#555", 0: "#888"}.get(t, "#aaa")
            edgelw  = {3: 1.5,       2: 1.0,       1: 0.35,  0: 0.25}.get(t, 0.25)

            ax.add_feature(
                ShapelyFeature([geom], proj,
                               facecolor=fill, edgecolor=edgeclr,
                               linewidth=edgelw, alpha=0.92),
                zorder=2,
            )

            # Label Emergency woredas
            if t == 3 and not np.isnan(val):
                try:
                    cx, cy = geom.centroid.x, geom.centroid.y
                    short  = zone_name[:14]
                    ax.text(cx, cy, f"★ {short}",
                            transform=proj, fontsize=4.8,
                            ha="center", va="center",
                            color="white", fontweight="bold", zorder=8,
                            bbox=dict(facecolor="#7f0000", alpha=0.85,
                                      edgecolor="none", pad=0.9,
                                      boxstyle="round,pad=0.2"))
                except Exception:
                    pass

        # ── Map frame ────────────────────────────────────────────────
        ax.set_extent(EXTENT, crs=proj)
        ax.add_feature(cfeature.OCEAN, facecolor="#c8dff0", zorder=3)
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "cultural", "admin_1_states_provinces", "10m",
                edgecolor="#333", facecolor="none", linewidth=0.65),
            zorder=5,
        )
        if _ETHIOPIA_GEOM:
            ax.add_feature(
                ShapelyFeature([_ETHIOPIA_GEOM], proj,
                               facecolor="none", edgecolor="#000",
                               linewidth=2.0),
                zorder=6,
            )
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "physical", "rivers_lake_centerlines", "10m",
                edgecolor="#6ea6cc", facecolor="none", linewidth=0.30),
            zorder=4,
        )

        # City markers for geographic reference
        _add_city_markers(ax, show_labels=True, fontsize=5.0)

        # North arrow on first panel only
        if panel_idx == 0:
            _north_arrow(ax, x=0.93, y=0.06, length=0.07)

        gl = ax.gridlines(draw_labels=True, linewidth=0, color="none",
                          x_inline=False, y_inline=False, crs=proj)
        gl.top_labels   = False
        gl.right_labels = False
        gl.xlocator     = mticker.FixedLocator([34, 36, 38, 40, 42, 44, 46, 48])
        gl.ylocator     = mticker.FixedLocator([4, 6, 8, 10, 12, 14])
        gl.xlabel_style = {"size": 5.5, "color": "#555"}
        gl.ylabel_style = {"size": 5.5, "color": "#555"}

        # ── Panel title ───────────────────────────────────────────────
        ax.set_title(ptitle, fontsize=11.5, fontweight="bold", pad=6,
                     color="#1a1a2e")
        ax.set_xlabel(subtitle, fontsize=7.2, labelpad=3, color="#444",
                      fontstyle="italic")

        # ── Tier count badge (bottom-left) ────────────────────────────
        badge_lines = [
            f"★ Emergency: {tier_counts[3]:>3}",
            f"■ Alert:     {tier_counts[2]:>3}",
            f"▲ Watch:     {tier_counts[1]:>3}",
            f"● Normal:    {tier_counts[0]:>3}",
        ]
        badge_colors = ["#b30000", "#ec7014", "#c9a000", "#1a9641"]
        for bi, (bline, bcol) in enumerate(zip(badge_lines, badge_colors)):
            ax.text(0.02, 0.985 - bi * 0.062, bline,
                    transform=ax.transAxes, fontsize=5.5,
                    va="top", ha="left", zorder=12,
                    fontfamily="monospace",
                    color=bcol, fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.88,
                              edgecolor="none", pad=0.6) if bi == 0 else None)

    # ── Shared bottom legend (styled IPC-tier boxes) ──────────────────────
    ax_leg = fig.add_subplot(gs[2, :])
    ax_leg.set_facecolor("#FAFAF8")
    ax_leg.axis("off")

    tier_data = list(zip(_ADV_COLORS, _ADV_LABELS, _ADV_DESCS, _ADV_SYMBOLS))
    n_tiers = len(tier_data)
    total_w = 0.88
    slot_w  = total_w / n_tiers
    x0      = (1.0 - total_w) / 2

    for i, (color, label, desc, sym) in enumerate(tier_data):
        x = x0 + i * slot_w
        # Coloured swatch
        rect = mpatches.FancyBboxPatch(
            (x, 0.12), 0.045, 0.70,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="#333", linewidth=1.0,
            transform=ax_leg.transAxes,
        )
        ax_leg.add_patch(rect)
        ax_leg.text(x + 0.052, 0.78, f"{sym}  {label}",
                    transform=ax_leg.transAxes,
                    fontsize=10.5, va="center", ha="left",
                    fontweight="bold", color="#111")
        ax_leg.text(x + 0.052, 0.30, desc,
                    transform=ax_leg.transAxes,
                    fontsize=8.5, va="center", ha="left", color="#555")

    # ── Figure title ──────────────────────────────────────────────────────
    fig.suptitle(
        (f"{model.upper()} Seasonal Forecast  ·  Ethiopia Woreda-Level Advisory Bulletin  ·  "
         f"Kiremt 2026 (JJAS)  ·  Init: {init_date}  ·  51-member Ensemble\n"
         f"⚠  EMI & ICPC El Niño Declared — 690 Woredas Assessed"),
        fontsize=13, fontweight="bold", y=0.97, color="#1a1a2e",
    )
    _source_note(fig, model, init_date)

    out_path = out_dir / "Fig8_location_advisory.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}  ({len(gdf)} zones)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def model_folder(model):
    return {
        "ecmwf": "ecmwf_system51", "ukmo": "ukmo_system610",
        "ncep":  "ncep_system2",   "jma":  "jma_system4",
        "cmcc":  "cmcc_system4",  "dwd":  "dwd_system22",
        "meteo_france": "meteo_france_system9",
        "eccc":  "eccc_system5",   "bom":  "bom_system2",
    }.get(model.lower(), model)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--year",  type=int, default=2026)
    p.add_argument("--month", type=int, default=5)
    p.add_argument("--day",   type=int, default=1)
    p.add_argument("--model", default="ecmwf")
    p.add_argument("--country", default=DEFAULT_COUNTRY)
    p.add_argument("--root",  default=None)
    p.add_argument("--outdir", default=None)
    p.add_argument("--figs", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8])
    return p.parse_args()


def main():
    args = parse_args()
    global EXTENT, _GADM_ADM3_PATH
    EXTENT = country_bbox_lonlat(args.country)
    _GADM_ADM3_PATH = boundary_path(args.country, 3)
    root = Path(args.root) if args.root else cds_root(args.country)
    init_date = f"{args.year:04d}-{args.month:02d}-{args.day:02d}"
    model_dir = (root / "seasonal-original-single-levels"
                 / f"{args.year:04d}" / f"{args.month:02d}"
                 / f"{args.day:02d}" / model_folder(args.model))

    ens_path = model_dir / "indices" / "ensemble_statistics.nc"
    if not ens_path.exists():
        print(f"ERROR: {ens_path} not found"); return

    out_dir = (Path(args.outdir) if args.outdir
               else model_dir / "indices" / "plots" / "publication")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*68}")
    print(f"  {args.model.upper()} | Ethiopia Kiremt | Init: {init_date}")
    print(f"  Output: {out_dir}")
    print(f"{'='*68}")

    print("  Loading geometries …", end=" ", flush=True)
    _load_geometries()
    print("done")

    ds_ens = xr.open_dataset(ens_path)
    ds_mem = xr.open_dataset(model_dir / "indices" / "indices_per_member.nc") \
             if (model_dir / "indices" / "indices_per_member.nc").exists() else None
    ds_mon = xr.open_dataset(model_dir / "indices" / "monthly_weather_summaries.nc") \
             if (model_dir / "indices" / "monthly_weather_summaries.nc").exists() else None

    figs = set(args.figs)
    lats, lons = ds_ens.latitude.values, ds_ens.longitude.values

    if 1 in figs: figure1_crop_water(ds_ens,   out_dir, init_date, args.model)
    if 2 in figs: figure2_season_timing(ds_ens, out_dir, init_date, args.model)
    if 3 in figs: figure3_livestock(ds_ens,     out_dir, init_date, args.model)
    if 4 in figs: figure4_advisory(ds_ens,      out_dir, init_date, args.model)
    if 5 in figs: figure5_uncertainty(ds_ens,   out_dir, init_date, args.model)
    if 6 in figs:
        if ds_mon: figure6_june(ds_mon, out_dir, init_date, args.model)
        else: print("  [skip] Fig 6: monthly_weather_summaries.nc not found")
    if 7 in figs:
        if ds_mem: figure7_city_boxplots(ds_mem, out_dir, init_date, args.model)
        else: print("  [skip] Fig 7: indices_per_member.nc not found")
    if 8 in figs:
        figure8_location_advisory(ds_ens, out_dir, init_date, args.model)

    ds_ens.close()
    if ds_mem: ds_mem.close()
    if ds_mon: ds_mon.close()

    print(f"\n  PNG (300 dpi) + PDF → {out_dir}")


if __name__ == "__main__":
    main()
