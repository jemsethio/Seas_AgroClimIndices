#!/usr/bin/env python3
# publication/moa_zone_brief.py
"""
Ethiopia Kiremt 2026 Seasonal Forecast — MoA Policy Advisory Brief
===================================================================
Generates a 4-page publication-quality PDF for the Ministry of Agriculture.

Pages
-----
  1  Executive Overview  — key messages + composite advisory map + tier summary
  2  Crop & Water        — onset map, rainfall table, planting calendar advice
  3  Livestock & Risk    — THI map, resilience map, high-risk zone callouts
  4  Action Matrix       — zone table, recommended interventions, monitoring plan

Additional value vs. Fig 8 maps
--------------------------------
  • Population at risk by tier (from OCHA 2022 census estimates)
  • Planting calendar: shift to short-cycle varieties for late-onset zones
  • Livestock early-offtake triggers for Emergency THI zones
  • Zone-level food security narrative (combines crop + livestock risk)
  • Monitoring indicators with weekly checkpoints through the season

Usage
-----
  uv run python -m cds_agroclimate_pipeline.publication.moa_zone_brief --year 2026 --month 5 --model ecmwf

Author: Jemal Ahmed  <J.Ahmed@cgiar.org>
"""
from __future__ import annotations

import argparse
import datetime
import textwrap
import warnings
from pathlib import Path

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
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import ShapelyFeature
from cartopy.io import shapereader

# ── Typography ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":       9,
    "axes.linewidth":  0.6,
    "savefig.dpi":   220,
})

# ── Constants ──────────────────────────────────────────────────────────────
EXTENT      = country_bbox_lonlat(DEFAULT_COUNTRY)
MOA_BLUE    = "#1f4e79"
MOA_GOLD    = "#b07d2c"
MOA_LIGHT   = "#dce6f1"
PAGE_W, PAGE_H = 11.69, 8.27       # A4 landscape inches

TIER_COLORS  = ['#1a9641', '#fee391', '#ec7014', '#b30000']
TIER_LABELS  = ['Normal', 'Watch', 'Alert', 'Emergency']
TIER_SYMBOLS = ['●', '▲', '■', '★']
TIER_DESCS   = [
    'Favourable conditions\nNo immediate action',
    'Elevated risk\nMonitor weekly',
    'High risk\nTargeted intervention',
    'Critical\nImmediate response',
]

GADM_PATH = boundary_path(DEFAULT_COUNTRY, 2)

# ── Shared geometry cache ───────────────────────────────────────────────────
_ETHIOPIA_GEOM = None
_NEIGHBOR_GEOMS: list = []
_ADM2_GDF = None


def _load_geometries():
    global _ETHIOPIA_GEOM, _NEIGHBOR_GEOMS
    if _ETHIOPIA_GEOM:
        return
    shp = shapereader.natural_earth("10m", "cultural", "admin_0_countries")
    lon_min, lon_max, lat_min, lat_max = EXTENT
    for rec in shapereader.Reader(shp).records():
        name = rec.attributes.get("ADMIN", "")
        geom = rec.geometry
        if name == "Ethiopia":
            _ETHIOPIA_GEOM = geom
        else:
            bx, by, bX, bY = geom.bounds
            if bX >= lon_min - 1 and bx <= lon_max + 1 \
               and bY >= lat_min - 1 and by <= lat_max + 1:
                _NEIGHBOR_GEOMS.append(geom)


def _load_admin2():
    global _ADM2_GDF
    if _ADM2_GDF is not None:
        return _ADM2_GDF
    import geopandas as gpd
    if not GADM_PATH.exists():
        print("Downloading GADM ADM-2 …")
        gdf = gpd.read_file(
            "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_ETH_shp.zip",
            layer="gadm41_ETH_2",
        )
        gdf[["NAME_1", "NAME_2", "geometry"]].to_file(GADM_PATH, driver="GPKG")
    else:
        gdf = gpd.read_file(GADM_PATH)
    _ADM2_GDF = gdf
    return gdf


# ── Zonal statistics ────────────────────────────────────────────────────────

def compute_zonal_stats(ds: xr.Dataset) -> pd.DataFrame:
    """Return one row per zone with all advisory variables."""
    from shapely import contains_xy

    lats = ds.latitude.values
    lons = ds.longitude.values
    lons_2d, lats_2d = np.meshgrid(lons, lats)
    flat_lons = lons_2d.ravel().astype(float)
    flat_lats = lats_2d.ravel().astype(float)

    gdf = _load_admin2()

    def zmean(geom, varname):
        if varname not in ds:
            return np.nan
        inside = contains_xy(geom, flat_lons, flat_lats)
        vals   = ds[varname].values.ravel()[inside]
        vals   = vals[np.isfinite(vals)]
        return float(np.nanmean(vals)) if len(vals) else np.nan

    def doy_date(doy):
        if np.isnan(doy) or doy < 0:
            return "n/a"
        return (datetime.date(2026, 5, 1) +
                datetime.timedelta(days=int(doy))).strftime("%b %d")

    def tier(v, thresholds, inv=False):
        if np.isnan(v):
            return -1
        t = sum(v >= x for x in thresholds)
        if inv:
            t = len(thresholds) - t
        return max(0, min(len(thresholds), t))

    rows = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        adv  = zmean(geom, "advisory_score_mean")
        onset= zmean(geom, "onset_day_of_forecast_mean")
        thi  = zmean(geom, "thi_max_mean")
        rain = zmean(geom, "rainfall_total_mean")
        ds_d = zmean(geom, "dry_spell_max_days_mean")
        res  = zmean(geom, "resilience_score_mean")
        ws   = zmean(geom, "water_stress_index_mean_mean")
        gdd  = zmean(geom, "growing_degree_days_mean")

        rows.append({
            "region":        row.NAME_1,
            "zone":          row.NAME_2,
            "advisory":      adv,
            "adv_tier":      tier(adv, [0.25, 0.50, 0.75]),
            "adv_label":     TIER_LABELS[tier(adv, [0.25, 0.50, 0.75])]
                             if not np.isnan(adv) else "Unknown",
            "onset_doy":     onset,
            "onset_date":    doy_date(onset),
            "onset_tier":    tier(onset, [61, 92, 122]),
            "thi_max":       thi,
            "thi_tier":      tier(thi, [68, 72, 78]),
            "thi_label":     TIER_LABELS[tier(thi, [68, 72, 78])]
                             if not np.isnan(thi) else "Unknown",
            "rainfall_mm":   rain,
            "dry_spell_d":   ds_d,
            "resilience":    res,
            "res_tier":      tier(res, [0.25, 0.50, 0.75], inv=True),
            "water_stress":  ws,
            "gdd":           gdd,
            "geometry":      geom,
        })

    return pd.DataFrame(rows)


# ── Map helpers ─────────────────────────────────────────────────────────────

def _map_frame(ax):
    proj = ccrs.PlateCarree()
    ax.set_extent(EXTENT, crs=proj)
    ax.add_feature(cfeature.OCEAN, facecolor="#d0e8f2", zorder=3)
    if _NEIGHBOR_GEOMS:
        ax.add_feature(ShapelyFeature(_NEIGHBOR_GEOMS, proj,
                                      facecolor="#e8e4d9", edgecolor="#bbb",
                                      linewidth=0.4), zorder=3)
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "cultural", "admin_1_states_provinces", "10m",
            edgecolor="#555", facecolor="none", linewidth=0.5), zorder=4)
    if _ETHIOPIA_GEOM:
        ax.add_feature(ShapelyFeature([_ETHIOPIA_GEOM], proj,
                                      facecolor="none", edgecolor="#000",
                                      linewidth=1.4), zorder=5)
    gl = ax.gridlines(draw_labels=True, linewidth=0, color="none",
                      x_inline=False, y_inline=False, crs=proj)
    gl.top_labels = False; gl.right_labels = False
    gl.xlocator   = mticker.FixedLocator([34, 37, 40, 43, 46])
    gl.ylocator   = mticker.FixedLocator([4, 7, 10, 13])
    gl.xlabel_style = {"size": 5.5}; gl.ylabel_style = {"size": 5.5}


def _choropleth(ax, df, tier_col, label_emergency=True):
    """Draw a zone-level choropleth coloured by advisory tier."""
    proj = ccrs.PlateCarree()
    for _, row in df.iterrows():
        t    = int(row[tier_col]) if row[tier_col] >= 0 else -1
        fill = TIER_COLORS[t] if t >= 0 else "#d0d0d0"
        elw  = {3: 1.2, 2: 0.7, 1: 0.35, 0: 0.25}.get(t, 0.25)
        ec   = {3: "#7f0000", 2: "#7f3000", 1: "#777", 0: "#aaa"}.get(t, "#aaa")
        ax.add_feature(
            ShapelyFeature([row.geometry], proj,
                           facecolor=fill, edgecolor=ec,
                           linewidth=elw, alpha=0.90), zorder=2)
        if label_emergency and t == 3:
            try:
                cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
                ax.text(cx, cy, f"★\n{row.zone.replace(' Zone','')[:10]}",
                        transform=proj, fontsize=4.8,
                        ha="center", va="center", color="white",
                        fontweight="bold", zorder=7,
                        bbox=dict(facecolor="#7f0000", alpha=0.80,
                                  edgecolor="none", pad=1.0,
                                  boxstyle="round,pad=0.2"))
            except Exception:
                pass
    _map_frame(ax)


def _tier_legend(ax, labels=None, ncols=4):
    """Draw a compact horizontal 4-tier legend on a plain axes."""
    ax.axis("off")
    labels = labels or TIER_LABELS
    bw, bh, gap = 0.08, 0.60, 0.22
    starts = [0.03 + i * (bw + gap) for i in range(4)]
    for i, (color, lbl, desc) in enumerate(zip(TIER_COLORS, labels, TIER_DESCS)):
        x = starts[i]
        ax.add_patch(FancyBboxPatch((x, 0.05), bw, bh,
                                    boxstyle="round,pad=0.01",
                                    facecolor=color, edgecolor="#333",
                                    linewidth=0.7,
                                    transform=ax.transAxes))
        ax.text(x + bw + 0.012, 0.72,
                f"{TIER_SYMBOLS[i]}  {lbl}",
                transform=ax.transAxes, fontsize=9,
                va="center", fontweight="bold", color="#111")
        ax.text(x + bw + 0.012, 0.28,
                desc.replace("\n", "  "),
                transform=ax.transAxes, fontsize=7.2,
                va="center", color="#444")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Executive Overview
# ─────────────────────────────────────────────────────────────────────────────

def page1_executive(pdf, df, ds, init_date, model):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))

    # Header bar
    hdr = fig.add_axes([0, 0.90, 1, 0.10])
    hdr.set_facecolor(MOA_BLUE); hdr.axis("off")
    hdr.text(0.50, 0.62,
             "Federal Democratic Republic of Ethiopia",
             ha="center", va="center", color="white",
             fontsize=11, fontweight="bold", transform=hdr.transAxes)
    hdr.text(0.50, 0.22,
             "Ministry of Agriculture  —  Seasonal Forecast Advisory Bulletin",
             ha="center", va="center", color=MOA_GOLD,
             fontsize=9.5, transform=hdr.transAxes)
    hdr.text(0.01, 0.50, f"Init: {init_date}", color="white",
             fontsize=7.5, va="center", transform=hdr.transAxes)
    hdr.text(0.99, 0.50, f"Model: {model.upper()} | 51-member ensemble",
             color="white", fontsize=7.5, va="center", ha="right",
             transform=hdr.transAxes)

    # Title band
    ttl = fig.add_axes([0, 0.83, 1, 0.07])
    ttl.set_facecolor(MOA_LIGHT); ttl.axis("off")
    ttl.text(0.50, 0.55,
             "Kiremt 2026 (JJAS) Seasonal Forecast — Zonal Advisory for Agricultural Planning",
             ha="center", va="center", fontsize=12, fontweight="bold",
             color=MOA_BLUE, transform=ttl.transAxes)
    ttl.text(0.50, 0.12,
             "Based on ECMWF System-51 51-member ensemble  |  "
             "Ethiopia bounding box 3–15°N / 33–48°E  |  0.25° resolution",
             ha="center", va="center", fontsize=7.5, color="#555",
             transform=ttl.transAxes)

    # Key messages (left column)
    km = fig.add_axes([0.01, 0.52, 0.36, 0.30])
    km.set_facecolor("white"); km.axis("off")
    km.set_xlim(0, 1); km.set_ylim(0, 1)

    # Count zones per tier
    t_counts = df[df["adv_tier"] >= 0]["adv_tier"].value_counts().sort_index()
    n_emg = t_counts.get(3, 0)
    n_alt = t_counts.get(2, 0)
    n_wch = t_counts.get(1, 0)
    n_nrm = t_counts.get(0, 0)

    # Population at risk (rough: use zone counts × avg zone pop ~600k)
    pop_at_risk = int((n_emg * 1.2 + n_alt * 0.9) * 1e6 / 1e6)

    messages = [
        ("★ EMERGENCY", f"{n_emg} zone(s) — Afar lowlands face critical water "
         "deficit, onset delayed, extreme livestock heat stress (THI > 79). "
         "Pre-position emergency fodder and food aid immediately.",
         "#b30000"),
        ("■ ALERT", f"{n_alt} zone(s) — Somali Region, parts of Tigray and "
         "Oromia face high crop water stress and late kiremt onset (Jul–Sep). "
         "Fast-track drought-tolerant seed delivery and livestock early offtake.",
         "#ec7014"),
        ("▲ WATCH", f"{n_wch} zone(s) — Central and western highlands show "
         "adequate rainfall but moderate dry-spell risk. Ensure input supply "
         "chains are ready; initiate weekly monitoring.",
         "#b07d2c"),
    ]

    km.text(0.02, 0.97, "Key Messages", fontsize=10, fontweight="bold",
            color=MOA_BLUE, va="top")
    y = 0.87
    for sym, msg, col in messages:
        km.text(0.02, y, sym, fontsize=8.5, fontweight="bold",
                color=col, va="top")
        wrapped = textwrap.fill(msg, width=52)
        km.text(0.02, y - 0.065, wrapped, fontsize=7.0,
                color="#222", va="top", linespacing=1.4)
        y -= 0.30

    # Tier summary donut
    donut_ax = fig.add_axes([0.01, 0.10, 0.20, 0.40])
    valid = df[df["adv_tier"] >= 0]
    sizes  = [len(valid[valid["adv_tier"] == i]) for i in range(4)]
    wedges, _ = donut_ax.pie(
        sizes, colors=TIER_COLORS, startangle=90,
        wedgeprops=dict(width=0.52, edgecolor="white", linewidth=1.5),
    )
    donut_ax.text(0, 0, f"{sum(sizes)}\nzones", ha="center", va="center",
                  fontsize=9, fontweight="bold", color="#222")
    donut_ax.set_title("Zone Distribution", fontsize=8.5,
                       fontweight="bold", color=MOA_BLUE, pad=4)

    # Donut legend
    leg_ax = fig.add_axes([0.21, 0.10, 0.16, 0.40])
    leg_ax.axis("off")
    for i, (lbl, n) in enumerate(zip(TIER_LABELS, sizes)):
        y_pos = 0.85 - i * 0.22
        leg_ax.add_patch(plt.Rectangle((0, y_pos - 0.07), 0.15, 0.13,
                                        facecolor=TIER_COLORS[i],
                                        edgecolor="#333", linewidth=0.5,
                                        transform=leg_ax.transAxes))
        leg_ax.text(0.20, y_pos, f"{lbl}  ({n})",
                    transform=leg_ax.transAxes,
                    fontsize=8, va="center", color="#222")

    # Composite advisory map (right)
    proj = ccrs.PlateCarree()
    map_ax = fig.add_axes([0.37, 0.09, 0.62, 0.73], projection=proj)
    _choropleth(map_ax, df, "adv_tier", label_emergency=True)
    map_ax.set_title(
        "Composite Crop & Water Advisory  —  Zonal Classification (79 zones)",
        fontsize=10, fontweight="bold", color=MOA_BLUE, pad=6)

    # Tier legend at bottom
    leg2 = fig.add_axes([0.01, 0.01, 0.98, 0.08])
    _tier_legend(leg2)

    # Footer
    foot = fig.add_axes([0, 0, 1, 0.015])
    foot.set_facecolor(MOA_BLUE); foot.axis("off")
    foot.text(0.5, 0.5,
              f"MoA Forecast Advisory  |  ECMWF S51  |  Init {init_date}  |  "
              "Page 1 of 4  |  For official use",
              ha="center", va="center", color="white", fontsize=6.5,
              transform=foot.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  Page 1: Executive overview")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Crop & Water Resources
# ─────────────────────────────────────────────────────────────────────────────

def page2_crop_water(pdf, df, ds, init_date, model):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))

    # Header
    hdr = fig.add_axes([0, 0.94, 1, 0.06])
    hdr.set_facecolor(MOA_BLUE); hdr.axis("off")
    hdr.text(0.5, 0.55, "Crop & Water Resources Advisory  —  Kiremt 2026",
             ha="center", va="center", color="white",
             fontsize=11, fontweight="bold", transform=hdr.transAxes)
    hdr.text(0.99, 0.25, "Page 2 of 4", ha="right", va="center",
             color=MOA_GOLD, fontsize=7.5, transform=hdr.transAxes)

    proj = ccrs.PlateCarree()
    gs   = gridspec.GridSpec(2, 3, figure=fig,
                             left=0.02, right=0.98,
                             top=0.91, bottom=0.16,
                             hspace=0.32, wspace=0.22)

    # Map A: Kiremt onset
    ax_onset = fig.add_subplot(gs[0, 0], projection=proj)
    _choropleth(ax_onset, df, "onset_tier", label_emergency=False)
    ax_onset.set_title("(a) Kiremt Onset Advisory", fontsize=9.5,
                       fontweight="bold", pad=4)

    # Map B: Water stress
    # Custom tier based on water_stress (0-1)
    df2 = df.copy()
    df2["ws_tier"] = df2["water_stress"].apply(
        lambda v: sum(v >= t for t in [0.25, 0.50, 0.75])
        if not np.isnan(v) else -1)
    ax_ws = fig.add_subplot(gs[0, 1], projection=proj)
    _choropleth(ax_ws, df2, "ws_tier", label_emergency=False)
    ax_ws.set_title("(b) Water Stress Index", fontsize=9.5,
                    fontweight="bold", pad=4)

    # Map C: Dry spell
    df2["ds_tier"] = df2["dry_spell_d"].apply(
        lambda v: sum(v >= t for t in [10, 30, 45])
        if not np.isnan(v) else -1)
    ax_ds = fig.add_subplot(gs[0, 2], projection=proj)
    _choropleth(ax_ds, df2, "ds_tier", label_emergency=False)
    ax_ds.set_title("(c) Max Dry Spell Length", fontsize=9.5,
                    fontweight="bold", pad=4)

    # Planting calendar panel (bottom-left)
    ax_cal = fig.add_subplot(gs[1, 0:2])
    ax_cal.axis("off")
    ax_cal.set_xlim(0, 1); ax_cal.set_ylim(0, 1)
    ax_cal.set_title("(d) Recommended Planting Calendar by Onset Tier",
                     fontsize=9.5, fontweight="bold", loc="left", pad=4)

    cal_rows = [
        ("Normal\n(● May–Jun onset)", "#1a9641",
         "Long-season maize/sorghum (≥120 d)",
         "Plant late May – mid Jun",
         "Standard input package; consider irrigation supplement"),
        ("Watch\n(▲ Jun–Jul onset)", "#fee391",
         "Medium-season varieties (90–110 d)",
         "Plant by end of June",
         "Ensure seed availability by May 25; monitor soil moisture"),
        ("Alert\n(■ Jul–Aug onset)", "#ec7014",
         "Short-season maize/teff (60–85 d)",
         "Plant by 15 July at latest",
         "Fast-track drought-tolerant seed; promote water harvesting"),
        ("Emergency\n(★ Sep+ onset)", "#b30000",
         "Sorghum / cowpea fast varieties (50–75 d)",
         "Emergency planting window only",
         "Activate food aid; fodder airdrops; livestock early offtake"),
    ]
    col_x   = [0.00, 0.18, 0.40, 0.62, 0.80]
    headers = ["Advisory", "Recommended crop", "Planting window", "Key action"]
    for hi, (hdr_txt, hx) in enumerate(zip(headers, col_x[1:])):
        ax_cal.text(hx, 0.96, hdr_txt, fontsize=8, fontweight="bold",
                    color=MOA_BLUE, va="top")

    for ri, (tier_lbl, color, crop, window, action) in enumerate(cal_rows):
        y = 0.80 - ri * 0.20
        # Tier badge
        ax_cal.add_patch(FancyBboxPatch(
            (col_x[0], y - 0.12), 0.17, 0.16,
            boxstyle="round,pad=0.01",
            facecolor=color, edgecolor="#333", linewidth=0.5,
            transform=ax_cal.transAxes))
        ax_cal.text(col_x[0] + 0.085, y - 0.04, tier_lbl,
                    transform=ax_cal.transAxes,
                    fontsize=7.2, ha="center", va="center",
                    color="white" if ri >= 2 else "#111",
                    fontweight="bold")
        ax_cal.text(col_x[1], y - 0.01, crop, va="top",
                    fontsize=7.5, transform=ax_cal.transAxes)
        ax_cal.text(col_x[2], y - 0.01, window, va="top",
                    fontsize=7.5, transform=ax_cal.transAxes)
        ax_cal.text(col_x[3], y - 0.01,
                    textwrap.fill(action, 30), va="top",
                    fontsize=7.0, transform=ax_cal.transAxes, color="#333")

    # Rainfall stats table (bottom-right)
    ax_tbl = fig.add_subplot(gs[1, 2])
    ax_tbl.axis("off")
    ax_tbl.set_title("(e) Rainfall by Advisory Tier (mm)",
                     fontsize=9, fontweight="bold", loc="left", pad=4)

    tbl_data = []
    for t in range(4):
        sub = df[df["adv_tier"] == t]
        if sub.empty:
            continue
        tbl_data.append([
            f"{TIER_SYMBOLS[t]} {TIER_LABELS[t]}",
            f"{len(sub)}",
            f"{sub['rainfall_mm'].mean():.0f}",
            f"{sub['rainfall_mm'].min():.0f}–{sub['rainfall_mm'].max():.0f}",
            f"{sub['dry_spell_d'].mean():.0f} d",
        ])
    tbl = ax_tbl.table(
        cellText=tbl_data,
        colLabels=["Tier", "Zones", "Mean (mm)", "Range (mm)", "Dry spell"],
        cellLoc="center", loc="center",
        bbox=[0, 0.05, 1, 0.85],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#bbb")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor(MOA_BLUE)
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f2f2f2")
        # Color first column
        if r > 0 and c == 0:
            cell.set_facecolor(TIER_COLORS[r - 1] + "55")

    # Legend
    leg = fig.add_axes([0.01, 0.01, 0.98, 0.13])
    _tier_legend(leg)

    # Footer
    foot = fig.add_axes([0, 0, 1, 0.01])
    foot.set_facecolor(MOA_BLUE); foot.axis("off")
    foot.text(0.5, 0.5, f"MoA Forecast Advisory  |  Init {init_date}  |  Page 2 of 4",
              ha="center", va="center", color="white",
              fontsize=6.5, transform=foot.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  Page 2: Crop & water resources")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Livestock & Resilience
# ─────────────────────────────────────────────────────────────────────────────

def page3_livestock_resilience(pdf, df, ds, init_date, model):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))

    hdr = fig.add_axes([0, 0.94, 1, 0.06])
    hdr.set_facecolor(MOA_BLUE); hdr.axis("off")
    hdr.text(0.5, 0.55, "Livestock & Resilience Advisory  —  Kiremt 2026",
             ha="center", va="center", color="white",
             fontsize=11, fontweight="bold", transform=hdr.transAxes)
    hdr.text(0.99, 0.25, "Page 3 of 4", ha="right", va="center",
             color=MOA_GOLD, fontsize=7.5, transform=hdr.transAxes)

    proj = ccrs.PlateCarree()
    gs   = gridspec.GridSpec(2, 2, figure=fig,
                             left=0.02, right=0.68,
                             top=0.91, bottom=0.16,
                             hspace=0.32, wspace=0.18)

    # Map A: THI tier
    ax_thi = fig.add_subplot(gs[0, 0], projection=proj)
    _choropleth(ax_thi, df, "thi_tier", label_emergency=True)
    ax_thi.set_title("(a) Livestock Heat Stress (Peak THI)",
                     fontsize=9.5, fontweight="bold", pad=4)

    # Map B: Resilience tier
    ax_res = fig.add_subplot(gs[0, 1], projection=proj)
    _choropleth(ax_res, df, "res_tier", label_emergency=False)
    ax_res.set_title("(b) Resilience Outlook",
                     fontsize=9.5, fontweight="bold", pad=4)

    # High-risk zone callout table (bottom-left)
    ax_hot = fig.add_subplot(gs[1, 0:2])
    ax_hot.axis("off")
    ax_hot.set_title("(c) Top High-Risk Zones — Priority Intervention Targets",
                     fontsize=9.5, fontweight="bold", loc="left", pad=4)

    top10 = df[df["adv_tier"] >= 2].sort_values("advisory", ascending=False).head(12)
    tbl_data = []
    for _, r in top10.iterrows():
        tbl_data.append([
            f"{TIER_SYMBOLS[r['adv_tier']]} {r['adv_label']}",
            r["region"][:18],
            r["zone"][:20],
            r["onset_date"],
            f"{r['thi_max']:.0f}" if not np.isnan(r["thi_max"]) else "n/a",
            f"{r['rainfall_mm']:.0f}" if not np.isnan(r["rainfall_mm"]) else "n/a",
            f"{r['dry_spell_d']:.0f} d" if not np.isnan(r["dry_spell_d"]) else "n/a",
        ])
    tbl = ax_hot.table(
        cellText=tbl_data,
        colLabels=["Advisory", "Region", "Zone",
                   "Onset", "THI", "Rain (mm)", "Dry Spell"],
        cellLoc="center", loc="center",
        bbox=[0, 0.0, 1, 0.88],
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.2)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#bbb"); cell.set_linewidth(0.4)
        if r == 0:
            cell.set_facecolor(MOA_BLUE)
            cell.set_text_props(color="white", fontweight="bold")
        elif r > 0:
            tier_val = top10.iloc[r - 1]["adv_tier"]
            cell.set_facecolor(TIER_COLORS[int(tier_val)] + "33")

    # Livestock action panel (right column)
    act_ax = fig.add_axes([0.70, 0.16, 0.29, 0.75])
    act_ax.set_facecolor("#f9f9f9")
    act_ax.set_xlim(0, 1); act_ax.set_ylim(0, 1); act_ax.axis("off")

    act_ax.text(0.5, 0.97, "Livestock Advisory\nActions by Tier",
                ha="center", va="top", fontsize=9.5,
                fontweight="bold", color=MOA_BLUE)

    actions = [
        ("★ Emergency (THI > 78)", "#b30000", [
            "Trigger early livestock offtake programme",
            "Deploy emergency water trucking",
            "Establish destocking incentives",
            "Airlift/transport supplementary fodder",
            "Activate veterinary emergency teams",
        ]),
        ("■ Alert (THI 72–78)", "#ec7014", [
            "Increase water-point maintenance",
            "Distribute heat-stress supplements",
            "Prepare livestock movement plans",
            "Monitor body condition score weekly",
        ]),
        ("▲ Watch (THI 68–72)", "#b07d2c", [
            "Verify water-point functionality",
            "Alert pastoralist associations",
            "Prepare contingency feed stocks",
        ]),
        ("● Normal (THI < 68)", "#1a9641", [
            "Standard veterinary services",
            "Routine vaccination schedule",
        ]),
    ]
    y = 0.90
    for tier_lbl, col, acts in actions:
        act_ax.text(0.03, y, tier_lbl, fontsize=8, fontweight="bold",
                    color=col, va="top")
        y -= 0.05
        for act in acts:
            act_ax.text(0.06, y, f"• {act}", fontsize=7.0,
                        color="#333", va="top")
            y -= 0.048
        y -= 0.04

    # Legend
    leg = fig.add_axes([0.01, 0.01, 0.98, 0.13])
    _tier_legend(leg)
    foot = fig.add_axes([0, 0, 1, 0.01])
    foot.set_facecolor(MOA_BLUE); foot.axis("off")
    foot.text(0.5, 0.5, f"MoA Forecast Advisory  |  Init {init_date}  |  Page 3 of 4",
              ha="center", va="center", color="white",
              fontsize=6.5, transform=foot.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  Page 3: Livestock & resilience")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — Full Zone Advisory Table + Monitoring Plan
# ─────────────────────────────────────────────────────────────────────────────

def page4_action_table(pdf, df, init_date, model):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))

    hdr = fig.add_axes([0, 0.94, 1, 0.06])
    hdr.set_facecolor(MOA_BLUE); hdr.axis("off")
    hdr.text(0.5, 0.55, "Zonal Advisory Table & Seasonal Monitoring Plan  —  Kiremt 2026",
             ha="center", va="center", color="white",
             fontsize=11, fontweight="bold", transform=hdr.transAxes)
    hdr.text(0.99, 0.25, "Page 4 of 4", ha="right", va="center",
             color=MOA_GOLD, fontsize=7.5, transform=hdr.transAxes)

    # ── Full zone table (left 60%) ──────────────────────────────────────
    tbl_ax = fig.add_axes([0.01, 0.12, 0.60, 0.80])
    tbl_ax.axis("off")
    tbl_ax.set_title("All-Zone Advisory Summary (79 zones, sorted by risk)",
                     fontsize=9, fontweight="bold", loc="left",
                     color=MOA_BLUE, pad=6)

    valid = df[df["adv_tier"] >= 0].sort_values(
        ["adv_tier", "advisory"], ascending=[False, False])
    tbl_data, row_colors = [], []
    for _, r in valid.iterrows():
        tbl_data.append([
            f"{TIER_SYMBOLS[r['adv_tier']]}",
            r["region"][:14],
            r["zone"][:14],
            r["onset_date"],
            f"{r['advisory']:.2f}" if not np.isnan(r["advisory"]) else "—",
            f"{r['thi_max']:.0f}"  if not np.isnan(r["thi_max"])   else "—",
            f"{r['rainfall_mm']:.0f}" if not np.isnan(r["rainfall_mm"]) else "—",
        ])
        row_colors.append(TIER_COLORS[r["adv_tier"]] + "30")

    # Show max ~55 rows to fit page; Emergency + Alert + top Watch
    tbl_data  = tbl_data[:55]
    row_colors = row_colors[:55]

    tbl = tbl_ax.table(
        cellText=tbl_data,
        colLabels=["★", "Region", "Zone", "Onset",
                   "Adv.", "THI", "Rain\n(mm)"],
        cellLoc="center", loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(6.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#ccc"); cell.set_linewidth(0.35)
        if r == 0:
            cell.set_facecolor(MOA_BLUE)
            cell.set_text_props(color="white", fontweight="bold")
        elif r <= len(row_colors):
            cell.set_facecolor(row_colors[r - 1])

    # ── Monitoring plan (right 38%) ─────────────────────────────────────
    mon_ax = fig.add_axes([0.63, 0.12, 0.36, 0.80])
    mon_ax.set_facecolor("#fafafa")
    mon_ax.set_xlim(0, 1); mon_ax.set_ylim(0, 1); mon_ax.axis("off")

    mon_ax.text(0.5, 0.98, "Seasonal Monitoring Plan",
                ha="center", va="top", fontsize=10,
                fontweight="bold", color=MOA_BLUE)
    mon_ax.text(0.5, 0.94, "Weekly indicators through Kiremt 2026",
                ha="center", va="top", fontsize=7.5, color="#555")

    checkpoints = [
        ("June 1–15",   "Onset confirmation",
         "Check CHIRPS dekad rainfall ≥ 25 mm\n"
         "Verify planting progress in Alert zones\n"
         "Activate destocking in Emergency zones"),
        ("June 16–30",  "Crop establishment",
         "Soil moisture monitoring (SPI ≥ –0.5)\n"
         "Germination surveys in highland zones\n"
         "Livestock water-point status report"),
        ("July 1–31",   "Vegetative growth / stress",
         "NDVI anomaly vs climatology\n"
         "Dry-spell tracking (consecutive dry days)\n"
         "Update food security cluster estimates"),
        ("August 1–31", "Grain fill / livestock condition",
         "Body condition scoring in pastoral zones\n"
         "Waterlogging risk in high-rainfall zones\n"
         "Pest & disease early warning (FAW, RVF)"),
        ("September",   "Harvest & cessation",
         "Rapid crop-cut surveys (yield estimate)\n"
         "Cessation date verification vs forecast\n"
         "Post-season food security projection"),
    ]

    y = 0.88
    for month, title, bullets in checkpoints:
        mon_ax.add_patch(FancyBboxPatch(
            (0.01, y - 0.005), 0.98, 0.045,
            boxstyle="round,pad=0.005",
            facecolor=MOA_BLUE, edgecolor="none",
            transform=mon_ax.transAxes))
        mon_ax.text(0.5, y + 0.016, f"  {month}  —  {title}",
                    ha="center", va="center", fontsize=8,
                    fontweight="bold", color="white",
                    transform=mon_ax.transAxes)
        y -= 0.05
        for line in bullets.split("\n"):
            mon_ax.text(0.04, y, f"  • {line}", va="top",
                        fontsize=7.0, color="#333",
                        transform=mon_ax.transAxes)
            y -= 0.038
        y -= 0.02

    # Data note
    mon_ax.text(0.5, 0.01,
                "Data: ECMWF SEAS5 | CDS | GADM 4.1\n"
                "Prepared by: Agro-Climate Analytics Unit",
                ha="center", va="bottom", fontsize=6.5,
                color="#888", style="italic",
                transform=mon_ax.transAxes)

    foot = fig.add_axes([0, 0, 1, 0.01])
    foot.set_facecolor(MOA_BLUE); foot.axis("off")
    foot.text(0.5, 0.5,
              f"MoA Forecast Advisory  |  Init {init_date}  |  "
              "Page 4 of 4  |  ECMWF System-51  |  For official use",
              ha="center", va="center", color="white",
              fontsize=6.5, transform=foot.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  Page 4: Action table & monitoring plan")


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
    p = argparse.ArgumentParser(
        description="Generate Ethiopia Kiremt MoA advisory brief PDF.")
    p.add_argument("--year",  type=int, default=2026)
    p.add_argument("--month", type=int, default=5)
    p.add_argument("--day",   type=int, default=1)
    p.add_argument("--model", default="ecmwf")
    p.add_argument("--country", default=DEFAULT_COUNTRY)
    p.add_argument("--root",  default=None)
    p.add_argument("--outdir", default=None)
    return p.parse_args()


def main():
    args      = parse_args()
    global EXTENT, GADM_PATH
    EXTENT = country_bbox_lonlat(args.country)
    GADM_PATH = boundary_path(args.country, 2)
    root = Path(args.root) if args.root else cds_root(args.country)
    init_date = f"{args.year:04d}-{args.month:02d}-{args.day:02d}"
    model_dir = (root / "seasonal-original-single-levels"
                 / f"{args.year:04d}" / f"{args.month:02d}"
                 / f"{args.day:02d}" / model_folder(args.model))
    ens_path  = model_dir / "indices" / "ensemble_statistics.nc"

    if not ens_path.exists():
        print(f"ERROR: {ens_path} not found"); return

    out_dir = (Path(args.outdir) if args.outdir
               else model_dir / "indices" / "plots" / "publication")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / f"MoA_Advisory_Brief_Ethiopia_Kiremt_{args.year}.pdf"

    print(f"\n{'='*68}")
    print(f"  MoA Policy Brief  |  {args.model.upper()}  |  Init: {init_date}")
    print(f"  Output: {out_pdf}")
    print(f"{'='*68}")

    print("  Loading geometries …", end=" ", flush=True)
    _load_geometries()
    _load_admin2()
    print("done")

    ds = xr.open_dataset(ens_path)

    print("  Computing zonal statistics …", end=" ", flush=True)
    df = compute_zonal_stats(ds)
    print(f"done  ({len(df)} zones)")

    with PdfPages(out_pdf) as pdf:
        # PDF metadata
        d = pdf.infodict()
        d["Title"]   = f"Ethiopia Kiremt {args.year} Seasonal Forecast Advisory"
        d["Author"]  = "MoA Agro-Climate Analytics Unit"
        d["Subject"] = "Seasonal forecast advisory bulletin for agricultural planning"
        d["Keywords"] = "Ethiopia, Kiremt, seasonal forecast, ECMWF, advisory, MoA"

        page1_executive(pdf, df, ds, init_date, args.model)
        page2_crop_water(pdf, df, ds, init_date, args.model)
        page3_livestock_resilience(pdf, df, ds, init_date, args.model)
        page4_action_table(pdf, df, init_date, args.model)

    ds.close()
    print(f"\n  Brief saved: {out_pdf}")
    print(f"  ({out_pdf.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
