from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "agroclimate_indices"))
sys.path.insert(0, str(PROJECT_ROOT / "agroclimate_indices" / ".deps"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "data" / "cache" / "matplotlib"))

import cartopy
import cartopy.crs as ccrs
import fsspec
import matplotlib
import matplotlib.patheffects as path_effects
import matplotlib.tri as mtri
import numpy as np
from cartopy.io import shapereader
from omfiles import OmFileReader
from omfiles.grids import OmGrid
from shapely import contains_xy
from shapely.geometry.base import BaseGeometry

from agroclimate_indices import AgroClimateConfig, compute_agroclimate_indices, summarize_indices
from cds_agroclimate_pipeline.paths import country_bbox_lonlat, country_root


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


cartopy.config["data_dir"] = str(PROJECT_ROOT / "agroclimate_indices" / ".cartopy-data")


OPENMETEO_BUCKET = "openmeteo"
DEFAULT_MODEL = "ecmwf_ifs"
ZAMBIA_BBOX = tuple(country_bbox_lonlat("zambia"))  # lon_min, lon_max, lat_min, lat_max
PLOT_BBOX_PADDING = 0.6
ECMWF_VARIABLES = (
    "precipitation",
    "temperature_2m",
    "temperature_2m_max",
    "temperature_2m_min",
    "dew_point_2m",
    "shortwave_radiation",
    "pressure_msl",
    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm",
    "wind_u_component_10m",
    "wind_v_component_10m",
)


@dataclass(frozen=True)
class GridSelection:
    xmin: int
    xmax: int
    ymin: int
    ymax: int


@dataclass(frozen=True)
class GridSubset:
    selection: tuple[slice, slice]
    mask: np.ndarray
    lon: np.ndarray
    lat: np.ndarray


def parse_utc_time(value: str) -> dt.datetime:
    raw_text = value.strip()
    text = f"{raw_text[:-1]}+00:00" if raw_text.endswith("Z") else raw_text
    parsed = dt.datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc).replace(second=0, microsecond=0)


def openmeteo_https_uri(model: str, filename: str) -> str:
    return f"https://{OPENMETEO_BUCKET}.s3.amazonaws.com/data_spatial/{model}/{filename}"


def load_latest_metadata(model: str) -> dict:
    with urlopen(openmeteo_https_uri(model, "latest.json"), timeout=30) as response:
        return json.load(response)


def build_spatial_uri(model: str, run_time: dt.datetime, valid_time: dt.datetime) -> str:
    return (
        f"s3://{OPENMETEO_BUCKET}/data_spatial/{model}/"
        f"{run_time.year}/{run_time.month:02}/{run_time.day:02}/"
        f"{run_time.strftime('%H%MZ')}/"
        f"{valid_time.strftime('%Y-%m-%dT%H%M')}.om"
    )


def choose_daily_valid_times(metadata: dict, max_days: int) -> list[dt.datetime]:
    times = [parse_utc_time(value) for value in metadata["valid_times"]]
    midnight = [value for value in times if value.hour == 0]
    if len(midnight) >= max_days:
        return midnight[:max_days]
    stride = max(1, len(times) // max_days)
    return times[::stride][:max_days]


def find_boundary_grid_indices(grid: OmGrid, bbox: tuple[float, float, float, float]) -> GridSelection:
    lon_min, lon_max, lat_min, lat_max = bbox
    edge_lons = np.linspace(lon_min, lon_max, 9)
    edge_lats = np.linspace(lat_min, lat_max, 9)
    points = []
    points.extend((lon, lat_min) for lon in edge_lons)
    points.extend((lon, lat_max) for lon in edge_lons)
    points.extend((lon_min, lat) for lat in edge_lats)
    points.extend((lon_max, lat) for lat in edge_lats)

    xmin, xmax = np.inf, -np.inf
    ymin, ymax = np.inf, -np.inf
    for lon, lat in points:
        grid_point = grid.find_point_xy(float(lat), float(lon))
        if grid_point is None:
            continue
        xmin = min(xmin, grid_point.x)
        xmax = max(xmax, grid_point.x)
        ymin = min(ymin, grid_point.y)
        ymax = max(ymax, grid_point.y)

    if not np.isfinite([xmin, xmax, ymin, ymax]).all():
        raise ValueError("Zambia bbox does not overlap the model grid.")

    return GridSelection(
        xmin=max(int(xmin) - 1, 0),
        xmax=min(int(xmax) + 2, grid.shape[1]),
        ymin=max(int(ymin) - 1, 0),
        ymax=min(int(ymax) + 2, grid.shape[0]),
    )


def build_grid_subset(grid: OmGrid, bbox: tuple[float, float, float, float]) -> GridSubset:
    selection = find_boundary_grid_indices(grid, bbox)
    lon_min, lon_max, lat_min, lat_max = bbox

    if grid.is_gaussian:
        flat_indices = np.arange(selection.xmin, selection.xmax)
        lons, lats = gaussian_flat_coordinates(grid, flat_indices)
        mask = (lons >= lon_min) & (lons <= lon_max) & (lats >= lat_min) & (lats <= lat_max)
        return GridSubset(
            selection=(slice(0, 1), slice(selection.xmin, selection.xmax)),
            mask=mask,
            lon=lons[mask],
            lat=lats[mask],
        )

    lon_grid, lat_grid = grid.get_meshgrid()
    lon_grid = lon_grid[selection.ymin : selection.ymax, selection.xmin : selection.xmax]
    lat_grid = lat_grid[selection.ymin : selection.ymax, selection.xmin : selection.xmax]
    mask = (
        (lon_grid >= lon_min)
        & (lon_grid <= lon_max)
        & (lat_grid >= lat_min)
        & (lat_grid <= lat_max)
    )
    return GridSubset(
        selection=(slice(selection.ymin, selection.ymax), slice(selection.xmin, selection.xmax)),
        mask=mask,
        lon=lon_grid[mask],
        lat=lat_grid[mask],
    )


def gaussian_flat_coordinates(grid: OmGrid, flat_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gaussian = grid._grid
    lons = np.empty(flat_indices.shape, dtype=float)
    lats = np.empty(flat_indices.shape, dtype=float)

    out_start = 0
    for y in range(2 * gaussian.latitude_lines):
        row_start = gaussian._integral(y)
        row_end = gaussian._integral(y + 1)
        start = max(row_start, int(flat_indices[0]))
        end = min(row_end, int(flat_indices[-1]) + 1)
        if start >= end:
            continue

        count = end - start
        local_x = np.arange(start - row_start, end - row_start, dtype=float)
        nx = gaussian._nx_of_y(y)
        dy = 180.0 / (2 * gaussian.latitude_lines + 0.5)
        lat = (gaussian.latitude_lines - y - 1) * dy + dy / 2.0
        lon = local_x * (360.0 / nx)
        lon = np.where(lon >= 180.0, lon - 360.0, lon)

        lats[out_start : out_start + count] = lat
        lons[out_start : out_start + count] = lon
        out_start += count

    if out_start != flat_indices.size:
        raise RuntimeError(f"Only computed {out_start} coordinates for {flat_indices.size} flat points.")

    return lons, lats


def read_subset_variable(reader: OmFileReader, variable: str, subset: GridSubset) -> np.ndarray:
    child = reader.get_child_by_name(variable)
    data = np.asarray(child[subset.selection], dtype=float)
    return data.reshape(-1)[subset.mask.reshape(-1)]


def read_zambia_spatial_values(
    *,
    model: str,
    run_time: dt.datetime,
    valid_time: dt.datetime,
    variables: tuple[str, ...],
    bbox: tuple[float, float, float, float],
    cache_storage: str,
    block_size: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    uri = build_spatial_uri(model, run_time, valid_time)
    backend = fsspec.open(
        f"blockcache::{uri}",
        mode="rb",
        s3={"anon": True, "default_block_size": block_size},
        blockcache={"cache_storage": cache_storage},
    )

    with OmFileReader(backend) as reader:
        first_child = reader.get_child_by_name(variables[0])
        num_y, num_x = first_child.shape
        grid = OmGrid(reader.get_child_by_name("crs_wkt").read_scalar(), (num_y, num_x))
        subset = build_grid_subset(grid, bbox)
        values = {variable: read_subset_variable(reader, variable, subset) for variable in variables}

    return values, subset.lon, subset.lat


def read_zambia_mean_values(
    *,
    model: str,
    run_time: dt.datetime,
    valid_time: dt.datetime,
    variables: tuple[str, ...],
    bbox: tuple[float, float, float, float],
    cache_storage: str,
    block_size: int,
) -> dict[str, float]:
    values, _lon, _lat = read_zambia_spatial_values(
        model=model,
        run_time=run_time,
        valid_time=valid_time,
        variables=variables,
        bbox=bbox,
        cache_storage=cache_storage,
        block_size=block_size,
    )
    return {variable: float(np.nanmean(data)) for variable, data in values.items()}


def load_ecmwf_zambia_weather(args: argparse.Namespace) -> dict[str, np.ndarray]:
    weather, _lon, _lat = load_ecmwf_zambia_spatial_weather(args)
    return {key: np.nanmean(value, axis=1) if key != "time" and np.ndim(value) == 2 else value for key, value in weather.items()}


def load_ecmwf_zambia_spatial_weather(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    metadata = load_latest_metadata(args.model)
    if not metadata.get("completed", False):
        raise RuntimeError(f"Latest spatial run for {args.model!r} is not complete.")
    available = set(metadata.get("variables", []))
    variables = tuple(variable for variable in ECMWF_VARIABLES if variable in available)
    missing = sorted(set(ECMWF_VARIABLES) - set(variables))
    if missing:
        print(f"Skipping variables not in latest metadata: {', '.join(missing)}")

    run_time = parse_utc_time(metadata["reference_time"])
    valid_times = choose_daily_valid_times(metadata, args.max_days)
    rows = []
    lon = None
    lat = None
    for valid_time in valid_times:
        print(f"Reading {args.model} {valid_time.strftime('%Y-%m-%dT%H:%MZ')} over Zambia bbox", flush=True)
        row, row_lon, row_lat = read_zambia_spatial_values(
            model=args.model,
            run_time=run_time,
            valid_time=valid_time,
            variables=variables,
            bbox=tuple(args.bbox),
            cache_storage=args.cache_storage,
            block_size=args.block_size,
        )
        rows.append(row)
        if lon is None:
            lon = row_lon
            lat = row_lat

    weather = {"time": np.array([value.date().isoformat() for value in valid_times], dtype="datetime64[D]")}
    for variable in variables:
        data = np.stack([row[variable] for row in rows], axis=0)
        if variable == "precipitation":
            data = np.maximum(data, 0.0)
        weather[variable] = data
    if args.clip_country:
        geometry = load_country_geometry(args.country)
        country_mask = mask_points_to_geometry(lon, lat, geometry)
        weather, lon, lat = apply_point_mask(weather, lon, lat, country_mask)
    return weather, lon, lat


def load_country_geometry(country: str) -> BaseGeometry:
    shp_path = shapereader.natural_earth(
        resolution="10m",
        category="cultural",
        name="admin_0_countries",
    )
    for record in shapereader.Reader(shp_path).records():
        attrs = record.attributes
        names = {
            attrs.get("ADMIN"),
            attrs.get("NAME"),
            attrs.get("NAME_LONG"),
            attrs.get("SOVEREIGNT"),
            attrs.get("BRK_NAME"),
        }
        if country in names:
            return record.geometry
    raise ValueError(f"Could not find {country!r} in Natural Earth admin_0_countries.")


def load_neighbor_geometries(bbox: tuple[float, float, float, float]) -> list[BaseGeometry]:
    shp_path = shapereader.natural_earth(
        resolution="10m",
        category="cultural",
        name="admin_0_countries",
    )
    lon_min, lon_max, lat_min, lat_max = bbox
    geometries = []
    for record in shapereader.Reader(shp_path).records():
        minx, miny, maxx, maxy = record.geometry.bounds
        if maxx >= lon_min and minx <= lon_max and maxy >= lat_min and miny <= lat_max:
            geometries.append(record.geometry)
    return geometries


def mask_points_to_geometry(lon: np.ndarray, lat: np.ndarray, geometry: BaseGeometry) -> np.ndarray:
    return np.asarray(contains_xy(geometry, np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)), dtype=bool)


def apply_point_mask(
    weather: dict[str, np.ndarray],
    lon: np.ndarray,
    lat: np.ndarray,
    mask: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    clipped: dict[str, np.ndarray] = {"time": weather["time"]}
    for key, value in weather.items():
        if key == "time":
            continue
        arr = np.asarray(value)
        if arr.ndim >= 2 and arr.shape[1] == mask.shape[0]:
            clipped[key] = arr[:, mask]
        else:
            clipped[key] = arr
    return clipped, lon[mask], lat[mask]


def professional_map(
    *,
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    title: str,
    subtitle: str,
    colorbar_label: str,
    output: Path,
    bbox: tuple[float, float, float, float],
    cmap: str,
    country_geometry: BaseGeometry,
    neighbor_geometries: list[BaseGeometry],
    value_range: tuple[float, float] | None = None,
    nonnegative: bool = False,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError(f"No finite values available for plot: {title}")

    plot_lon = lon[finite]
    plot_lat = lat[finite]
    plot_values = values[finite]
    if nonnegative:
        plot_values = np.maximum(plot_values, 0.0)
    vmin, vmax = professional_value_range(plot_values, value_range, nonnegative=nonnegative)

    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(10.5, 9.0), facecolor="white")
    ax = plt.axes(projection=projection)
    padded_bbox = pad_bbox(bbox, PLOT_BBOX_PADDING)
    ax.set_extent(padded_bbox, crs=projection)
    ax.set_facecolor("#eef3f7")

    ax.add_geometries(
        neighbor_geometries,
        crs=projection,
        facecolor="#f4f1e8",
        edgecolor="#c9c5bb",
        linewidth=0.65,
        zorder=1,
    )
    ax.add_geometries(
        [country_geometry],
        crs=projection,
        facecolor="#fbfaf6",
        edgecolor="#30343b",
        linewidth=1.35,
        zorder=2,
    )

    mappable = draw_surface(
        ax=ax,
        lon=plot_lon,
        lat=plot_lat,
        values=plot_values,
        geometry=country_geometry,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        transform=projection,
    )

    ax.add_geometries(
        [country_geometry],
        crs=projection,
        facecolor="none",
        edgecolor="#14171c",
        linewidth=1.6,
        zorder=8,
    )

    gl = ax.gridlines(
        crs=projection,
        draw_labels=True,
        linewidth=0.45,
        color="#9aa4ad",
        alpha=0.42,
        linestyle="-",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 9, "color": "#48525c"}
    gl.ylabel_style = {"size": 9, "color": "#48525c"}

    title_text = ax.text(
        0.0,
        1.075,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=18,
        fontweight="bold",
        color="#151922",
    )
    title_text.set_path_effects([path_effects.withStroke(linewidth=3, foreground="white")])
    ax.text(
        0.0,
        1.032,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        color="#56616d",
    )

    cbar = fig.colorbar(mappable, ax=ax, orientation="vertical", shrink=0.76, pad=0.025)
    cbar.set_label(colorbar_label, fontsize=10.5, color="#29313a")
    cbar.ax.tick_params(labelsize=9, colors="#39434d")
    cbar.outline.set_edgecolor("#b6bec6")

    ax.text(
        0.0,
        -0.075,
        "Source: Open-Meteo spatial ECMWF IFS; boundary: Natural Earth. Values clipped to Zambia.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#6b7480",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def professional_value_range(
    values: np.ndarray,
    value_range: tuple[float, float] | None,
    *,
    nonnegative: bool = False,
) -> tuple[float, float]:
    if value_range is not None:
        return value_range
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    if nonnegative:
        vmax = float(np.nanpercentile(finite, 98))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = max(float(np.nanmax(finite)), 0.1)
        return 0.0, vmax
    vmin, vmax = np.nanpercentile(finite, [2, 98])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        center = float(np.nanmean(finite)) if finite.size else 0.0
        spread = max(abs(center) * 0.05, 0.5)
        return center - spread, center + spread
    return float(vmin), float(vmax)


def pad_bbox(bbox: tuple[float, float, float, float], padding: float) -> tuple[float, float, float, float]:
    return (bbox[0] - padding, bbox[1] + padding, bbox[2] - padding, bbox[3] + padding)


def draw_surface(
    *,
    ax,
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    geometry: BaseGeometry,
    cmap: str,
    vmin: float,
    vmax: float,
    transform,
):
    if lon.size >= 3 and np.nanstd(values) > 1e-12:
        triangulation = mtri.Triangulation(lon, lat)
        triangles = triangulation.triangles
        centroid_lon = lon[triangles].mean(axis=1)
        centroid_lat = lat[triangles].mean(axis=1)
        outside = ~mask_points_to_geometry(centroid_lon, centroid_lat, geometry)
        triangulation.set_mask(outside)
        levels = np.linspace(vmin, vmax, 15)
        return ax.tricontourf(
            triangulation,
            values,
            levels=levels,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extend="both",
            transform=transform,
            zorder=4,
            antialiased=True,
        )

    marker_size = max(4.0, min(22.0, 35000.0 / max(lon.size, 1)))
    return ax.scatter(
        lon,
        lat,
        c=values,
        s=marker_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
        transform=transform,
        zorder=4,
    )


def plot_style_for(name: str) -> tuple[str, str, tuple[float, float] | None, bool]:
    if "stress" in name or "drought" in name or "advisory" in name:
        return "YlOrRd", "score (0-1)", (0.0, 1.0), True
    if "rain" in name or "precipitation" in name:
        return "YlGnBu", "mm", None, True
    if name == "gdd":
        return "YlOrBr", "degree days", None, True
    return "viridis", name, None, False


def model_label(model: str) -> str:
    labels = {"ecmwf_ifs": "ECMWF IFS"}
    return labels.get(model, model.replace("_", " ").upper())


def plot_test_outputs(
    *,
    args: argparse.Namespace,
    weather: dict[str, np.ndarray],
    result: dict,
    lon: np.ndarray,
    lat: np.ndarray,
) -> list[Path]:
    output_dir = Path(args.output_dir)
    bbox = tuple(args.bbox)
    outputs = []
    country_geometry = load_country_geometry(args.country)
    neighbor_geometries = load_neighbor_geometries(pad_bbox(bbox, PLOT_BBOX_PADDING))

    valid_date = str(weather["time"][0])
    if args.plot_variable in weather:
        cmap, label, value_range, nonnegative = plot_style_for(args.plot_variable)
        outputs.append(
            professional_map(
                lon=lon,
                lat=lat,
                values=weather[args.plot_variable][0],
                title=f"{args.plot_variable.replace('_', ' ').title()}",
                subtitle=f"{model_label(args.model)} over Zambia | Valid {valid_date} | {lon.size:,} grid cells",
                colorbar_label=label,
                output=output_dir / f"{args.model}_zambia_{args.plot_variable}_{valid_date}.png",
                bbox=bbox,
                cmap=args.raw_cmap or cmap,
                country_geometry=country_geometry,
                neighbor_geometries=neighbor_geometries,
                value_range=value_range,
                nonnegative=nonnegative,
            )
        )

    index_values = {
        "rainfall_total": result["crop"]["rainfall_total"],
        "gdd": result["crop"]["growing_degree_days"],
        "water_stress_index": result["crop"]["water_stress_index"]["mean"],
        "pasture_drought_score": result["livestock"]["pasture_drought_index"]["score"],
        "seasonal_advisory_score": result["integrated"]["seasonal_advisory_class"]["score"],
    }
    for name in args.plot_indices:
        if name not in index_values:
            raise ValueError(f"Unknown plot index {name!r}. Available: {', '.join(index_values)}")
        cmap, label, value_range, nonnegative = plot_style_for(name)
        outputs.append(
            professional_map(
                lon=lon,
                lat=lat,
                values=index_values[name],
                title=name.replace("_", " ").title(),
                subtitle=f"{model_label(args.model)} agro-climate index over Zambia | {args.max_days} forecast day(s) | {lon.size:,} grid cells",
                colorbar_label=label,
                output=output_dir / f"{args.model}_zambia_index_{name}.png",
                bbox=bbox,
                cmap=args.index_cmap or cmap,
                country_geometry=country_geometry,
                neighbor_geometries=neighbor_geometries,
                value_range=value_range,
                nonnegative=nonnegative,
            )
        )

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test agro-climate indices with ECMWF IFS over Zambia.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--country", default="Zambia")
    parser.add_argument("--clip-country", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-days", type=int, default=5)
    parser.add_argument("--bbox", type=float, nargs=4, default=ZAMBIA_BBOX)
    parser.add_argument("--cache-storage", default=".openmeteo_cache")
    parser.add_argument("--block-size", type=int, default=65536)
    parser.add_argument(
        "--output-dir",
        default=str(country_root("zambia") / "smoke_tests" / "ecmwf_ifs"),
    )
    parser.add_argument("--plot-variable", default="precipitation")
    parser.add_argument(
        "--plot-indices",
        nargs="+",
        default=["rainfall_total", "water_stress_index", "seasonal_advisory_score"],
    )
    parser.add_argument("--raw-cmap", default=None)
    parser.add_argument("--index-cmap", default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weather, lon, lat = load_ecmwf_zambia_spatial_weather(args)
    cfg = AgroClimateConfig(latitude=(args.bbox[2] + args.bbox[3]) / 2.0, elevation_m=1100.0)
    result = compute_agroclimate_indices(weather, config=cfg)
    summary = summarize_indices(result)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{args.model}_zambia_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")

    print(json.dumps(summary, indent=2, default=str))
    print(f"Summary saved to: {summary_path}")
    if not args.no_plots:
        for output in plot_test_outputs(args=args, weather=weather, result=result, lon=lon, lat=lat):
            print(f"Plot saved to: {output}")


if __name__ == "__main__":
    main()
