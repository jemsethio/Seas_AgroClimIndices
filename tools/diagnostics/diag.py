"""Quick diagnostic: load one member from each variable and run indices."""
import sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "agroclimate_indices"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import xarray as xr
from cds_agroclimate_pipeline.cds.compute_indices import (
    extract_step_hours, aggregate_instantaneous_daily,
    aggregate_accumulated_daily, aggregate_daily_extreme,
    CDS_RAW_VARIABLE_CONFIG, CDS_DERIVED_VARIABLE_CONFIG,
    build_weather_dict_for_member,
)
from agroclimate_indices import AgroClimateConfig, compute_agroclimate_indices

MODEL_DIR = (
    PROJECT_ROOT
    / "data/countries/ethiopia/seasonal/cds"
    / "seasonal-original-single-levels/2026/05/01/ecmwf_system51"
)
MEMBER = 0

print("=== Diagnostic: load member 0, run indices ===\n")

variables = {}
lats = lons = None

# --- Load raw variables one at a time (member 0 only) ---
for stem, spec in CDS_RAW_VARIABLE_CONFIG.items():
    nc = f'{MODEL_DIR}/{stem}.nc'
    t0 = time.time()
    try:
        ds = xr.open_dataset(nc, engine='netcdf4')
        for dim in ('forecast_reference_time',):
            if dim in ds.dims and ds.sizes[dim] == 1:
                ds = ds.squeeze(dim, drop=True)

        step_hours = extract_step_hours(ds)
        var_name = list(ds.data_vars)[0]
        da = ds[var_name]
        step_dim = next(
            (d for d in ('forecast_period', 'step', 'leadtime_hour') if d in da.dims), None
        )
        if step_dim is None:
            print(f"  SKIP {stem} — no step dim, dims={da.dims}")
            ds.close()
            continue

        da = da.transpose('number', step_dim, 'latitude', 'longitude')
        da_m = da.isel(number=MEMBER)        # (step, lat, lon) — one member only
        arr_2d = da_m.values.astype(np.float32)

        if lats is None:
            lats = da['latitude'].values.astype(np.float64)
            lons = da['longitude'].values.astype(np.float64)
        ds.close()

        # Apply scale + offset
        if spec.scale != 1.0:
            arr_2d *= float(spec.scale)
        if spec.offset != 0.0:
            arr_2d += float(spec.offset)

        # Aggregate to daily — wrap in fake member dim (1, step, lat, lon)
        arr = arr_2d[np.newaxis, ...]
        if spec.agg_type == 'accumulated':
            daily = aggregate_accumulated_daily(arr, step_hours)
        elif spec.agg_type == 'instantaneous':
            daily = aggregate_instantaneous_daily(arr, step_hours)
        else:
            daily = aggregate_daily_extreme(arr, step_hours)

        daily_slice = daily[0]   # (n_days, lat, lon)
        variables[spec.weather_key] = daily_slice
        print(f"  OK  {stem[:42]:<42} → {spec.weather_key:<26} shape={daily_slice.shape} [{spec.unit}] {time.time()-t0:.1f}s")

    except Exception as exc:
        print(f"  ERR {stem}: {exc}")

# --- Load derived variables ---
derived_dir = f'{MODEL_DIR}/derived'
for stem, dspec in CDS_DERIVED_VARIABLE_CONFIG.items():
    nc = f'{derived_dir}/{stem}.nc'
    try:
        ds = xr.open_dataset(nc, engine='netcdf4')
        for dim in ('forecast_reference_time',):
            if dim in ds.dims and ds.sizes[dim] == 1:
                ds = ds.squeeze(dim, drop=True)
        var_name = list(ds.data_vars)[0]
        da = ds[var_name]
        step_dim = next(
            (d for d in ('forecast_period', 'step', 'leadtime_hour') if d in da.dims), None
        )
        da_m = da.isel(number=MEMBER) if 'number' in da.dims else da
        da_m = da_m.transpose(*(d for d in (step_dim, 'latitude', 'longitude') if d in da_m.dims))
        arr_2d = da_m.values.astype(np.float32)
        ds.close()
        arr = arr_2d[np.newaxis, ...]
        daily = aggregate_instantaneous_daily(arr, extract_step_hours(
            xr.Dataset({step_dim: da_m[step_dim]}) if step_dim else xr.Dataset()
        ))
        daily_slice = daily[0]
        variables[dspec.weather_key] = daily_slice
        print(f"  OK  derived/{stem:<38} shape={daily_slice.shape} [{dspec.unit}]")
    except Exception as exc:
        print(f"  SKIP derived/{stem}: {exc}")

print(f"\n--- Loaded variables: {sorted(variables.keys())} ---")
print(f"    Grid: lat={lats.shape}, lon={lons.shape}")

# --- Build weather dict and run indices ---
# Wrap variables as (1, days, lat, lon) for compatibility
wrapped = {k: v[np.newaxis, ...] for k, v in variables.items()}

def _get_m(key, m=0, n_days=215):
    arr = wrapped.get(key)
    if arr is None:
        return None
    return arr[0, :n_days, :, :].astype(np.float64)

n_days = min(v.shape[0] for v in variables.values())
print(f"    n_days={n_days}")

weather = {}
for k, v in variables.items():
    if k.startswith('_'):
        continue
    weather[k] = v[:n_days].astype(np.float64)

# Derive wind from u/v if needed
if 'wind_speed_10m' not in weather and '_u10' in variables and '_v10' in variables:
    u = variables['_u10'][:n_days].astype(float)
    v = variables['_v10'][:n_days].astype(float)
    weather['wind_speed_10m'] = np.sqrt(u**2 + v**2)
    print("  Derived wind_speed_10m from u10+v10")

# Derive RH from dew point if needed
if 'relative_humidity_2m' not in weather and '_dew_point_degc' in variables:
    t_c = weather.get('temperature_2m_mean')
    d_c = variables['_dew_point_degc'][:n_days].astype(float)
    if t_c is not None:
        es = 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))
        ea = 0.6108 * np.exp(17.27 * d_c / (d_c + 237.3))
        weather['relative_humidity_2m'] = np.clip(100.0 * ea / es, 0, 100)
        print("  Derived relative_humidity_2m from dew point")

print(f"\nWeather dict keys: {sorted(weather.keys())}")
for k, v in weather.items():
    print(f"  {k:<30}  shape={v.shape}  min={np.nanmin(v):.2f}  max={np.nanmax(v):.2f}")

lat_2d = np.tile(lats[:, np.newaxis], (1, lons.size))
cfg = AgroClimateConfig(latitude=lat_2d, elevation_m=500.0, shortwave_radiation_unit='mj_m2_day')

print("\n--- Running compute_agroclimate_indices ---")
t0 = time.time()
result = compute_agroclimate_indices(weather, config=cfg)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s")
print(f"Crop keys:      {sorted(result['crop'].keys())[:5]} ...")
print(f"Advisory score: {np.nanmean(result['integrated']['seasonal_advisory_class']['score']):.3f}")
print("\n=== DIAGNOSTIC PASSED ===")
