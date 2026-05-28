"""
Explore all earthkit package APIs relevant to the CDS pipeline.
Run with:  uv run python tools/diagnostics/explore_earthkit.py
"""
import sys, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(
    PROJECT_ROOT
    / "data/countries/ethiopia/seasonal/cds"
    / "seasonal-original-single-levels/2026/05/01/ecmwf_system51"
)
NC_T2M  = MODEL_DIR / "2m_temperature.nc"
NC_TP   = MODEL_DIR / "total_precipitation.nc"
NC_U10  = MODEL_DIR / "10m_u_component_of_wind.nc"

sep = lambda t: print(f"\n{'='*60}\n{t}\n{'='*60}")

# ──────────────────────────────────────────────────────────────
# 1. earthkit-data — read existing NetCDF files
# ──────────────────────────────────────────────────────────────
sep("1. earthkit-data")
import earthkit.data as ekd
print("version:", ekd.__version__)

# Read one variable file
ds = ekd.from_source("file", str(NC_T2M))
print("Type:", type(ds))
print("Len (fields):", len(ds))
print("First field metadata:", ds[0].metadata())

# Convert to xarray
xr_ds = ds.to_xarray()
print("Xarray dims:", dict(xr_ds.dims))
print("Xarray vars:", list(xr_ds.data_vars))

# ──────────────────────────────────────────────────────────────
# 2. earthkit-transforms — temporal aggregation
# ──────────────────────────────────────────────────────────────
sep("2. earthkit-transforms")
import earthkit.transforms
print("version:", earthkit.transforms.__version__)

from earthkit.transforms.aggregate import temporal as ekt_temporal
# What functions exist?
fns = [f for f in dir(ekt_temporal) if not f.startswith("_")]
print("temporal functions:", fns)

# Try daily_mean on a small xarray slice (member 0, first 4 steps = 1 day)
import xarray as xr
t2m_xr = xr_ds[list(xr_ds.data_vars)[0]].isel(
    forecast_reference_time=0, number=0
)  # (forecast_period, lat, lon)
print("t2m_xr shape:", t2m_xr.shape, "dims:", t2m_xr.dims)
print("forecast_period dtype:", t2m_xr.forecast_period.values.dtype)

# Build absolute valid_time coordinate for resampling
ref_time = xr_ds.forecast_reference_time.values[0]
fp = t2m_xr.forecast_period.values   # timedelta64
valid_times = ref_time + fp
t2m_time = t2m_xr.assign_coords(forecast_period=valid_times).rename(
    {"forecast_period": "time"}
)
print("t2m_time shape:", t2m_time.shape, "time dtype:", t2m_time.time.values.dtype)

# Try resample via earthkit-transforms
try:
    daily = ekt_temporal.daily_mean(t2m_time, time_dim="time")
    print("daily_mean output shape:", daily.shape)
except Exception as e:
    print("daily_mean error:", e)
    # Try alternative
    try:
        daily = t2m_time.resample(time="1D").mean()
        print("xarray resample fallback shape:", daily.shape)
    except Exception as e2:
        print("xarray resample error:", e2)

# Try resample_and_reduce
try:
    fn = getattr(ekt_temporal, "resample_and_reduce", None)
    if fn:
        daily2 = fn(t2m_time, "1D", how="mean")
        print("resample_and_reduce shape:", daily2.shape)
except Exception as e:
    print("resample_and_reduce:", e)

# ──────────────────────────────────────────────────────────────
# 3. earthkit-meteo — met formulas (already using)
# ──────────────────────────────────────────────────────────────
sep("3. earthkit-meteo")
import earthkit.meteo as ekm
print("version:", ekm.__version__)
import earthkit.meteo.thermo as th
import earthkit.meteo.wind as wnd

t_k = np.array([300.0, 295.0])
td_k = np.array([290.0, 285.0])
print("RH(%):", th.relative_humidity_from_dewpoint(t_k, td_k))
print("SVP(Pa):", th.saturation_vapour_pressure(t_k))

# Check if potential_evaporation exists
et_fns = [f for f in dir(ekm) if "evap" in f.lower() or "et" in f.lower()]
print("ET-related functions:", et_fns)

# Check thermo for ET
et_thermo = [f for f in dir(th) if "evap" in f.lower() or "et0" in f.lower()]
print("thermo ET functions:", et_thermo)

# ──────────────────────────────────────────────────────────────
# 4. earthkit-hydro — hydrological indices
# ──────────────────────────────────────────────────────────────
sep("4. earthkit-hydro")
import earthkit.hydro as ekh
print("version:", ekh.__version__)
print("hydro submodules:", [x for x in dir(ekh) if not x.startswith("_")])

# Explore ET0
try:
    from earthkit.hydro import evapotranspiration as et
    print("ET functions:", [f for f in dir(et) if not f.startswith("_")])
except ImportError as e:
    print("evapotranspiration:", e)

# Explore drought indices
try:
    from earthkit.hydro import drought
    print("drought functions:", [f for f in dir(drought) if not f.startswith("_")])
except ImportError as e:
    print("drought:", e)

# Explore all submodules
for mod_name in [x for x in dir(ekh) if not x.startswith("_")]:
    try:
        mod = getattr(ekh, mod_name)
        fns = [f for f in dir(mod) if not f.startswith("_") and callable(getattr(mod, f, None))]
        if fns:
            print(f"  hydro.{mod_name}: {fns[:8]}")
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────
# 5. earthkit-geo — geographic operations
# ──────────────────────────────────────────────────────────────
sep("5. earthkit-geo")
import earthkit.geo as ekg
print("version:", ekg.__version__)
print("geo submodules:", [x for x in dir(ekg) if not x.startswith("_")])

for mod_name in [x for x in dir(ekg) if not x.startswith("_")]:
    try:
        mod = getattr(ekg, mod_name)
        fns = [f for f in dir(mod) if not f.startswith("_")]
        if fns:
            print(f"  geo.{mod_name}: {fns[:6]}")
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────
# 6. earthkit-plots — visualisation
# ──────────────────────────────────────────────────────────────
sep("6. earthkit-plots")
import earthkit.plots as ekp
print("version:", ekp.__version__)
print("plots API:", [x for x in dir(ekp) if not x.startswith("_")])

# Check Chart/Figure API
for cls in ["Chart", "Figure", "Map", "quickplot", "plot"]:
    obj = getattr(ekp, cls, None)
    if obj:
        print(f"  ekp.{cls}: {type(obj)}")
        if hasattr(obj, "__init__"):
            import inspect
            try:
                sig = inspect.signature(obj)
                print(f"    signature: {sig}")
            except Exception:
                pass

print("\n=== EXPLORATION COMPLETE ===")
