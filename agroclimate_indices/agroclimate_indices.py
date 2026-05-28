from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Union

import numpy as np


ArrayLike = Any
Weather = Mapping[str, ArrayLike]
Climatology = Mapping[str, ArrayLike]


REQUIRED_VARIABLES: dict[str, dict[str, list[str]]] = {
    "precipitation": {
        "used_for": [
            "rainfall_total",
            "anomaly",
            "onset",
            "dry_spell",
            "wet_spell",
            "SPI",
            "pasture_growth",
            "surface_water_proxy",
        ]
    },
    "temperature_2m_max": {
        "used_for": ["heat_stress_days", "GDD", "THI", "livestock_heat_stress"]
    },
    "temperature_2m_min": {
        "used_for": ["GDD", "cold_stress", "night_recovery_index"]
    },
    "temperature_2m_mean": {
        "used_for": ["SPEI", "PET", "THI", "crop_livestock_stress"]
    },
    "relative_humidity_2m": {
        "used_for": ["THI", "disease_vector_suitability"]
    },
    "wind_speed_10m": {
        "used_for": ["ET0", "wind_risk", "wind_chill", "dust_risk"]
    },
    "shortwave_radiation": {
        "used_for": ["ET0", "solar_radiation_anomaly", "crop_growth_potential"]
    },
    "surface_pressure": {"used_for": ["FAO56_ET0"]},
    "soil_moisture": {
        "used_for": ["soil_moisture_anomaly", "pasture_drought", "crop_water_stress"]
    },
    "elevation": {
        "used_for": [
            "temperature_lapse_rate_correction",
            "agroecological_zone_adjustment",
            "highland_cold_risk",
        ]
    },
}


INDEX_DEFINITIONS: dict[str, dict[str, str]] = {
    "rainfall_total": {
        "equation": "Psum = sum(P)",
        "interpretation": "Seasonal or monthly rainfall amount.",
    },
    "rainfall_anomaly": {
        "equation": "Panom = Pforecast - Pclim",
        "interpretation": "Positive is wetter than normal; negative is drier.",
    },
    "rainfall_anomaly_pct": {
        "equation": "100 * (Pforecast - Pclim) / Pclim",
        "interpretation": "Percent departure from normal rainfall.",
    },
    "tercile_class": {
        "equation": "Compare forecast rainfall to 33rd and 66th climatological percentiles.",
        "interpretation": "BN, N, or AN seasonal rainfall class.",
    },
    "spi": {
        "equation": "Standardized rolling precipitation accumulation.",
        "interpretation": "Meteorological drought or wetness index.",
    },
    "spei": {
        "equation": "Standardized rolling (P - PET) accumulation.",
        "interpretation": "Drought or wetness including atmospheric water demand.",
    },
    "onset": {
        "equation": "First wet window meeting rainfall threshold and not followed by a long dry spell.",
        "interpretation": "Likely planting window start.",
    },
    "false_start_risk": {
        "equation": "Wet onset condition followed by a long dry spell within a look-ahead window.",
        "interpretation": "Planting failure risk.",
    },
    "cessation": {
        "equation": "Last date with rainfall or water balance support after onset.",
        "interpretation": "End of effective growing season.",
    },
    "lgp": {
        "equation": "LGP = cessation - onset",
        "interpretation": "Length of growing period.",
    },
    "gdd": {
        "equation": "sum(max(((Tmax + Tmin) / 2) - Tbase, 0))",
        "interpretation": "Crop phenology and maturity heat units.",
    },
    "et0": {
        "equation": "FAO-56 Penman-Monteith daily ET0.",
        "interpretation": "Reference evapotranspiration and atmospheric water demand.",
    },
    "thi": {
        "equation": "THI = 0.8T + RH(T - 14.4) + 46.4, RH as fraction.",
        "interpretation": "Livestock heat stress screening.",
    },
    "water_stress_index": {
        "equation": "WSI = 1 - AET / ETc",
        "interpretation": "0 is no water stress; values near 1 are severe stress.",
    },
}


@dataclass(frozen=True)
class AgroClimateConfig:
    """Thresholds and assumptions used by compute_agroclimate_indices.

    All weather arrays are expected to have time on axis 0. Daily inputs are
    the default. Monthly inputs are supported for SPI/SPEI by setting
    frequency="monthly".
    """

    frequency: str = "daily"
    latitude: Optional[Union[float, ArrayLike]] = None
    elevation_m: Union[float, ArrayLike] = 0.0
    albedo: float = 0.23
    shortwave_radiation_unit: str = "auto"  # "auto", "mj_m2_day", or "w_m2"

    crop_tbase_c: float = 10.0
    crop_heat_threshold_c: float = 35.0
    crop_cold_threshold_c: float = 5.0
    kc: Union[float, ArrayLike] = 1.0
    actual_et_key: str = "actual_evapotranspiration"

    dry_day_mm: float = 1.0
    wet_day_mm: float = 1.0
    extreme_rain_thresholds_mm: tuple[float, ...] = (20.0, 50.0)
    onset_window_days: int = 3
    onset_total_rain_mm: float = 20.0
    onset_min_wet_days: int = 2
    false_start_dry_spell_days: int = 7
    false_start_lookahead_days: int = 30
    cessation_window_days: int = 10
    cessation_rain_mm: float = 10.0
    cessation_water_balance_mm: float = 0.0

    wind_risk_threshold_m_s: float = 10.0
    livestock_cold_threshold_c: float = 5.0
    night_tmin_recovery_threshold_c: float = 24.0
    night_thi_recovery_threshold: float = 68.0
    thi_mild_threshold: float = 68.0
    thi_moderate_threshold: float = 72.0
    thi_severe_threshold: float = 78.0
    thi_emergency_threshold: float = 84.0
    baseline_livestock_water_l_day: float = 50.0

    pasture_temperature_min_c: float = 10.0
    pasture_temperature_optimum_c: float = 25.0
    pasture_temperature_max_c: float = 35.0
    pasture_target_rain_30d_mm: float = 75.0
    soil_moisture_wilting: Union[float, ArrayLike] = 0.10
    soil_moisture_field_capacity: Union[float, ArrayLike] = 0.35

    rainfall_windows_days: tuple[int, ...] = (10, 30, 90)
    standardization_scales: tuple[int, ...] = (1, 3, 6)
    month_days: int = 30

    crop_water_stress_weight: float = 1.0
    pasture_stress_weight: float = 1.0
    livestock_heat_stress_weight: float = 1.0
    cropping_land_pressure: float = 0.5
    livestock_units: float = 1.0
    weights: dict[str, float] = field(default_factory=dict)


def compute_agroclimate_indices(
    weather: Weather,
    climatology: Optional[Climatology] = None,
    config: Optional[AgroClimateConfig] = None,
) -> dict[str, Any]:
    """Compute crop, livestock, and integrated agro-climate indices.

    Parameters
    ----------
    weather:
        Mapping of variable name to arrays. The first axis must be time. The
        function accepts point series shaped (time,) and grids shaped
        (time, y, x). Required core variable is precipitation. Other indices
        are computed when the variables needed for them are present.
    climatology:
        Optional climatological means, standard deviations, percentiles, or
        time series. Without climatology, anomaly and standardized indices are
        returned as unavailable rather than guessed.
    config:
        Thresholds and crop/livestock assumptions.

    Returns
    -------
    dict
        A nested dict with "crop", "livestock", "integrated", "availability",
        and "definitions" keys. Values are numpy arrays when the input is
        gridded.
    """

    cfg = config or AgroClimateConfig()
    clim = climatology or {}

    p = np.maximum(_require(weather, "precipitation"), 0.0)
    time = _time_axis(weather, p.shape[0])
    spatial_shape = p.shape[1:]

    tmax = _temperature_max(weather)
    tmin = _temperature_min(weather)
    tmean = _temperature_mean(weather, tmax, tmin)
    rh = _relative_humidity(weather, tmean)
    wind10 = _wind_speed_10m(weather)
    soil_moisture = _soil_moisture(weather)
    shortwave = _array_or_none(weather, "shortwave_radiation")
    pressure = _surface_pressure_kpa(weather, cfg)
    latitude = _broadcast_param(cfg.latitude, spatial_shape, "latitude")
    elevation = _broadcast_param(_value_or_weather(weather, "elevation", cfg.elevation_m), spatial_shape, "elevation")

    availability: dict[str, Any] = {
        "input_variables": sorted(k for k in weather.keys() if k != "time"),
        "missing_variables": _missing_variables(weather),
        "notes": [],
    }

    rainfall_total = np.nansum(p, axis=0)
    rainfall_clim_total = _climatology_total(clim, "precipitation")
    rainfall_anomaly = _anomaly(rainfall_total, rainfall_clim_total)
    rainfall_anomaly_pct = _percent_anomaly(rainfall_total, rainfall_clim_total)
    tercile = _tercile_class(rainfall_total, clim)

    if rainfall_clim_total is None:
        availability["notes"].append("Rainfall anomaly needs climatological precipitation total.")
    if tercile is None:
        availability["notes"].append("Tercile class needs precipitation p33 and p66 climatology.")

    et0, et0_meta = _reference_et0_fao56(
        time=time,
        tmax=tmax,
        tmin=tmin,
        tmean=tmean,
        rh=rh,
        wind10=wind10,
        shortwave=shortwave,
        pressure_kpa=pressure,
        latitude=latitude,
        elevation=elevation,
        cfg=cfg,
    )
    availability["et0_method"] = et0_meta

    etc = _crop_et(et0, cfg.kc)
    water_balance_et0 = _safe_subtract(p, et0)
    water_balance_etc = _safe_subtract(p, etc)
    aet, aet_source = _actual_et(weather, p, etc, cfg)
    water_stress_index = _water_stress_index(aet, etc)
    availability["actual_et_source"] = aet_source

    spi = _standardized_indices(
        values=p,
        base_key="precipitation",
        climatology=clim,
        cfg=cfg,
        output_name="spi",
    )
    spei = _standardized_indices(
        values=water_balance_et0,
        base_key="water_balance",
        climatology=clim,
        cfg=cfg,
        output_name="spei",
    )

    onset = _onset_and_false_start(p, time, cfg)
    cessation = _cessation(p, water_balance_etc, time, onset["onset_index"], cfg)
    lgp_days = _lgp_days(onset["onset_index"], cessation["cessation_index"])

    dry_spell = _spell_summary(p < cfg.dry_day_mm)
    wet_spell = _spell_summary(p >= cfg.wet_day_mm)
    extreme_rain = _extreme_rainfall(p, clim, cfg)

    dtr = _safe_subtract(tmax, tmin)
    gdd = _gdd(tmax, tmin, cfg.crop_tbase_c)
    crop_heat = _spell_summary(tmax >= cfg.crop_heat_threshold_c) if tmax is not None else _unavailable()
    crop_cold = _spell_summary(tmin <= cfg.crop_cold_threshold_c) if tmin is not None else _unavailable()

    soil_moisture_mean = np.nanmean(soil_moisture, axis=0) if soil_moisture is not None else None
    soil_moisture_anomaly = _anomaly(
        soil_moisture_mean,
        _clim_lookup(clim, "soil_moisture_mean", "soil_moisture"),
    )
    solar_radiation_total = np.nansum(shortwave, axis=0) if shortwave is not None else None
    solar_radiation_anomaly = _anomaly(
        solar_radiation_total,
        _clim_lookup(clim, "shortwave_radiation_total", "shortwave_radiation_mean", "shortwave_radiation"),
    )
    wind_risk = _spell_summary(wind10 >= cfg.wind_risk_threshold_m_s) if wind10 is not None else _unavailable()

    thi = _thi(tmean, rh)
    thi_summary = _thi_summary(thi, cfg)
    heat_duration = (
        _spell_summary(thi >= cfg.thi_moderate_threshold) if thi is not None else _unavailable()
    )
    night_recovery = _night_recovery(tmin, thi, cfg)
    water_demand = _water_demand_index(thi, cfg)
    pasture_rain = _pasture_rainfall_index(p, clim, cfg)
    forage_growth = _forage_growth_proxy(p, rainfall_anomaly_pct, soil_moisture, tmean, cfg)
    pasture_drought = _pasture_drought_index(
        p=p,
        et0=et0,
        soil_moisture=soil_moisture,
        rainfall_anomaly_pct=rainfall_anomaly_pct,
        dry_spell=dry_spell,
        cfg=cfg,
    )
    livestock_cold = _spell_summary(tmin <= cfg.livestock_cold_threshold_c) if tmin is not None else _unavailable()
    wind_chill = _wind_chill_risk(tmean, wind10)
    mud_wetness = _mud_wetness_risk(p, wet_spell, rainfall_anomaly_pct, cfg)
    vector_suitability = _vector_suitability_proxy(p, rh, tmean)
    surface_water_stress = _surface_water_stress_proxy(
        rainfall_anomaly_pct=rainfall_anomaly_pct,
        et0=et0,
        dry_spell=dry_spell,
    )
    dust_dryness = _dust_dryness_risk(dry_spell, wind10, rh, cfg)

    crop_water_stress_score = _crop_water_stress_score(water_stress_index, rainfall_anomaly_pct, dry_spell)
    livestock_heat_score = _livestock_heat_score(thi, cfg)
    agro_pastoral_drought = _agro_pastoral_drought_risk(
        spei=spei,
        rainfall_anomaly_pct=rainfall_anomaly_pct,
        dry_spell=dry_spell,
        soil_moisture_anomaly=soil_moisture_anomaly,
    )
    feed_water_stress = _weighted_mean(
        [pasture_drought["score"], surface_water_stress["score"], livestock_heat_score],
        [
            cfg.weights.get("pasture_drought", 1.0),
            cfg.weights.get("water_stress", 1.0),
            cfg.weights.get("thi", 1.0),
        ],
    )
    crop_livelihood_stress = _weighted_mean(
        [crop_water_stress_score, pasture_drought["score"], livestock_heat_score],
        [
            cfg.crop_water_stress_weight,
            cfg.pasture_stress_weight,
            cfg.livestock_heat_stress_weight,
        ],
    )
    crop_residue = _crop_residue_outlook(rainfall_total, rainfall_anomaly_pct, lgp_days, gdd, cfg)
    planting_grazing_conflict = _planting_grazing_conflict_risk(
        onset=onset,
        pasture_drought=pasture_drought,
        cfg=cfg,
    )
    input_timing = _input_timing_suitability(onset, rainfall_anomaly_pct, dry_spell)
    water_allocation = _water_allocation_pressure(
        etc=etc,
        water_demand=water_demand,
        rainfall_anomaly_pct=rainfall_anomaly_pct,
        cfg=cfg,
    )
    mobility = _weighted_mean(
        [pasture_drought["score"], surface_water_stress["score"], livestock_heat_score],
        [1.0, 1.0, 1.0],
    )
    resilience = _integrated_resilience_opportunity(
        rainfall_anomaly_pct=rainfall_anomaly_pct,
        livestock_heat_score=livestock_heat_score,
        lgp_days=lgp_days,
        cfg=cfg,
    )
    advisory_score = _weighted_mean(
        [agro_pastoral_drought["score"], feed_water_stress, crop_livelihood_stress],
        [1.0, 1.0, 1.0],
    )

    crop = {
        "rainfall_total": rainfall_total,
        "rainfall_anomaly": rainfall_anomaly,
        "rainfall_anomaly_pct": rainfall_anomaly_pct,
        "tercile_class": tercile,
        "spi": spi,
        "spei": spei,
        "onset": onset,
        "false_start_risk": {
            "risk": onset["false_start_risk"],
            "first_candidate_date": onset["first_candidate_date"],
            "first_candidate_index": onset["first_candidate_index"],
        },
        "cessation": cessation,
        "length_of_growing_period_days": lgp_days,
        "dry_spell": dry_spell,
        "wet_spell": wet_spell,
        "extreme_rainfall_days": extreme_rain,
        "growing_degree_days": gdd,
        "heat_stress_days": crop_heat,
        "cold_stress_days": crop_cold,
        "diurnal_temperature_range": {
            "series": dtr,
            "mean": np.nanmean(dtr, axis=0) if dtr is not None else None,
            "max": np.nanmax(dtr, axis=0) if dtr is not None else None,
        },
        "reference_et0": {
            "series": et0,
            "total": np.nansum(et0, axis=0) if et0 is not None else None,
            "method": et0_meta,
        },
        "crop_et": {
            "series": etc,
            "total": np.nansum(etc, axis=0) if etc is not None else None,
        },
        "water_balance": {
            "p_minus_et0": water_balance_et0,
            "p_minus_etc": water_balance_etc,
            "total_p_minus_et0": np.nansum(water_balance_et0, axis=0) if water_balance_et0 is not None else None,
            "total_p_minus_etc": np.nansum(water_balance_etc, axis=0) if water_balance_etc is not None else None,
        },
        "soil_moisture_anomaly": soil_moisture_anomaly,
        "water_stress_index": {
            "series": water_stress_index,
            "mean": _nanmean_axis0(water_stress_index) if water_stress_index is not None else None,
            "max": _nanmax_axis0(water_stress_index) if water_stress_index is not None else None,
        },
        "solar_radiation_anomaly": solar_radiation_anomaly,
        "wind_risk": wind_risk,
    }

    livestock = {
        "temperature_humidity_index": thi_summary,
        "thi_risk_class": thi_summary["risk_class"] if thi_summary is not None else None,
        "heat_stress_duration": heat_duration,
        "night_recovery_index": night_recovery,
        "water_demand_index": water_demand,
        "pasture_rainfall_index": pasture_rain,
        "forage_growth_proxy": forage_growth,
        "pasture_drought_index": pasture_drought,
        "cold_stress_index": livestock_cold,
        "wind_chill_risk": wind_chill,
        "mud_wetness_risk": mud_wetness,
        "vector_suitability_proxy": vector_suitability,
        "surface_water_stress_proxy": surface_water_stress,
        "dust_dryness_risk": dust_dryness,
    }

    integrated = {
        "agro_pastoral_drought_risk": agro_pastoral_drought,
        "feed_water_stress_index": {
            "score": feed_water_stress,
            "class": _risk_class(feed_water_stress),
        },
        "crop_livestock_livelihood_stress_score": {
            "score": crop_livelihood_stress,
            "class": _risk_class(crop_livelihood_stress),
        },
        "crop_residue_outlook": crop_residue,
        "planting_grazing_conflict_risk": planting_grazing_conflict,
        "seasonal_advisory_class": {
            "score": advisory_score,
            "class": _advisory_class(advisory_score),
        },
        "input_timing_suitability": input_timing,
        "water_allocation_pressure": water_allocation,
        "transhumance_mobility_stress": {
            "score": mobility,
            "class": _risk_class(mobility),
        },
        "integrated_resilience_opportunity": resilience,
    }

    availability["computed"] = {
        "crop": sorted(crop.keys()),
        "livestock": sorted(livestock.keys()),
        "integrated": sorted(integrated.keys()),
    }

    return {
        "crop": crop,
        "livestock": livestock,
        "integrated": integrated,
        "availability": availability,
        "definitions": INDEX_DEFINITIONS,
        "required_variables": REQUIRED_VARIABLES,
    }


def summarize_indices(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact scalar summary from compute_agroclimate_indices output."""

    crop = result["crop"]
    livestock = result["livestock"]
    integrated = result["integrated"]

    return {
        "rainfall_total": _nanmean_scalar(crop["rainfall_total"]),
        "rainfall_anomaly_pct": _nanmean_scalar(crop["rainfall_anomaly_pct"]),
        "tercile_class": _majority_label(crop["tercile_class"]),
        "onset_date": _majority_label(crop["onset"]["onset_date"]),
        "cessation_date": _majority_label(crop["cessation"]["cessation_date"]),
        "length_of_growing_period_days": _nanmean_scalar(crop["length_of_growing_period_days"]),
        "max_dry_spell_days": _nanmean_scalar(crop["dry_spell"]["max_length"]),
        "gdd": _nanmean_scalar(crop["growing_degree_days"]),
        "et0_total": _nanmean_scalar(crop["reference_et0"]["total"]),
        "water_stress_index": _nanmean_scalar(crop["water_stress_index"]["mean"]),
        "thi_max": _nanmean_scalar(livestock["temperature_humidity_index"]["max"])
        if livestock["temperature_humidity_index"] is not None
        else None,
        "thi_risk_class": _majority_label(livestock["thi_risk_class"]),
        "pasture_drought_score": _nanmean_scalar(livestock["pasture_drought_index"]["score"]),
        "seasonal_advisory_class": _majority_label(integrated["seasonal_advisory_class"]["class"]),
        "seasonal_advisory_score": _nanmean_scalar(integrated["seasonal_advisory_class"]["score"]),
    }


def _require(weather: Weather, key: str) -> np.ndarray:
    value = _array_or_none(weather, key)
    if value is None:
        raise KeyError(f"Missing required weather variable: {key}")
    return value.astype(float, copy=False)


def _array_or_none(mapping: Mapping[str, Any], key: str) -> Optional[np.ndarray]:
    if key not in mapping or mapping[key] is None:
        return None
    return np.asarray(mapping[key])


def _value_or_weather(weather: Weather, key: str, default: Any) -> Any:
    return weather[key] if key in weather and weather[key] is not None else default


def _time_axis(weather: Weather, length: int) -> np.ndarray:
    if "time" not in weather or weather["time"] is None:
        start = np.datetime64(dt.date.today())
        return start + np.arange(length).astype("timedelta64[D]")
    time = np.asarray(weather["time"])
    if time.shape[0] != length:
        raise ValueError(f"time length {time.shape[0]} does not match data time length {length}")
    if np.issubdtype(time.dtype, np.datetime64):
        return time.astype("datetime64[D]")
    return np.asarray(time)


def _temperature_max(weather: Weather) -> Optional[np.ndarray]:
    return _array_or_none(weather, "temperature_2m_max")


def _temperature_min(weather: Weather) -> Optional[np.ndarray]:
    return _array_or_none(weather, "temperature_2m_min")


def _temperature_mean(
    weather: Weather,
    tmax: Optional[np.ndarray],
    tmin: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    tmean = _array_or_none(weather, "temperature_2m_mean")
    if tmean is not None:
        return tmean.astype(float, copy=False)
    t = _array_or_none(weather, "temperature_2m")
    if t is not None:
        return t.astype(float, copy=False)
    if tmax is not None and tmin is not None:
        return (tmax.astype(float) + tmin.astype(float)) / 2.0
    return None


def _relative_humidity(weather: Weather, tmean: Optional[np.ndarray]) -> Optional[np.ndarray]:
    rh = _array_or_none(weather, "relative_humidity_2m")
    if rh is not None:
        rh = rh.astype(float, copy=False)
        return np.where(rh > 1.5, rh / 100.0, rh)
    dew = _array_or_none(weather, "dew_point_2m")
    if dew is None or tmean is None:
        return None
    rh = _svp_kpa(dew.astype(float)) / _svp_kpa(tmean.astype(float))
    return np.clip(rh, 0.0, 1.0)


def _wind_speed_10m(weather: Weather) -> Optional[np.ndarray]:
    wind = _array_or_none(weather, "wind_speed_10m")
    if wind is not None:
        return wind.astype(float, copy=False)
    u = _array_or_none(weather, "wind_u_component_10m")
    v = _array_or_none(weather, "wind_v_component_10m")
    if u is None or v is None:
        return None
    return np.hypot(u.astype(float), v.astype(float))


def _soil_moisture(weather: Weather) -> Optional[np.ndarray]:
    direct = _array_or_none(weather, "soil_moisture")
    if direct is not None:
        return direct.astype(float, copy=False)

    layers = [
        ("soil_moisture_0_to_7cm", 7.0),
        ("soil_moisture_7_to_28cm", 21.0),
        ("soil_moisture_28_to_100cm", 72.0),
    ]
    values = []
    weights = []
    for key, weight in layers:
        value = _array_or_none(weather, key)
        if value is not None:
            values.append(value.astype(float))
            weights.append(weight)
    if not values:
        return None
    stacked = np.stack(values, axis=0)
    weight_array = np.asarray(weights, dtype=float).reshape((len(weights),) + (1,) * values[0].ndim)
    return np.nansum(stacked * weight_array, axis=0) / np.sum(weights)


def _surface_pressure_kpa(weather: Weather, cfg: AgroClimateConfig) -> Union[np.ndarray, float]:
    pressure = _array_or_none(weather, "surface_pressure")
    if pressure is None:
        pressure = _array_or_none(weather, "pressure_msl")
    if pressure is not None:
        pressure = pressure.astype(float)
        median = np.nanmedian(pressure)
        if median > 2000.0:
            return pressure / 1000.0  # Pa to kPa
        if median > 200.0:
            return pressure / 10.0  # hPa to kPa
        return pressure  # already kPa
    z = np.asarray(_value_or_weather(weather, "elevation", cfg.elevation_m), dtype=float)
    return 101.3 * np.power((293.0 - 0.0065 * z) / 293.0, 5.26)


def _missing_variables(weather: Weather) -> list[str]:
    present = set(weather.keys())
    aliases = {
        "temperature_2m_mean": {"temperature_2m_mean", "temperature_2m"},
        "relative_humidity_2m": {"relative_humidity_2m", "dew_point_2m"},
        "wind_speed_10m": {"wind_speed_10m", "wind_u_component_10m"},
        "surface_pressure": {"surface_pressure", "pressure_msl"},
        "soil_moisture": {
            "soil_moisture",
            "soil_moisture_0_to_7cm",
            "soil_moisture_7_to_28cm",
            "soil_moisture_28_to_100cm",
        },
    }
    missing = []
    for key in REQUIRED_VARIABLES:
        keys = aliases.get(key, {key})
        if not present.intersection(keys):
            missing.append(key)
    return missing


def _broadcast_param(value: Any, shape: tuple[int, ...], name: str) -> Optional[Union[np.ndarray, float]]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.shape == ():
        return float(arr)
    try:
        return np.broadcast_to(arr, shape)
    except ValueError as exc:
        raise ValueError(f"{name} shape {arr.shape} cannot broadcast to spatial shape {shape}") from exc


def _climatology_total(clim: Climatology, variable: str) -> Optional[np.ndarray]:
    direct = _clim_lookup(
        clim,
        f"{variable}_total",
        f"{variable}_seasonal_total",
        "rainfall_total" if variable == "precipitation" else "",
    )
    if direct is not None:
        return direct
    series = _clim_lookup(clim, variable)
    if series is None:
        return None
    series = np.asarray(series, dtype=float)
    if series.ndim == 0:
        return series
    return np.nansum(series, axis=0)


def _clim_lookup(clim: Climatology, *keys: str) -> Optional[np.ndarray]:
    for key in keys:
        if key and key in clim and clim[key] is not None:
            return np.asarray(clim[key], dtype=float)
    return None


def _anomaly(value: Optional[np.ndarray], clim_value: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if value is None or clim_value is None:
        return None
    return np.asarray(value, dtype=float) - np.asarray(clim_value, dtype=float)


def _percent_anomaly(value: Optional[np.ndarray], clim_value: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if value is None or clim_value is None:
        return None
    clim_value = np.asarray(clim_value, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100.0 * (np.asarray(value, dtype=float) - clim_value) / clim_value


def _tercile_class(value: np.ndarray, clim: Climatology) -> Optional[np.ndarray]:
    p33 = _clim_lookup(clim, "precipitation_p33", "rainfall_p33", "precipitation_total_p33")
    p66 = _clim_lookup(clim, "precipitation_p66", "rainfall_p66", "precipitation_total_p66")
    if p33 is None or p66 is None:
        totals = _clim_lookup(clim, "precipitation_totals", "rainfall_totals")
        if totals is None:
            return None
        totals = np.asarray(totals, dtype=float)
        p33 = np.nanpercentile(totals, 33.333, axis=0)
        p66 = np.nanpercentile(totals, 66.667, axis=0)
    out = np.full(np.asarray(value).shape, "N", dtype=object)
    out = np.where(value < p33, "BN", out)
    out = np.where(value > p66, "AN", out)
    return out


def _reference_et0_fao56(
    *,
    time: np.ndarray,
    tmax: Optional[np.ndarray],
    tmin: Optional[np.ndarray],
    tmean: Optional[np.ndarray],
    rh: Optional[np.ndarray],
    wind10: Optional[np.ndarray],
    shortwave: Optional[np.ndarray],
    pressure_kpa: Union[np.ndarray, float],
    latitude: Optional[Union[np.ndarray, float]],
    elevation: Optional[Union[np.ndarray, float]],
    cfg: AgroClimateConfig,
) -> tuple[Optional[np.ndarray], dict[str, Any]]:
    if tmax is None or tmin is None or tmean is None:
        return None, {"available": False, "reason": "temperature_2m_max/min/mean are needed"}

    method = "FAO-56 Penman-Monteith"
    notes = []
    if rh is None:
        rh = _svp_kpa(tmin.astype(float)) / _svp_kpa(tmean.astype(float))
        rh = np.clip(rh, 0.0, 1.0)
        notes.append("relative humidity estimated from Tmin dewpoint assumption")
    if wind10 is None:
        wind10 = np.full_like(tmean, 2.0, dtype=float)
        notes.append("wind speed defaulted to 2 m/s")
    u2 = wind10.astype(float) * 4.87 / np.log(67.8 * 10.0 - 5.42)

    if shortwave is None and latitude is None:
        return None, {
            "available": False,
            "reason": "shortwave_radiation or latitude is needed for radiation term",
        }

    doy = _day_of_year(time)
    if latitude is not None:
        ra = _extraterrestrial_radiation(doy, latitude)
        if ra.ndim == 1 and tmean.ndim > 1:
            ra = ra.reshape((-1,) + (1,) * (tmean.ndim - 1))
    else:
        ra = None

    if shortwave is not None:
        rs = _shortwave_to_mj_m2_day(shortwave.astype(float), cfg.shortwave_radiation_unit)
    else:
        rs = None

    if rs is None:
        rs = 0.16 * ra * np.sqrt(np.maximum(tmax - tmin, 0.0))
        method = "FAO-56 Penman-Monteith with Hargreaves radiation estimate"
        notes.append("shortwave radiation estimated from temperature range")
    if ra is None:
        rn = (1.0 - cfg.albedo) * rs
        notes.append("latitude absent; longwave net radiation omitted")
    else:
        rso = (0.75 + 2e-5 * np.asarray(elevation if elevation is not None else 0.0, dtype=float)) * ra
        rns = (1.0 - cfg.albedo) * rs
        es = (_svp_kpa(tmax) + _svp_kpa(tmin)) / 2.0
        ea = np.clip(rh, 0.0, 1.0) * es
        sigma = 4.903e-9
        cloud = 1.35 * np.clip(rs / np.maximum(rso, 1e-6), 0.0, 1.0) - 0.35
        cloud = np.clip(cloud, 0.05, 1.0)
        rnl = sigma * ((np.power(tmax + 273.16, 4) + np.power(tmin + 273.16, 4)) / 2.0)
        rnl = rnl * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0.0))) * cloud
        rn = rns - rnl

    es = (_svp_kpa(tmax) + _svp_kpa(tmin)) / 2.0
    ea = np.clip(rh, 0.0, 1.0) * es
    delta = 4098.0 * _svp_kpa(tmean) / np.power(tmean + 237.3, 2)
    gamma = 0.000665 * pressure_kpa
    numerator = 0.408 * delta * rn + gamma * (900.0 / (tmean + 273.0)) * u2 * (es - ea)
    denominator = delta + gamma * (1.0 + 0.34 * u2)
    et0 = numerator / denominator
    et0 = np.where(np.isfinite(et0), np.maximum(et0, 0.0), np.nan)
    return et0, {"available": True, "method": method, "notes": notes}


def _svp_kpa(temp_c: Union[np.ndarray, float]) -> np.ndarray:
    return 0.6108 * np.exp((17.27 * np.asarray(temp_c, dtype=float)) / (np.asarray(temp_c, dtype=float) + 237.3))


def _day_of_year(time: np.ndarray) -> np.ndarray:
    if np.issubdtype(time.dtype, np.datetime64):
        dates = time.astype("datetime64[D]")
        years = dates.astype("datetime64[Y]")
        return (dates - years).astype(int) + 1
    return (np.arange(time.shape[0]) % 365) + 1


def _extraterrestrial_radiation(doy: np.ndarray, latitude: Union[np.ndarray, float]) -> np.ndarray:
    lat_rad = np.deg2rad(np.asarray(latitude, dtype=float))
    j = np.asarray(doy, dtype=float).reshape((-1,) + (1,) * np.ndim(lat_rad))
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * j / 365.0)
    solar_declination = 0.409 * np.sin(2.0 * np.pi * j / 365.0 - 1.39)
    sunset_hour_angle = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(solar_declination), -1.0, 1.0))
    gsc = 0.0820
    return (
        (24.0 * 60.0 / np.pi)
        * gsc
        * dr
        * (
            sunset_hour_angle * np.sin(lat_rad) * np.sin(solar_declination)
            + np.cos(lat_rad) * np.cos(solar_declination) * np.sin(sunset_hour_angle)
        )
    )


def _shortwave_to_mj_m2_day(values: np.ndarray, unit: str) -> np.ndarray:
    if unit == "mj_m2_day":
        return values
    if unit == "w_m2":
        return values * 0.0864
    median = np.nanmedian(values)
    if median > 80.0:
        return values * 0.0864
    return values


def _crop_et(et0: Optional[np.ndarray], kc: ArrayLike) -> Optional[np.ndarray]:
    if et0 is None:
        return None
    return et0 * np.asarray(kc, dtype=float)


def _safe_subtract(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if a is None or b is None:
        return None
    return np.asarray(a, dtype=float) - np.asarray(b, dtype=float)


def _actual_et(weather: Weather, p: np.ndarray, etc: Optional[np.ndarray], cfg: AgroClimateConfig) -> tuple[Optional[np.ndarray], str]:
    direct = _array_or_none(weather, cfg.actual_et_key)
    if direct is not None:
        return direct.astype(float), cfg.actual_et_key
    if etc is None:
        return None, "unavailable"
    return np.minimum(np.maximum(p.astype(float), 0.0), etc), "estimated_min_precipitation_crop_et"


def _water_stress_index(aet: Optional[np.ndarray], etc: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if aet is None or etc is None:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.clip(1.0 - aet / etc, 0.0, 1.0)


def _standardized_indices(
    *,
    values: Optional[np.ndarray],
    base_key: str,
    climatology: Climatology,
    cfg: AgroClimateConfig,
    output_name: str,
) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    if values is None:
        return {scale: {"available": False, "reason": f"{base_key} unavailable"} for scale in cfg.standardization_scales}

    for scale in cfg.standardization_scales:
        window = scale if cfg.frequency == "monthly" else scale * cfg.month_days
        if values.shape[0] < window:
            out[scale] = {
                "available": False,
                "window": window,
                "reason": f"Need at least {window} {cfg.frequency} steps.",
            }
            continue
        rolling = _rolling_sum_trailing(values, window)
        mean_key = f"{output_name}_{scale}_mean"
        std_key = f"{output_name}_{scale}_std"
        accum_key = f"{base_key}_{scale}_accumulations"
        clim_mean = _clim_lookup(climatology, mean_key, f"{base_key}_{scale}_mean")
        clim_std = _clim_lookup(climatology, std_key, f"{base_key}_{scale}_std")
        if clim_mean is None or clim_std is None:
            accumulations = _clim_lookup(climatology, accum_key)
            if accumulations is not None:
                clim_mean = np.nanmean(accumulations, axis=0)
                clim_std = np.nanstd(accumulations, axis=0, ddof=1)
        if clim_mean is None or clim_std is None:
            out[scale] = {
                "available": False,
                "window": window,
                "accumulation": rolling,
                "reason": "Needs climatological mean/std or historical accumulations.",
            }
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (rolling - clim_mean) / clim_std
        out[scale] = {
            "available": True,
            "window": window,
            "series": z,
            "latest": z[-1],
            "accumulation": rolling,
        }
    return out


def _rolling_sum_trailing(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values.copy()
    out = np.full_like(values, np.nan, dtype=float)
    csum = np.cumsum(np.where(np.isfinite(values), values, 0.0), axis=0)
    csum = np.concatenate([np.zeros_like(csum[:1]), csum], axis=0)
    out[window - 1 :] = csum[window:] - csum[:-window]
    return out


def _rolling_sum_forward(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full_like(values, np.nan, dtype=float)
    if window <= 1:
        return values.copy()
    reversed_sum = _rolling_sum_trailing(values[::-1], window)[::-1]
    out[: values.shape[0] - window + 1] = reversed_sum[: values.shape[0] - window + 1]
    return out


def _onset_and_false_start(p: np.ndarray, time: np.ndarray, cfg: AgroClimateConfig) -> dict[str, Any]:
    wet_sum = _rolling_sum_forward(p, cfg.onset_window_days)
    wet_days = _rolling_sum_forward((p >= cfg.wet_day_mm).astype(float), cfg.onset_window_days)
    candidate = (
        (p >= cfg.wet_day_mm)
        & (wet_sum >= cfg.onset_total_rain_mm)
        & (wet_days >= cfg.onset_min_wet_days)
    )
    long_dry_after = _long_dry_after_each_day(p, cfg)
    safe_candidate = candidate & ~long_dry_after
    first_candidate = _first_true_index(candidate)
    onset_index = _first_true_index(safe_candidate)
    false_risk = (first_candidate >= 0) & _take_time(
        long_dry_after,
        np.maximum(first_candidate, 0),
        fill_value=False,
    )
    false_risk = np.where(first_candidate < 0, False, false_risk)
    return {
        "onset_index": onset_index,
        "onset_date": _dates_from_index(time, onset_index),
        "first_candidate_index": first_candidate,
        "first_candidate_date": _dates_from_index(time, first_candidate),
        "false_start_risk": false_risk,
        "candidate_mask": candidate,
    }


def _long_dry_after_each_day(p: np.ndarray, cfg: AgroClimateConfig) -> np.ndarray:
    dry = p < cfg.dry_day_mm
    out = np.zeros_like(dry, dtype=bool)
    t_len = p.shape[0]
    for t in range(t_len):
        start = min(t + cfg.onset_window_days, t_len)
        stop = min(t + cfg.false_start_lookahead_days, t_len)
        if start >= stop:
            continue
        out[t] = _max_spell_length(dry[start:stop]) >= cfg.false_start_dry_spell_days
    return out


def _first_true_index(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    any_true = np.any(mask, axis=0)
    first = np.argmax(mask, axis=0)
    return np.where(any_true, first, -1)


def _take_time(values: np.ndarray, index: np.ndarray, fill_value: Any = np.nan) -> np.ndarray:
    values = np.asarray(values)
    idx = np.asarray(index)
    flat_values = values.reshape((values.shape[0], -1))
    flat_idx = idx.reshape(-1)
    out = np.full(flat_idx.shape, fill_value, dtype=values.dtype)
    valid = (flat_idx >= 0) & (flat_idx < values.shape[0])
    if np.any(valid):
        out[valid] = flat_values[flat_idx[valid], np.nonzero(valid)[0]]
    return out.reshape(idx.shape)


def _dates_from_index(time: np.ndarray, index: np.ndarray) -> np.ndarray:
    idx = np.asarray(index)
    shape = idx.shape
    out = np.full(shape, "NaT", dtype="datetime64[D]")
    valid = idx >= 0
    if not np.issubdtype(time.dtype, np.datetime64):
        return np.where(valid, idx, -1)
    if np.any(valid):
        out[valid] = time[idx[valid]].astype("datetime64[D]")
    return out


def _cessation(
    p: np.ndarray,
    water_balance: Optional[np.ndarray],
    time: np.ndarray,
    onset_index: np.ndarray,
    cfg: AgroClimateConfig,
) -> dict[str, Any]:
    rain_support = _rolling_sum_forward(p, cfg.cessation_window_days) >= cfg.cessation_rain_mm
    if water_balance is not None:
        wb_support = _rolling_sum_forward(water_balance, cfg.cessation_window_days) >= cfg.cessation_water_balance_mm
        support = rain_support | wb_support
    else:
        support = rain_support

    cessation_index = _last_true_index_after(support, onset_index)
    return {
        "cessation_index": cessation_index,
        "cessation_date": _dates_from_index(time, cessation_index),
        "support_mask": support,
    }


def _last_true_index_after(mask: np.ndarray, after_index: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    after = np.asarray(after_index)
    out = np.full(mask.shape[1:], -1, dtype=int)
    for t in range(mask.shape[0]):
        valid_after = (after < 0) | (t >= after)
        out = np.where(mask[t] & valid_after, t, out)
    return out


def _lgp_days(onset_index: np.ndarray, cessation_index: np.ndarray) -> np.ndarray:
    onset = np.asarray(onset_index)
    cessation = np.asarray(cessation_index)
    valid = (onset >= 0) & (cessation >= onset)
    return np.where(valid, cessation - onset + 1, np.nan)


def _spell_summary(mask: np.ndarray) -> dict[str, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    starts = mask & ~np.concatenate([np.zeros_like(mask[:1], dtype=bool), mask[:-1]], axis=0)
    count = np.sum(starts, axis=0)
    max_length = _max_spell_length(mask)
    total_days = np.sum(mask, axis=0)
    return {"count": count, "max_length": max_length, "total_days": total_days}


def _max_spell_length(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[0] == 0:
        return np.zeros(mask.shape[1:], dtype=int)
    run = np.zeros(mask.shape[1:], dtype=int)
    max_run = np.zeros(mask.shape[1:], dtype=int)
    for t in range(mask.shape[0]):
        run = np.where(mask[t], run + 1, 0)
        max_run = np.maximum(max_run, run)
    return max_run


def _unavailable() -> dict[str, Any]:
    return {"available": False}


def _extreme_rainfall(p: np.ndarray, clim: Climatology, cfg: AgroClimateConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for threshold in cfg.extreme_rain_thresholds_mm:
        out[f"days_ge_{_format_threshold(threshold)}mm"] = np.sum(p >= threshold, axis=0)
    p95 = _clim_lookup(clim, "precipitation_daily_p95", "rainfall_daily_p95")
    if p95 is not None:
        out["days_ge_p95"] = np.sum(p >= p95, axis=0)
    else:
        out["days_ge_p95"] = None
    return out


def _format_threshold(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")


def _gdd(tmax: Optional[np.ndarray], tmin: Optional[np.ndarray], tbase: float) -> Optional[np.ndarray]:
    if tmax is None or tmin is None:
        return None
    return np.nansum(np.maximum(((tmax + tmin) / 2.0) - tbase, 0.0), axis=0)


def _thi(tmean: Optional[np.ndarray], rh: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if tmean is None or rh is None:
        return None
    return 0.8 * tmean + np.clip(rh, 0.0, 1.0) * (tmean - 14.4) + 46.4


def _thi_summary(thi: Optional[np.ndarray], cfg: AgroClimateConfig) -> Optional[dict[str, Any]]:
    if thi is None:
        return None
    latest = thi[-1]
    maximum = np.nanmax(thi, axis=0)
    mean = np.nanmean(thi, axis=0)
    return {
        "series": thi,
        "latest": latest,
        "mean": mean,
        "max": maximum,
        "risk_class": _thi_risk_class(maximum, cfg),
    }


def _thi_risk_class(value: np.ndarray, cfg: AgroClimateConfig) -> np.ndarray:
    out = np.full(np.asarray(value).shape, "low", dtype=object)
    out = np.where(value >= cfg.thi_mild_threshold, "mild", out)
    out = np.where(value >= cfg.thi_moderate_threshold, "moderate", out)
    out = np.where(value >= cfg.thi_severe_threshold, "severe", out)
    out = np.where(value >= cfg.thi_emergency_threshold, "emergency", out)
    return out


def _night_recovery(tmin: Optional[np.ndarray], thi: Optional[np.ndarray], cfg: AgroClimateConfig) -> dict[str, Any]:
    if thi is not None:
        poor = thi >= cfg.night_thi_recovery_threshold
        basis = "THI"
    elif tmin is not None:
        poor = tmin >= cfg.night_tmin_recovery_threshold_c
        basis = "Tmin"
    else:
        return _unavailable()
    summary = _spell_summary(poor)
    summary["basis"] = basis
    return summary


def _water_demand_index(thi: Optional[np.ndarray], cfg: AgroClimateConfig) -> dict[str, Any]:
    if thi is None:
        return _unavailable()
    multiplier = np.ones_like(thi, dtype=float)
    multiplier = np.where(thi >= cfg.thi_mild_threshold, 1.10, multiplier)
    multiplier = np.where(thi >= cfg.thi_moderate_threshold, 1.25, multiplier)
    multiplier = np.where(thi >= cfg.thi_severe_threshold, 1.50, multiplier)
    multiplier = np.where(thi >= cfg.thi_emergency_threshold, 1.80, multiplier)
    demand = cfg.baseline_livestock_water_l_day * multiplier
    return {
        "series_l_day": demand,
        "mean_l_day": np.nanmean(demand, axis=0),
        "max_l_day": np.nanmax(demand, axis=0),
        "multiplier": multiplier,
    }


def _pasture_rainfall_index(p: np.ndarray, clim: Climatology, cfg: AgroClimateConfig) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for window in cfg.rainfall_windows_days:
        if p.shape[0] < window:
            out[window] = {"available": False, "reason": f"Need at least {window} days."}
            continue
        rolling = _rolling_sum_trailing(p, window)
        latest = rolling[-1]
        clim_mean = _clim_lookup(clim, f"precipitation_{window}d_mean", f"rainfall_{window}d_mean")
        anomaly = _anomaly(latest, clim_mean)
        anomaly_pct = _percent_anomaly(latest, clim_mean)
        out[window] = {
            "available": True,
            "rolling_total": rolling,
            "latest_total": latest,
            "anomaly": anomaly,
            "anomaly_pct": anomaly_pct,
        }
    return out


def _forage_growth_proxy(
    p: np.ndarray,
    rainfall_anomaly_pct: Optional[np.ndarray],
    soil_moisture: Optional[np.ndarray],
    tmean: Optional[np.ndarray],
    cfg: AgroClimateConfig,
) -> dict[str, Any]:
    if p.shape[0] >= 30:
        rain30 = _rolling_sum_trailing(p, 30)[-1]
    else:
        rain30 = np.nansum(p, axis=0) * 30.0 / max(p.shape[0], 1)
    if rainfall_anomaly_pct is not None:
        rain_score = _score_from_anomaly_pct(rainfall_anomaly_pct, invert=False)
    else:
        rain_score = np.clip(rain30 / cfg.pasture_target_rain_30d_mm, 0.0, 1.0)

    if soil_moisture is not None:
        sm = np.nanmean(soil_moisture, axis=0)
        sm_score = np.clip(
            (sm - np.asarray(cfg.soil_moisture_wilting))
            / (np.asarray(cfg.soil_moisture_field_capacity) - np.asarray(cfg.soil_moisture_wilting)),
            0.0,
            1.0,
        )
    else:
        sm_score = np.nan

    if tmean is not None:
        temp_score = _temperature_suitability(np.nanmean(tmean, axis=0), cfg)
    else:
        temp_score = np.nan

    score = _mean_available([rain_score, sm_score, temp_score])
    return {"score": score, "class": _opportunity_class(score)}


def _temperature_suitability(t: np.ndarray, cfg: AgroClimateConfig) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    below = (t - cfg.pasture_temperature_min_c) / (
        cfg.pasture_temperature_optimum_c - cfg.pasture_temperature_min_c
    )
    above = (cfg.pasture_temperature_max_c - t) / (
        cfg.pasture_temperature_max_c - cfg.pasture_temperature_optimum_c
    )
    return np.clip(np.where(t <= cfg.pasture_temperature_optimum_c, below, above), 0.0, 1.0)


def _pasture_drought_index(
    *,
    p: np.ndarray,
    et0: Optional[np.ndarray],
    soil_moisture: Optional[np.ndarray],
    rainfall_anomaly_pct: Optional[np.ndarray],
    dry_spell: dict[str, Any],
    cfg: AgroClimateConfig,
) -> dict[str, Any]:
    rain_deficit = _score_from_anomaly_pct(rainfall_anomaly_pct, invert=True)
    if np.all(np.isnan(rain_deficit)):
        total = np.nansum(p, axis=0)
        expected = cfg.pasture_target_rain_30d_mm * p.shape[0] / 30.0
        rain_deficit = np.clip(1.0 - total / max(expected, 1e-6), 0.0, 1.0)
    et_score = _high_et_score(et0)
    if soil_moisture is not None:
        sm = np.nanmean(soil_moisture, axis=0)
        sm_deficit = np.clip(
            (np.asarray(cfg.soil_moisture_field_capacity) - sm)
            / (np.asarray(cfg.soil_moisture_field_capacity) - np.asarray(cfg.soil_moisture_wilting)),
            0.0,
            1.0,
        )
    else:
        sm_deficit = np.nan
    dry_score = np.clip(np.asarray(dry_spell["max_length"], dtype=float) / 20.0, 0.0, 1.0)
    score = _mean_available([rain_deficit, et_score, sm_deficit, dry_score])
    return {"score": score, "class": _risk_class(score)}


def _high_et_score(et0: Optional[np.ndarray]) -> np.ndarray:
    if et0 is None:
        return np.nan
    mean_et0 = np.nanmean(et0, axis=0)
    return np.clip((mean_et0 - 3.0) / 5.0, 0.0, 1.0)


def _wind_chill_risk(tmean: Optional[np.ndarray], wind10: Optional[np.ndarray]) -> dict[str, Any]:
    if tmean is None or wind10 is None:
        return _unavailable()
    wind_kmh = np.maximum(wind10 * 3.6, 4.8)
    wind_chill = 13.12 + 0.6215 * tmean - 11.37 * np.power(wind_kmh, 0.16) + 0.3965 * tmean * np.power(wind_kmh, 0.16)
    risk = (tmean <= 10.0) & (wind_chill <= 5.0)
    summary = _spell_summary(risk)
    summary["wind_chill_min_c"] = np.nanmin(wind_chill, axis=0)
    return summary


def _mud_wetness_risk(
    p: np.ndarray,
    wet_spell: dict[str, Any],
    rainfall_anomaly_pct: Optional[np.ndarray],
    cfg: AgroClimateConfig,
) -> dict[str, Any]:
    wet_score = np.clip(np.asarray(wet_spell["max_length"], dtype=float) / 10.0, 0.0, 1.0)
    excess_score = _score_from_anomaly_pct(rainfall_anomaly_pct, invert=False)
    if np.all(np.isnan(excess_score)):
        excess_score = np.clip(np.nanmax(_rolling_sum_trailing(p, min(7, p.shape[0])), axis=0) / 50.0, 0.0, 1.0)
    score = _mean_available([wet_score, excess_score])
    return {"score": score, "class": _risk_class(score)}


def _vector_suitability_proxy(
    p: np.ndarray,
    rh: Optional[np.ndarray],
    tmean: Optional[np.ndarray],
) -> dict[str, Any]:
    rain_score = np.clip(np.nansum(p, axis=0) / 100.0, 0.0, 1.0)
    if rh is not None:
        humidity_score = np.clip((np.nanmean(rh, axis=0) - 0.55) / 0.35, 0.0, 1.0)
    else:
        humidity_score = np.nan
    if tmean is not None:
        temp = np.nanmean(tmean, axis=0)
        temp_score = np.clip(np.minimum((temp - 15.0) / 10.0, (35.0 - temp) / 10.0), 0.0, 1.0)
    else:
        temp_score = np.nan
    score = _mean_available([rain_score, humidity_score, temp_score])
    return {"score": score, "class": _opportunity_class(score)}


def _surface_water_stress_proxy(
    *,
    rainfall_anomaly_pct: Optional[np.ndarray],
    et0: Optional[np.ndarray],
    dry_spell: dict[str, Any],
) -> dict[str, Any]:
    rain_deficit = _score_from_anomaly_pct(rainfall_anomaly_pct, invert=True)
    dry_score = np.clip(np.asarray(dry_spell["max_length"], dtype=float) / 30.0, 0.0, 1.0)
    et_score = _high_et_score(et0)
    score = _mean_available([rain_deficit, dry_score, et_score])
    return {"score": score, "class": _risk_class(score)}


def _dust_dryness_risk(
    dry_spell: dict[str, Any],
    wind10: Optional[np.ndarray],
    rh: Optional[np.ndarray],
    cfg: AgroClimateConfig,
) -> dict[str, Any]:
    dry_score = np.clip(np.asarray(dry_spell["max_length"], dtype=float) / 20.0, 0.0, 1.0)
    if wind10 is not None:
        wind_score = np.clip((np.nanmean(wind10, axis=0) - 4.0) / (cfg.wind_risk_threshold_m_s - 4.0), 0.0, 1.0)
    else:
        wind_score = np.nan
    if rh is not None:
        humidity_score = np.clip((0.45 - np.nanmean(rh, axis=0)) / 0.30, 0.0, 1.0)
    else:
        humidity_score = np.nan
    score = _mean_available([dry_score, wind_score, humidity_score])
    return {"score": score, "class": _risk_class(score)}


def _crop_water_stress_score(
    wsi: Optional[np.ndarray],
    rainfall_anomaly_pct: Optional[np.ndarray],
    dry_spell: dict[str, Any],
) -> np.ndarray:
    if wsi is not None:
        wsi_score = _nanmean_axis0(wsi)
    else:
        wsi_score = np.nan
    rain_deficit = _score_from_anomaly_pct(rainfall_anomaly_pct, invert=True)
    dry_score = np.clip(np.asarray(dry_spell["max_length"], dtype=float) / 20.0, 0.0, 1.0)
    return _mean_available([wsi_score, rain_deficit, dry_score])


def _livestock_heat_score(thi: Optional[np.ndarray], cfg: AgroClimateConfig) -> np.ndarray:
    if thi is None:
        return np.nan
    max_thi = np.nanmax(thi, axis=0)
    return np.clip((max_thi - cfg.thi_mild_threshold) / (cfg.thi_emergency_threshold - cfg.thi_mild_threshold), 0.0, 1.0)


def _agro_pastoral_drought_risk(
    *,
    spei: dict[int, dict[str, Any]],
    rainfall_anomaly_pct: Optional[np.ndarray],
    dry_spell: dict[str, Any],
    soil_moisture_anomaly: Optional[np.ndarray],
) -> dict[str, Any]:
    spei_values = [item["latest"] for item in spei.values() if item.get("available")]
    if spei_values:
        spei_score = np.clip((-_mean_available(spei_values) + 1.0) / 3.0, 0.0, 1.0)
    else:
        spei_score = np.nan
    rain_deficit = _score_from_anomaly_pct(rainfall_anomaly_pct, invert=True)
    dry_score = np.clip(np.asarray(dry_spell["max_length"], dtype=float) / 20.0, 0.0, 1.0)
    if soil_moisture_anomaly is not None:
        sm_score = np.clip(-np.asarray(soil_moisture_anomaly, dtype=float) / 0.15, 0.0, 1.0)
    else:
        sm_score = np.nan
    score = _mean_available([spei_score, rain_deficit, dry_score, sm_score])
    return {"score": score, "class": _risk_class(score)}


def _crop_residue_outlook(
    rainfall_total: np.ndarray,
    rainfall_anomaly_pct: Optional[np.ndarray],
    lgp_days: np.ndarray,
    gdd: Optional[np.ndarray],
    cfg: AgroClimateConfig,
) -> dict[str, Any]:
    if rainfall_anomaly_pct is not None:
        rain_score = _score_from_anomaly_pct(rainfall_anomaly_pct, invert=False)
    else:
        rain_score = np.clip(np.asarray(rainfall_total, dtype=float) / (cfg.pasture_target_rain_30d_mm * 3.0), 0.0, 1.0)
    lgp_score = np.clip(np.asarray(lgp_days, dtype=float) / 120.0, 0.0, 1.0)
    gdd_score = np.clip(np.asarray(gdd, dtype=float) / 1500.0, 0.0, 1.0) if gdd is not None else np.nan
    score = _mean_available([rain_score, lgp_score, gdd_score])
    return {"score": score, "class": _opportunity_class(score)}


def _planting_grazing_conflict_risk(
    onset: dict[str, Any],
    pasture_drought: dict[str, Any],
    cfg: AgroClimateConfig,
) -> dict[str, Any]:
    onset_index = np.asarray(onset["onset_index"], dtype=float)
    delayed = np.where(onset_index < 0, 1.0, np.clip((onset_index - 30.0) / 60.0, 0.0, 1.0))
    false_start = np.asarray(onset["false_start_risk"], dtype=float)
    pressure = np.clip(cfg.cropping_land_pressure, 0.0, 1.0)
    score = _mean_available([delayed, pasture_drought["score"], false_start, pressure])
    return {"score": score, "class": _risk_class(score)}


def _input_timing_suitability(
    onset: dict[str, Any],
    rainfall_anomaly_pct: Optional[np.ndarray],
    dry_spell: dict[str, Any],
) -> dict[str, Any]:
    onset_exists = np.asarray(onset["onset_index"]) >= 0
    onset_score = np.where(onset_exists, 1.0, 0.0)
    rain_score = _score_from_anomaly_pct(rainfall_anomaly_pct, invert=False)
    dry_score = 1.0 - np.clip(np.asarray(dry_spell["max_length"], dtype=float) / 20.0, 0.0, 1.0)
    false_score = 1.0 - np.asarray(onset["false_start_risk"], dtype=float)
    score = _mean_available([onset_score, rain_score, dry_score, false_score])
    return {"score": score, "class": _opportunity_class(score)}


def _water_allocation_pressure(
    *,
    etc: Optional[np.ndarray],
    water_demand: dict[str, Any],
    rainfall_anomaly_pct: Optional[np.ndarray],
    cfg: AgroClimateConfig,
) -> dict[str, Any]:
    if etc is not None:
        crop_demand_score = np.clip(np.nansum(etc, axis=0) / 500.0, 0.0, 1.0)
    else:
        crop_demand_score = np.nan
    if water_demand.get("available") is False:
        livestock_score = np.nan
    else:
        livestock_score = np.clip(
            (water_demand["mean_l_day"] * cfg.livestock_units - cfg.baseline_livestock_water_l_day)
            / max(cfg.baseline_livestock_water_l_day, 1e-6),
            0.0,
            1.0,
        )
    rain_deficit = _score_from_anomaly_pct(rainfall_anomaly_pct, invert=True)
    score = _mean_available([crop_demand_score, livestock_score, rain_deficit])
    return {"score": score, "class": _risk_class(score)}


def _integrated_resilience_opportunity(
    *,
    rainfall_anomaly_pct: Optional[np.ndarray],
    livestock_heat_score: np.ndarray,
    lgp_days: np.ndarray,
    cfg: AgroClimateConfig,
) -> dict[str, Any]:
    rain_score = _score_from_anomaly_pct(rainfall_anomaly_pct, invert=False)
    low_heat_score = 1.0 - livestock_heat_score if not np.all(np.isnan(livestock_heat_score)) else np.nan
    lgp_score = np.clip(np.asarray(lgp_days, dtype=float) / 120.0, 0.0, 1.0)
    score = _mean_available([rain_score, low_heat_score, lgp_score])
    return {"score": score, "class": _opportunity_class(score)}


def _score_from_anomaly_pct(anomaly_pct: Optional[np.ndarray], *, invert: bool) -> np.ndarray:
    if anomaly_pct is None:
        return np.nan
    anomaly_pct = np.asarray(anomaly_pct, dtype=float)
    if invert:
        return np.clip(-anomaly_pct / 50.0, 0.0, 1.0)
    return np.clip((anomaly_pct + 25.0) / 75.0, 0.0, 1.0)


def _mean_available(values: list[Any]) -> np.ndarray:
    arrays = []
    for value in values:
        if value is None:
            continue
        arr = np.asarray(value, dtype=float)
        arrays.append(arr)
    if not arrays:
        return np.asarray(np.nan)
    shape = np.broadcast_shapes(*[arr.shape for arr in arrays])
    stacked = np.stack([np.broadcast_to(arr, shape) for arr in arrays], axis=0)
    with np.errstate(invalid="ignore"):
        return np.nanmean(stacked, axis=0)


def _nanmean_axis0(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    count = np.sum(finite, axis=0)
    total = np.sum(np.where(finite, arr, 0.0), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(count > 0, total / count, np.nan)


def _nanmax_axis0(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    has_value = np.any(finite, axis=0)
    max_values = np.max(np.where(finite, arr, -np.inf), axis=0)
    return np.where(has_value, max_values, np.nan)


def _weighted_mean(values: list[Any], weights: list[float]) -> np.ndarray:
    arrays = []
    valid_weights = []
    for value, weight in zip(values, weights):
        if value is None:
            continue
        arrays.append(np.asarray(value, dtype=float))
        valid_weights.append(float(weight))
    if not arrays:
        return np.asarray(np.nan)
    shape = np.broadcast_shapes(*[arr.shape for arr in arrays])
    num = np.zeros(shape, dtype=float)
    den = np.zeros(shape, dtype=float)
    for arr, weight in zip(arrays, valid_weights):
        arr = np.broadcast_to(arr, shape)
        valid = np.isfinite(arr)
        num += np.where(valid, arr * weight, 0.0)
        den += np.where(valid, weight, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / den


def _risk_class(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    out = np.full(score.shape, "Normal", dtype=object)
    out = np.where(score >= 0.25, "Watch", out)
    out = np.where(score >= 0.50, "Alert", out)
    out = np.where(score >= 0.75, "Emergency", out)
    return out


def _advisory_class(score: np.ndarray) -> np.ndarray:
    return _risk_class(score)


def _opportunity_class(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    out = np.full(score.shape, "Low", dtype=object)
    out = np.where(score >= 0.40, "Moderate", out)
    out = np.where(score >= 0.70, "High", out)
    return out


def _nanmean_scalar(value: Any) -> Optional[float]:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.dtype == object or np.issubdtype(arr.dtype, np.str_):
        return None
    arr = arr.astype(float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _majority_label(value: Any) -> Any:
    if value is None:
        return None
    arr = np.asarray(value).ravel()
    arr = arr[arr != None]  # noqa: E711
    if arr.size == 0:
        return None
    arr = arr[arr.astype(str) != "NaT"]
    if arr.size == 0:
        return None
    labels, counts = np.unique(arr.astype(str), return_counts=True)
    return labels[np.argmax(counts)]
