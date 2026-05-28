from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def _default_project_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "config" / "countries").exists():
        return cwd
    for parent in Path(__file__).resolve().parents:
        if (parent / "config" / "countries").exists():
            return parent
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = Path(os.environ.get("AGROCLIMATE_PROJECT_ROOT", _default_project_root())).resolve()
DATA_ROOT = Path(os.environ.get("AGROCLIMATE_DATA_ROOT", PROJECT_ROOT / "data")).resolve()
DEFAULT_COUNTRY = os.environ.get("AGROCLIMATE_COUNTRY", "ethiopia")


def country_slug(country: str | None = None) -> str:
    value = country or DEFAULT_COUNTRY
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def country_root(country: str | None = None) -> Path:
    return DATA_ROOT / "countries" / country_slug(country)


def country_config_path(country: str | None = None) -> Path:
    return PROJECT_ROOT / "config" / "countries" / f"{country_slug(country)}.yaml"


@lru_cache(maxsize=None)
def load_country_config(country: str | None = None) -> dict[str, Any]:
    path = country_config_path(country)
    if not path.exists():
        raise FileNotFoundError(f"Country config not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def country_name(country: str | None = None) -> str:
    return str(load_country_config(country).get("name", country_slug(country).title()))


def country_bbox_lonlat(country: str | None = None) -> list[float]:
    return list(load_country_config(country)["bbox"]["lon_lat"])


def country_cds_area(country: str | None = None) -> list[float]:
    return list(load_country_config(country)["bbox"]["cds_area_nwse"])


def cds_root(country: str | None = None) -> Path:
    return country_root(country) / "seasonal" / "cds"


def boundaries_dir(country: str | None = None) -> Path:
    return country_root(country) / "boundaries"


def boundary_path(country: str | None = None, admin_level: int = 3) -> Path:
    slug = country_slug(country)
    return boundaries_dir(country) / f"{slug}_adm{admin_level}_gadm.gpkg"


def reports_dir(country: str | None = None) -> Path:
    return country_root(country) / "reports"


def regional_openmeteo_root() -> Path:
    return DATA_ROOT / "regional" / "africa" / "openmeteo"


def configure_matplotlib_cache() -> Path:
    cache_dir = DATA_ROOT / "cache" / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    return cache_dir
