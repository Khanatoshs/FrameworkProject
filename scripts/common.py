from __future__ import annotations

import configparser
import json
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import rasterio
from shapely import make_valid
from shapely.geometry import GeometryCollection
from shapely.ops import unary_union


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path | None = None) -> configparser.ConfigParser:
    path = Path(config_path) if config_path else PROJECT_ROOT / "config.ini"
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    config = configparser.ConfigParser(interpolation=None)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open(encoding="utf-8") as file:
        config.read_file(file)
    required = ("paths", "PLOTS", "labels", "tiling", "processing", "outputs")
    missing = [section for section in required if section not in config]
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")
    return config


def configured_path(config: configparser.ConfigParser, key: str) -> Path:
    value = config.get("paths", key).strip()
    if not value:
        raise ValueError(f"Set [paths] {key} in config.ini")
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def plots_enabled(config: configparser.ConfigParser) -> bool:
    return config.getboolean("PLOTS", "enabled")


def raw_rasters(config: configparser.ConfigParser) -> list[Path]:
    directory = configured_path(config, "raster_dir")
    pattern = config.get("paths", "raster_pattern").strip()
    if not directory.is_dir():
        raise NotADirectoryError(f"Raster directory not found: {directory}")
    paths = sorted(
        path for path in directory.glob(pattern)
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )
    if not paths:
        raise FileNotFoundError(
            f"No GeoTIFF rasters matching {pattern!r} in {directory}"
        )
    return paths


def processing_rasters(config: configparser.ConfigParser) -> list[Path]:
    if plots_enabled(config):
        directory = output_dir(config) / "plot_rasters"
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"No plot rasters found in {directory}; run scripts/cut_plots.py first"
            )
        names = json.loads(manifest_path.read_text(encoding="utf-8"))
        paths = [directory / name for name in names]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Plot raster files listed in the manifest are missing: "
                + ", ".join(map(str, missing[:20]))
            )
        if not paths:
            raise ValueError(f"No plot rasters were recorded in {manifest_path}")
        return paths
    return raw_rasters(config)


def output_dir(config: configparser.ConfigParser) -> Path:
    return configured_path(config, "output_dir")


def read_polygon_layer(
    path: Path,
    required_field: str | None = None,
    repair_invalid: bool = False,
) -> gpd.GeoDataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Vector file not found: {path}")
    frame = gpd.read_file(path)
    if frame.empty:
        raise ValueError(f"Vector file contains no features: {path}")
    if frame.crs is None:
        raise ValueError(f"Vector file has no CRS: {path}")
    if required_field and required_field not in frame.columns:
        raise ValueError(
            f"Required field {required_field!r} is missing from {path}. "
            f"Available fields: {', '.join(map(str, frame.columns))}"
        )
    if frame.geometry.isna().any() or frame.geometry.is_empty.any():
        bad = frame.index[frame.geometry.isna() | frame.geometry.is_empty].tolist()
        raise ValueError(f"Null or empty geometries in {path} at rows: {bad[:20]}")

    geometries = []
    invalid_rows = []
    for row_index, geometry in frame.geometry.items():
        if not geometry.is_valid:
            if not repair_invalid:
                invalid_rows.append(row_index)
                geometries.append(geometry)
                continue
            geometry = _polygonal(make_valid(geometry))
        if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError(
                f"Expected polygon geometry in {path} at row {row_index}; "
                f"found {geometry.geom_type}"
            )
        geometries.append(geometry)
    if invalid_rows:
        raise ValueError(
            f"Invalid polygon geometries in {path} at rows {invalid_rows[:20]}. "
            "Correct them or enable [labels] repair_invalid_geometries."
        )
    frame = frame.copy()
    frame.geometry = geometries
    return frame


def _polygonal(geometry):
    if geometry.geom_type in {"Polygon", "MultiPolygon"}:
        return geometry
    if isinstance(geometry, GeometryCollection):
        parts = [
            item
            for item in geometry.geoms
            if item.geom_type in {"Polygon", "MultiPolygon"} and not item.is_empty
        ]
        if parts:
            return unary_union(parts)
    raise ValueError("Geometry repair did not produce a polygonal geometry")


def plot_records(
    frame: gpd.GeoDataFrame, id_field: str | None
) -> list[tuple[int, str, object]]:
    if id_field:
        if id_field not in frame.columns:
            raise ValueError(
                f"Plot ID field {id_field!r} is missing. "
                f"Available fields: {', '.join(map(str, frame.columns))}"
            )
        values = frame[id_field]
        if values.isna().any():
            raise ValueError(f"Plot ID field {id_field!r} contains null values")
        identifiers = values.astype(str).tolist()
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"Plot ID field {id_field!r} must contain unique values")
    else:
        identifiers = [str(index + 1) for index in range(len(frame))]
    return [
        (index + 1, identifiers[index], row.geometry)
        for index, row in enumerate(frame.itertuples())
    ]


def check_raster_alignment(paths: Iterable[Path]) -> None:
    paths = list(paths)
    if len(paths) < 2:
        if paths:
            print(f"Found one raster; alignment check not needed: {paths[0]}")
        return

    with rasterio.open(paths[0]) as reference:
        if reference.crs is None:
            raise ValueError(f"Raster has no CRS: {paths[0]}")
        reference_crs = reference.crs
        reference_transform = reference.transform
        reference_grid = (
            reference_transform.a,
            reference_transform.b,
            reference_transform.d,
            reference_transform.e,
        )

    mismatches = []
    for path in paths[1:]:
        with rasterio.open(path) as dataset:
            if dataset.crs is None:
                mismatches.append(f"{path}: missing CRS")
                continue
            if dataset.crs != reference_crs:
                mismatches.append(f"{path}: CRS differs from {paths[0]}")
                continue
            transform = dataset.transform
            grid = (transform.a, transform.b, transform.d, transform.e)
            if any(
                abs(actual - expected) > 1e-9
                for actual, expected in zip(grid, reference_grid)
            ):
                mismatches.append(f"{path}: pixel size or rotation differs")
                continue
            column, row = (~reference_transform) * (transform.c, transform.f)
            if (
                abs(column - round(column)) > 1e-6
                or abs(row - round(row)) > 1e-6
            ):
                mismatches.append(f"{path}: origin is off the reference pixel grid")
    if mismatches:
        raise ValueError("Rasters are not aligned:\n- " + "\n- ".join(mismatches))
    print(f"Aligned raster grid verified for {len(paths)} rasters.")
