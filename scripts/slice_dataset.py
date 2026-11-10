from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import box

if __package__:
    from .common import load_config, output_dir, plots_enabled, processing_rasters
else:
    from common import load_config, output_dir, plots_enabled, processing_rasters


def _rle(mask: np.ndarray) -> dict:
    flat = mask.astype(np.uint8, copy=False).ravel(order="F")
    counts = []
    current = 0
    run_length = 0
    for value in flat:
        value = int(value)
        if value == current:
            run_length += 1
        else:
            counts.append(run_length)
            run_length = 1
            current = value
    counts.append(run_length)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def _tile_starts(length: int, step: int) -> range:
    return range(0, length, step)


def _write_tile(
    source: rasterio.io.DatasetReader,
    window: Window,
    destination: Path,
) -> tuple[np.ndarray, Affine]:
    tile_size = int(window.width)
    data = source.read(
        window=window,
        boundless=True,
        fill_value=source.nodata if source.nodata is not None else 0,
    )
    band_masks = source.read_masks(window=window, boundless=True)
    valid = (band_masks > 0).any(axis=0)
    transform = source.window_transform(window)
    profile = source.profile.copy()
    profile.update(
        driver="GTiff",
        width=tile_size,
        height=tile_size,
        transform=transform,
        compress="deflate",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(destination, "w", **profile) as tile:
        tile.write(data)
        tile.write_mask(valid.astype(np.uint8) * 255)
        tile.update_tags(
            pipeline_source_raster=source.name,
            pipeline_window_col=str(int(window.col_off)),
            pipeline_window_row=str(int(window.row_off)),
        )
    return valid, transform


def _tile_candidates(
    labels: gpd.GeoDataFrame,
    spatial_index,
    bounds,
    plot_index: int | None,
) -> list:
    candidates = spatial_index.query(bounds, predicate="intersects")
    rows = []
    for position in candidates:
        row = labels.iloc[int(position)]
        if plot_index is not None and int(row["plot_index"]) != plot_index:
            continue
        rows.append(row)
    rows.sort(key=lambda row: int(row["instance_id"]))
    return rows


def run(config_path=None) -> None:
    config = load_config(config_path)
    output = output_dir(config)
    prepared = output / "prepared"
    labels_path = prepared / "labels.gpkg"
    classes_path = prepared / "classes.json"
    if not labels_path.is_file() or not classes_path.is_file():
        raise FileNotFoundError(
            "Prepared labels are missing; run scripts/process_shapefile.py first"
        )
    labels = gpd.read_file(labels_path, layer="instances")
    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    if labels.crs is None:
        raise ValueError(f"Prepared labels have no CRS: {labels_path}")
    spatial_index = labels.sindex
    tile_size = config.getint("tiling", "tile_size")
    overlap = config.getint("tiling", "overlap_pixels")
    min_pixels = config.getint("processing", "min_instance_pixels")
    all_touched = config.getboolean("processing", "all_touched")
    include_empty = config.getboolean("outputs", "include_empty_tiles")
    if tile_size <= 0:
        raise ValueError("[tiling] tile_size must be greater than zero")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("[tiling] overlap_pixels must be >= 0 and < tile_size")
    if min_pixels < 1:
        raise ValueError("[processing] min_instance_pixels must be at least one")
    step = tile_size - overlap
    image_dir = output / "images"
    mask_dir = output / "instance_masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    coco = {
        "info": {"description": "Georeferenced forestry instance-segmentation tiles"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": classes,
    }
    image_id = 1
    annotation_id = 1
    for raster_path in processing_rasters(config):
        with rasterio.open(raster_path) as source:
            if source.crs is None:
                raise ValueError(f"Raster has no CRS: {raster_path}")
            raster_labels = labels.to_crs(source.crs)
            source_tags = source.tags()
            plot_index = (
                int(source_tags["pipeline_plot_index"])
                if plots_enabled(config)
                else None
            )
            if plots_enabled(config) and plot_index is None:
                raise ValueError(
                    f"Plot raster is missing pipeline_plot_index metadata: {raster_path}"
                )

            raster_spatial_index = raster_labels.sindex
            for row in _tile_starts(source.height, step):
                for col in _tile_starts(source.width, step):
                    window = Window(col, row, tile_size, tile_size)
                    tile_name = f"{raster_path.stem}__r{row:06d}_c{col:06d}"
                    image_path = image_dir / f"{tile_name}.tif"
                    valid_pixels, transform = _write_tile(source, window, image_path)
                    tile_bounds = rasterio.transform.array_bounds(
                        tile_size, tile_size, transform
                    )
                    candidates = _tile_candidates(
                        raster_labels,
                        raster_spatial_index,
                        box(*tile_bounds),
                        plot_index,
                    )
                    instance_masks = []
                    instance_ids = []
                    category_ids = []
                    tile_annotations = []
                    for label in candidates:
                        instance_mask = rasterize(
                            [(label.geometry, 1)],
                            out_shape=(tile_size, tile_size),
                            transform=transform,
                            fill=0,
                            all_touched=all_touched,
                            dtype="uint8",
                        )
                        instance_mask &= valid_pixels.astype(np.uint8)
                        area = int(instance_mask.sum())
                        if area < min_pixels:
                            continue
                        ys, xs = np.nonzero(instance_mask)
                        x_min, x_max = int(xs.min()), int(xs.max())
                        y_min, y_max = int(ys.min()), int(ys.max())
                        category_id = int(label["category_id"])
                        tile_annotations.append(
                            {
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": category_id,
                                "segmentation": _rle(instance_mask),
                                "area": area,
                                "bbox": [
                                    x_min,
                                    y_min,
                                    x_max - x_min + 1,
                                    y_max - y_min + 1,
                                ],
                                "iscrowd": 0,
                            }
                        )
                        annotation_id += 1
                        instance_masks.append(instance_mask)
                        instance_ids.append(int(label["instance_id"]))
                        category_ids.append(category_id)

                    if not tile_annotations and not include_empty:
                        image_path.unlink()
                        mask_sidecar = Path(f"{image_path}.msk")
                        if mask_sidecar.exists():
                            mask_sidecar.unlink()
                        continue

                    stack = (
                        np.stack(instance_masks, axis=0)
                        if instance_masks
                        else np.zeros((0, tile_size, tile_size), dtype=np.uint8)
                    )
                    mask_path = mask_dir / f"{tile_name}.npz"
                    np.savez_compressed(
                        mask_path,
                        masks=stack,
                        instance_ids=np.asarray(instance_ids, dtype=np.int64),
                        category_ids=np.asarray(category_ids, dtype=np.int32),
                        crs=np.asarray(source.crs.to_wkt()),
                        transform=np.asarray(tuple(transform)[:6], dtype=np.float64),
                    )
                    coco["images"].append(
                        {
                            "id": image_id,
                            "file_name": image_path.relative_to(output).as_posix(),
                            "width": tile_size,
                            "height": tile_size,
                            "crs": source.crs.to_string(),
                            "transform": list(tuple(transform)[:6]),
                            "instance_masks": mask_path.relative_to(output).as_posix(),
                        }
                    )
                    coco["annotations"].extend(tile_annotations)
                    image_id += 1
    annotations_path = output / "annotations.json"
    annotations_path.write_text(
        json.dumps(coco, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(coco['images'])} tiles and "
        f"{len(coco['annotations'])} instances to {output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tile imagery and create per-instance masks and COCO annotations."
    )
    parser.add_argument("--config", help="Path to config.ini")
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
