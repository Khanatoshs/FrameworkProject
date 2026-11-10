from __future__ import annotations

import argparse
import json

import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import box

if __package__:
    from .common import (
        configured_path,
        load_config,
        output_dir,
        plot_records,
        plots_enabled,
        raw_rasters,
        read_polygon_layer,
    )
else:
    from common import (
        configured_path,
        load_config,
        output_dir,
        plot_records,
        plots_enabled,
        raw_rasters,
        read_polygon_layer,
    )


def run(config_path=None) -> None:
    config = load_config(config_path)
    if not plots_enabled(config):
        print("PLOTS.enabled is false; skipping plot clipping.")
        return

    plots_path = configured_path(config, "plots_shapefile")
    repair = config.getboolean("labels", "repair_invalid_geometries")
    id_field = config.get("PLOTS", "id_field").strip() or None
    plots = read_polygon_layer(plots_path, repair_invalid=repair)
    target_dir = output_dir(config) / "plot_rasters"
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = target_dir / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()
    all_touched = config.getboolean("processing", "all_touched")
    written_names = []
    covered_plot_indices = set()

    for raster_path in raw_rasters(config):
        with rasterio.open(raster_path) as source:
            if source.crs is None:
                raise ValueError(f"Raster has no CRS: {raster_path}")
            raster_plots = plots.to_crs(source.crs)
            raster_records = plot_records(raster_plots, id_field)
            raster_bounds = box(*source.bounds)
            for plot_index, plot_id, geometry in raster_records:
                if geometry.intersection(raster_bounds).area == 0:
                    continue
                data, transform = mask(
                    source,
                    [geometry],
                    crop=True,
                    all_touched=all_touched,
                    filled=False,
                )
                profile = source.profile.copy()
                profile.update(
                    driver="GTiff",
                    height=data.shape[1],
                    width=data.shape[2],
                    transform=transform,
                    compress="deflate",
                )
                destination = target_dir / (
                    f"{raster_path.stem}__plot_{plot_index:06d}.tif"
                )
                mask_array = np.ma.getmaskarray(data)
                valid_pixels = (~mask_array).any(axis=0).astype("uint8") * 255
                with rasterio.open(destination, "w", **profile) as output:
                    output.write(np.ma.getdata(data))
                    output.write_mask(valid_pixels)
                    output.update_tags(
                        pipeline_plot_index=str(plot_index),
                        pipeline_plot_id=plot_id,
                        pipeline_source_raster=raster_path.name,
                    )
                written_names.append(destination.name)
                covered_plot_indices.add(plot_index)
                print(f"Wrote {destination}")

    if not written_names:
        raise ValueError("No plot polygons intersected any configured raster")
    manifest_path.write_text(
        json.dumps(written_names, indent=2) + "\n", encoding="utf-8"
    )
    uncovered = sorted(
        plot_index
        for plot_index, _, _ in plot_records(plots, id_field)
        if plot_index not in covered_plot_indices
    )
    if uncovered:
        print(f"Warning: plots with no raster overlap were skipped: {uncovered}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clip rasters to plot polygons.")
    parser.add_argument("--config", help="Path to config.ini")
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
