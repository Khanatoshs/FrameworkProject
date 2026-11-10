from __future__ import annotations

import argparse
import json

import geopandas as gpd
import rasterio
from shapely.ops import unary_union

if __package__:
    from .common import (
        configured_path,
        load_config,
        output_dir,
        plot_records,
        plots_enabled,
        processing_rasters,
        read_polygon_layer,
    )
else:
    from common import (
        configured_path,
        load_config,
        output_dir,
        plot_records,
        plots_enabled,
        processing_rasters,
        read_polygon_layer,
    )


def run(config_path=None) -> None:
    config = load_config(config_path)
    labels_path = configured_path(config, "labels_shapefile")
    species_field = config.get("labels", "species_field").strip()
    repair = config.getboolean("labels", "repair_invalid_geometries")
    if not species_field:
        raise ValueError("Set [labels] species_field in config.ini")
    labels = read_polygon_layer(
        labels_path, required_field=species_field, repair_invalid=repair
    )
    species_values = labels[species_field]
    if species_values.isna().any():
        bad = labels.index[species_values.isna()].tolist()
        raise ValueError(f"Null species values in {labels_path} at rows: {bad[:20]}")
    labels["_species"] = species_values.astype(str).str.strip()
    empty_species = labels["_species"].eq("")
    if empty_species.any():
        bad = labels.index[empty_species].tolist()
        raise ValueError(f"Empty species values in {labels_path} at rows: {bad[:20]}")

    rasters = processing_rasters(config)
    with rasterio.open(rasters[0]) as dataset:
        if dataset.crs is None:
            raise ValueError(f"Raster has no CRS: {rasters[0]}")
        target_crs = dataset.crs
    labels = labels.to_crs(target_crs)

    plot_geometries = None
    if plots_enabled(config):
        plots_path = configured_path(config, "plots_shapefile")
        plots = read_polygon_layer(plots_path, repair_invalid=repair).to_crs(target_crs)
        id_field = config.get("PLOTS", "id_field").strip() or None
        plot_geometries = plot_records(plots, id_field)

    records = []
    instance_id = 1
    spatial_index = labels.sindex

    if plot_geometries is None:
        for _, row in labels.iterrows():
            records.append(
                {
                    "instance_id": instance_id,
                    "species": row["_species"],
                    "plot_index": -1,
                    "geometry": row.geometry,
                }
            )
            instance_id += 1
    else:
        for plot_index, _, plot_geometry in plot_geometries:
            for position in sorted(
                spatial_index.query(plot_geometry, predicate="intersects")
            ):
                row = labels.iloc[int(position)]
                clipped = row.geometry.intersection(plot_geometry)
                if clipped.is_empty:
                    continue
                if clipped.geom_type == "GeometryCollection":
                    polygon_parts = [
                        part
                        for part in clipped.geoms
                        if part.geom_type in {"Polygon", "MultiPolygon"}
                        and not part.is_empty
                    ]
                    if not polygon_parts:
                        continue
                    clipped = unary_union(polygon_parts)
                elif clipped.geom_type not in {"Polygon", "MultiPolygon"}:
                    continue
                records.append(
                    {
                        "instance_id": instance_id,
                        "species": row["_species"],
                        "plot_index": plot_index,
                        "geometry": clipped,
                    }
                )
                instance_id += 1

    if not records:
        raise ValueError("No label polygons remain after applying the plot filter")
    species_names = sorted(
        {record["species"] for record in records},
        key=lambda value: (value.casefold(), value),
    )
    category_ids = {name: index + 1 for index, name in enumerate(species_names)}
    for record in records:
        record["category_id"] = category_ids[record["species"]]
    prepared = gpd.GeoDataFrame(records, geometry="geometry", crs=target_crs)
    destination = output_dir(config) / "prepared"
    destination.mkdir(parents=True, exist_ok=True)
    labels_output = destination / "labels.gpkg"
    prepared.to_file(labels_output, layer="instances", driver="GPKG")
    classes_output = destination / "classes.json"
    classes_output.write_text(
        json.dumps(
            [
                {"id": category_ids[name], "name": name}
                for name in species_names
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote validated instance polygons to {labels_output}")
    print(f"Wrote species classes to {classes_output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate species polygons and prepare per-instance labels."
    )
    parser.add_argument("--config", help="Path to config.ini")
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
