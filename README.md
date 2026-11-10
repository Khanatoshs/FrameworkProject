# FrameworkProject

Forestry and AI data preparation tools.

## Raster-to-Mask R-CNN pipeline

The pipeline validates raster pixel-grid alignment, optionally clips rasters to
plot polygons, validates species polygons, then writes fixed-size GeoTIFF tiles,
COCO annotations, and overlap-preserving per-instance mask arrays.

### Setup

1. Install the dependencies with `python -m pip install -r requirements.txt`.
2. Put input GeoTIFFs in `data/rasters/`, plot polygons in `data/plots.shp`
   (when enabled), and species polygons in `data/labels.shp`.
3. Edit `config.ini`. Set `labels.species_field` to the field containing the
   species names. Set `PLOTS.enabled = true` to clip imagery and labels to plots.
   `PLOTS.id_field` is optional; when set, it must contain unique plot IDs.
4. Run `python scripts/run_pipeline.py`.

All relative paths in `config.ini` are resolved from the project root.
Run the automated checks with `python -m unittest discover -s tests`.

### Stages

- `python scripts/check_rasters.py`: check that multiple rasters share a CRS,
  pixel size, rotation, and pixel-aligned origin. Different extents are allowed.
  A single raster skips the alignment check. The pipeline fails on misalignment;
  it does not silently resample source imagery.
- `python scripts/cut_plots.py`: when `PLOTS.enabled` is true, clip each raster
  into one GeoTIFF per plot. This stage can be run independently.
- `python scripts/process_shapefile.py`: validate polygon geometry and species
  values, assign stable class IDs, and write normalized instance polygons.
  Invalid geometries fail by default; `labels.repair_invalid_geometries = true`
  enables Shapely geometry repair.
- `python scripts/slice_dataset.py`: create image tiles, per-instance masks, and
  `output/annotations.json`. This stage expects the previous label-preparation
  step to have completed.

`tiling.tile_size` is measured in pixels. `tiling.overlap_pixels` is the number
of shared pixels between adjacent tiles. Tile dimensions are padded at raster
edges; the GeoTIFF validity mask marks padding as invalid. Set
`outputs.include_empty_tiles = false` to omit tiles with no visible instances.
`processing.all_touched` controls whether polygons touching a pixel are
rasterized into it, and `processing.min_instance_pixels` filters tiny masks.

The COCO annotations use uncompressed COCO RLE. Each `.npz` in
`output/instance_masks/` contains a `masks` array shaped `(instances, height,
width)`, `instance_ids`, and `category_ids`. Separate masks preserve overlapping
instances; class IDs and names are in `output/prepared/classes.json`. Image
tiles are in `output/images/`, and prepared GeoPackage labels are in
`output/prepared/labels.gpkg`.
