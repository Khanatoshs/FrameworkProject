from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from scripts.common import check_raster_alignment
from scripts.run_pipeline import run
from scripts.slice_dataset import _rle


class PipelineTest(unittest.TestCase):
    def _write_raster(self, path: Path, transform, size: int = 32) -> None:
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=size,
            height=size,
            count=3,
            dtype="uint8",
            crs="EPSG:32631",
            transform=transform,
        ) as dataset:
            dataset.write(np.full((3, size, size), 100, dtype=np.uint8))

    def _write_config(
        self,
        root: Path,
        raster_dir: Path,
        plots_path: Path,
        labels_path: Path,
        output_dir: Path,
        plots_enabled: bool,
        tile_size: int = 16,
        overlap: int = 4,
    ) -> Path:
        path = root / "config.ini"
        path.write_text(
            f"""[paths]
raster_dir = {raster_dir}
raster_pattern = *.tif
plots_shapefile = {plots_path}
labels_shapefile = {labels_path}
output_dir = {output_dir}

[PLOTS]
enabled = {str(plots_enabled).lower()}
id_field = plot_id

[labels]
species_field = species
repair_invalid_geometries = false

[tiling]
tile_size = {tile_size}
overlap_pixels = {overlap}

[processing]
all_touched = false
min_instance_pixels = 1

[outputs]
include_empty_tiles = true
""",
            encoding="utf-8",
        )
        return path

    def test_alignment_accepts_integer_pixel_shift_and_rejects_fractional_shift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aligned = root / "aligned.tif"
            shifted = root / "shifted.tif"
            fractional = root / "fractional.tif"
            self._write_raster(aligned, from_origin(0, 32, 1, 1))
            self._write_raster(shifted, from_origin(4, 32, 1, 1))
            self._write_raster(fractional, from_origin(0.5, 32, 1, 1))

            check_raster_alignment([aligned, shifted])
            with self.assertRaisesRegex(ValueError, "off the reference pixel grid"):
                check_raster_alignment([aligned, fractional])

    def test_plot_pipeline_preserves_overlapping_instance_masks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raster_dir = root / "rasters"
            raster_dir.mkdir()
            self._write_raster(
                raster_dir / "scene.tif", from_origin(0, 32, 1, 1)
            )
            plots_path = root / "plots.shp"
            gpd.GeoDataFrame(
                {"plot_id": ["plot-1"]},
                geometry=[box(4, 4, 28, 28)],
                crs="EPSG:32631",
            ).to_file(plots_path)
            labels_path = root / "labels.shp"
            gpd.GeoDataFrame(
                {"species": ["oak", "pine", "outside"]},
                geometry=[
                    box(8, 8, 22, 22),
                    box(14, 14, 24, 24),
                    box(50, 50, 55, 55),
                ],
                crs="EPSG:32631",
            ).to_file(labels_path)
            output = root / "output"
            config = self._write_config(
                root,
                raster_dir,
                plots_path,
                labels_path,
                output,
                plots_enabled=True,
            )

            run(config)
            run(config)

            coco = json.loads((output / "annotations.json").read_text())
            self.assertEqual(len(coco["images"]), 4)
            self.assertEqual(
                {category["name"] for category in coco["categories"]},
                {"oak", "pine"},
            )
            overlap_found = False
            padded_tile_found = False
            for image in coco["images"]:
                with np.load(output / image["instance_masks"]) as masks:
                    self.assertEqual(masks["masks"].shape[1:], (16, 16))
                    self.assertEqual(
                        masks["masks"].shape[0], len(masks["category_ids"])
                    )
                    overlap_found |= bool((masks["masks"].sum(axis=0) > 1).any())
                with rasterio.open(output / image["file_name"]) as tile:
                    col = int(tile.tags()["pipeline_window_col"])
                    if col == 12:
                        padded_tile_found |= bool((tile.read_masks(1)[:, 12:] == 0).all())
            self.assertTrue(overlap_found)
            self.assertTrue(padded_tile_found)

    def test_pipeline_runs_without_plot_inputs_when_plots_are_disabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raster_dir = root / "rasters"
            raster_dir.mkdir()
            self._write_raster(
                raster_dir / "scene.tif", from_origin(0, 32, 1, 1)
            )
            labels_path = root / "labels.shp"
            gpd.GeoDataFrame(
                {"species": ["spruce"]},
                geometry=[box(8, 8, 14, 14)],
                crs="EPSG:32631",
            ).to_file(labels_path)
            output = root / "output"
            config = self._write_config(
                root,
                raster_dir,
                root / "plots-do-not-exist.shp",
                labels_path,
                output,
                plots_enabled=False,
                overlap=0,
            )

            run(config)

            coco = json.loads((output / "annotations.json").read_text())
            self.assertEqual(len(coco["images"]), 4)
            self.assertEqual(
                {category["name"] for category in coco["categories"]}, {"spruce"}
            )

    def test_rle_round_trips_mask_in_coco_column_major_order(self):
        mask = np.array(
            [[0, 1, 1], [1, 0, 1], [0, 0, 1]], dtype=np.uint8
        )
        encoded = _rle(mask)
        flat = np.zeros(mask.size, dtype=np.uint8)
        position = 0
        value = 0
        for count in encoded["counts"]:
            flat[position : position + count] = value
            position += count
            value = 1 - value
        decoded = flat.reshape(mask.shape, order="F")
        np.testing.assert_array_equal(decoded, mask)


if __name__ == "__main__":
    unittest.main()
