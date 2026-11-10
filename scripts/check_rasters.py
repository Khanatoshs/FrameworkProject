from __future__ import annotations

import argparse

if __package__:
    from .common import check_raster_alignment, load_config, raw_rasters
else:
    from common import check_raster_alignment, load_config, raw_rasters


def run(config_path=None) -> None:
    config = load_config(config_path)
    check_raster_alignment(raw_rasters(config))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check GeoTIFF pixel-grid alignment.")
    parser.add_argument("--config", help="Path to config.ini")
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
