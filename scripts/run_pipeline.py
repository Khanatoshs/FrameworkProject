from __future__ import annotations

import argparse

if __package__:
    from .check_rasters import run as check_rasters
    from .common import load_config, plots_enabled
    from .cut_plots import run as cut_plots
    from .process_shapefile import run as process_shapefile
    from .slice_dataset import run as slice_dataset
else:
    from check_rasters import run as check_rasters
    from common import load_config, plots_enabled
    from cut_plots import run as cut_plots
    from process_shapefile import run as process_shapefile
    from slice_dataset import run as slice_dataset


def run(config_path=None) -> None:
    config = load_config(config_path)
    check_rasters(config_path)
    if plots_enabled(config):
        cut_plots(config_path)
    process_shapefile(config_path)
    slice_dataset(config_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full forestry data pipeline.")
    parser.add_argument("--config", help="Path to config.ini")
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
