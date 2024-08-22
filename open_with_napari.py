from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import coloredlogs
import napari

from st_modules.config import logger
from st_modules.draw import draw_and_get_np_vol
from st_modules.rice_RSA import RiceRSA


@dataclass
class Args:
    src: Path
    debug: bool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--debug", action="store_true")
    args = Args(**parser.parse_args().__dict__)

    if args.debug:
        coloredlogs.install(level=logging.DEBUG, logger=logger)
    else:
        coloredlogs.install(level=logging.INFO, logger=logger)

    logger.debug(args.__dict__)

    if not args.src.exists():
        logger.error("indicate valid path.")
        exit()

    if str(args.src).lower().endswith(".json"):
        json_path = args.src
    else:
        logger.info("indicate json file.")
        exit()

    rice_rsa = RiceRSA()
    rice_rsa.load(json_path)

    np_vol = draw_and_get_np_vol(rice_rsa)

    v = napari.Viewer(ndisplay=3)
    v.add_image(
        np_vol,
        colormap=("viridis"),
        scale=tuple([rice_rsa.mm_resolution] * 3),
        contrast_limits=[0, 210],
    )

    napari.run()


if __name__ == "__main__":
    main()
