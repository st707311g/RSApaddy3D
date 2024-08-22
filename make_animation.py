from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import coloredlogs
import napari
from napari_animation import Animation

from st_modules.config import logger
from st_modules.draw import draw_and_get_np_vol
from st_modules.rice_RSA import RiceRSA
from st_modules.rsa_dataset import RSA_Dataset, walk_to_find_rsa_dataset


@dataclass
class Args:
    src: Path
    series_in: str
    series_out: str
    debug: bool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--series_in", type=str, default="root_paths")
    parser.add_argument("--series_out", type=str, default="animation")
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

    for dataset, be_skipped in walk_to_find_rsa_dataset(args.src, [args.series_in], [args.series_out]):
        relative_path = dataset._config.path.relative_to(args.src)
        logger.info(f"target: {relative_path=}")

        if be_skipped:
            logger.info("skipped: already processed.")
            continue

        proceed(dataset, relative_path, args)


def proceed(dataset: RSA_Dataset, relative_path: Path, args: Args):
    rice_rsa = RiceRSA()
    rice_rsa.load(dataset.get_series_path(args.series_in))

    np_vol = draw_and_get_np_vol(rice_rsa)

    print(np_vol.shape)

    v = napari.Viewer(ndisplay=3)
    v.add_image(
        np_vol,
        colormap=("viridis"),
        scale=tuple([rice_rsa.mm_resolution] * 3),
        contrast_limits=[0, 210],
    )
    for act in v.window.window_menu.actions():
        if act.isChecked():
            act.trigger()

    v.window.resize(1200, 1400)
    v.camera.angles = (0, 0, 180)
    v.camera.zoom = 4  # // change it according to volume size
    v.scale_bar.visible = True
    v.scale_bar.unit = "mm"

    animation = Animation(v)
    animation.capture_keyframe()
    v.camera.angles = (0.0, 180, 180)
    animation.capture_keyframe(steps=120)
    v.camera.angles = (0.0, 360, 180)
    animation.capture_keyframe(steps=120)
    dst_path = dataset.create_new_series(args.series_out, ".mp4")
    animation.animate(dst_path, canvas_only=True, fps=30)

    v.close_all()


if __name__ == "__main__":
    main()
