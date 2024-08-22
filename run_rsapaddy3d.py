import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import coloredlogs
from st_modules.config import DESCRIPTION, is_cupy_available, logger
from st_modules.rsa_dataset import RSA_Dataset, walk_to_find_rsa_dataset
from st_modules.rsapaddy3d import RSApaddy3D


@dataclass(frozen=True)
class Args:
    src: Path
    kernel_radius: int
    size_filter_threshold: int
    kernel_weight: float
    gpu: list[int]
    archive: bool
    debug: bool


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--kernel_radius", type=int, default=5)
    parser.add_argument("--size_filter_threshold", type=int, default=255)
    parser.add_argument("--kernel_weight", type=float, default=0.16)
    parser.add_argument("--gpu", type=int, nargs="*", default=None)
    parser.add_argument("--archive", action="store_true")
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

    if args.gpu:
        if is_cupy_available:
            logger.info(f"Using GPU IDs: {args.gpu}")
        else:
            logger.error("CuPy is not available")
            exit(1)

    for dataset, be_skipped in walk_to_find_rsa_dataset(args.src, ["ct", "rsavis3d", "mask"], ["rsapaddy3d"]):
        relative_path = dataset._config.path.relative_to(args.src)
        logger.info(f"target: {relative_path=}")

        if be_skipped:
            logger.info("skipped: already processed.")
            continue

        proceed(dataset, relative_path, args)


def proceed(dataset: RSA_Dataset, relative_path: Path, args: Args):
    if not dataset._config.volume_info_path.exists():
        logger.error("could not find volume info file.")
        return

    rsavis3d_path = dataset.get_series_path("rsavis3d")
    logger.debug(f"{rsavis3d_path=}")

    dst_path = dataset.create_new_series("rsapaddy3d", ".zip" if args.archive else "")
    logger.debug(f"{dst_path=}")

    volume_info = dataset.load_volume_info()
    logger.debug(f"{volume_info=}")

    mm_resolution = float(volume_info["mm_resolution"])

    rsapaddy3d = RSApaddy3D(
        src_path=rsavis3d_path,
        dst_path=dst_path,
        kernel_radius=args.kernel_radius,
        size_filter_threshold=args.size_filter_threshold,
        mm_resolution=mm_resolution,
        kernel_weight=args.kernel_weight,
        using_gpu_ids=args.gpu,
    )
    rsapaddy3d.prepare_src_imgs_dir()
    rsapaddy3d.proceed_filtering()
    rsapaddy3d.make_distance_map()
    rsapaddy3d.save_result(args.archive)

    dataset.update_log(
        {
            dst_path.stem: {
                "kernel_radius": args.kernel_radius,
                "kernel_weight": args.kernel_weight,
                "size_filter_threshold": args.size_filter_threshold,
            }
        }
    )


if __name__ == "__main__":
    main()
