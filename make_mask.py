"""
!!! NOTE !!!
This source code works with CT images taken under specific conditions.
If you need to use it with other images, please modify the codes.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import warnings
import zipfile
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory

import coloredlogs
import numpy as np
from skimage import io

from st_modules.config import logger
from st_modules.multi_threading import ForEach, MultiThreading
from st_modules.rsa_dataset import RSA_Dataset, walk_to_find_rsa_dataset
from st_modules.volume import VolumeData

warnings.filterwarnings("ignore")


@dataclass
class Args:
    src: Path
    mask_thr: int
    closing_itr: int
    erosion_itr: int
    series_src: str
    series_dst: str
    archive: bool
    debug: bool


@dataclass
class MaskMaker(object):
    _src_path: Path
    _dst_path: Path
    _layer_size: int = 128
    _overlap: int = 24
    _label: str = None
    _using_gpu_id: int = None

    def __post_init__(self):
        if self._src_path.is_dir():
            self._src_dir_to_load = Path(self._src_path)
        else:
            self._src_tmpdir = TemporaryDirectory()
            with zipfile.ZipFile(self._src_path, "r") as zf:
                zf.extractall(self._src_tmpdir.name)
            self._src_dir_to_load = Path(self._src_tmpdir.name)

        self._voxel_counts = [-1] * self._image_file_number

    @cached_property
    def _image_file_names(self):
        return VolumeData(self._src_dir_to_load).image_files

    @cached_property
    def _image_file_number(self):
        return len(self._image_file_names)

        self.print_callback_i = 0

    def _print_callback(self, i: int, total: int, label="processing"):
        percent = round((i + 1) * 100 / total, 1)
        print(f"\r{label}: {percent}%", end="")
        if i + 1 == total:
            print("")

    def _get_mask_voxel_counts(self, file, threshold: int):
        img = io.imread(file)
        return np.sum(img > threshold)

    def run(self, threshold: int, closing_iteration: int, erosion_iteration: int):
        temp_dst_dir = TemporaryDirectory()
        temp_dst_file_index = 0

        with MultiThreading() as mt:
            mask_voxel_counts = mt.run([ForEach(self._image_file_names), threshold], self._get_mask_voxel_counts)

        start_slice_index = self._image_file_number // 2
        middle_count = mask_voxel_counts[start_slice_index]

        while start_slice_index > 0:
            start_slice_index -= 1
            if mask_voxel_counts[start_slice_index] * 2 < middle_count:
                break

        start_slice_index += 1

        end_slice_index = self._image_file_number // 2

        while end_slice_index < self._image_file_number - 1:
            end_slice_index += 1
            if mask_voxel_counts[end_slice_index] * 2 < middle_count:
                break

        for i in range(self._layer_number):
            self._print_callback(i, self._layer_number, label="making mask")
            layer = self._get_layer(i)
            mask = layer > threshold
            del layer

            if self._using_gpu_id is None:
                from scipy.ndimage import binary_closing

                mask = binary_closing(
                    mask,
                    iterations=closing_iteration,
                    brute_force=True,
                )

            else:
                import cupy as cp
                from cupyx.scipy.ndimage import binary_closing

                mask = binary_closing(
                    cp.asarray(mask),
                    iterations=closing_iteration,
                    brute_force=True,
                ).get()

            if self._using_gpu_id is None:
                from skimage.segmentation import clear_border

                def func(mask):
                    inv_mask = mask == 0
                    inv_mask_no_border = clear_border(inv_mask)
                    border = np.bitwise_xor(inv_mask, inv_mask_no_border)

                    return border == 0

                with MultiThreading() as mt:
                    mask = np.asarray(mt.run([ForEach(mask)], func))
            else:
                from cucim.skimage.segmentation import clear_border

                for i in range(len(mask)):
                    inv_mask = cp.asarray(mask[i]) == 0
                    inv_mask_no_border = clear_border(inv_mask)
                    border = cp.bitwise_xor(inv_mask, inv_mask_no_border)

                    mask[i] = (border == 0).get()

            if self._using_gpu_id is None:
                from scipy.ndimage import binary_erosion

                mask = binary_erosion(
                    mask,
                    iterations=erosion_iteration,
                    brute_force=True,
                )
            else:
                from cupyx.scipy.ndimage import binary_erosion

                mask = binary_erosion(
                    cp.asarray(mask),
                    iterations=erosion_iteration,
                    brute_force=True,
                ).get()

            mask = mask[self._overlap : -self._overlap]

            dst_file_list = []
            max_i = -1
            for i_, slice_ in enumerate(mask):
                if 0 <= temp_dst_file_index < self._shape[0]:
                    dst_file = Path(temp_dst_dir.name, f"img_{temp_dst_file_index:04}.png")
                    dst_file_list.append(dst_file)
                    max_i = i_ if i_ > max_i else max_i

                    if temp_dst_file_index < start_slice_index or temp_dst_file_index >= end_slice_index:
                        mask[i_] = False

                temp_dst_file_index += 1

            mask = np.asarray(mask * 255, dtype=np.uint8)
            with MultiThreading() as mt:
                mt.run([ForEach(dst_file_list), ForEach(mask[0 : max_i + 1])], io.imsave)

        if str(self._dst_path).endswith(".zip"):
            with zipfile.ZipFile(self._dst_path, "w", zipfile.ZIP_STORED) as zf:
                for p in sorted(Path(temp_dst_dir.name).glob("*")):
                    zf.write(p, p.name)
        else:
            self._dst_path.mkdir(parents=True, exist_ok=True)
            for p in sorted(Path(temp_dst_dir.name).glob("*")):
                shutil.copyfile(p, Path(self._dst_path, p.name))

        temp_dst_dir.cleanup()

    @property
    def _inner_size(self):
        return self._layer_size - self._overlap * 2

    @property
    def _layer_number(self):
        return int(np.ceil(self._image_file_number / self._inner_size))

    def _get_layer(self, layer_index: int):
        indexes = [
            i
            for i in range(
                layer_index * self._inner_size - self._overlap,
                (layer_index + 1) * self._inner_size + self._overlap,
            )
        ]
        indexes = [i if i >= 0 else -i for i in indexes]
        indexes = [i if i < self._image_file_number else self._image_file_number * 2 - i - 2 for i in indexes]

        img_files = [self._image_file_names[i] for i in indexes]

        with MultiThreading() as mt:
            layer = np.asarray(mt.run([ForEach(img_files)], io.imread))

        return layer

    @cached_property
    def _shape(self):
        m_img = io.imread(self._image_file_names[self._image_file_number // 2])
        return (self._image_file_number,) + m_img.shape

    @cached_property
    def shape_with_block_unit(self):
        return tuple([self._inner_size * ((s + self._inner_size - 1) // self._inner_size) for s in self.shape])

    def __del__(self):
        try:
            self.tmpdir.cleanup()
        except:  # noqa
            pass


def main():
    parser = argparse.ArgumentParser(description="RSA mask maker")
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--mask_thr", type=int, default=100)
    parser.add_argument("--closing_itr", type=int, default=10)
    parser.add_argument("--erosion_itr", type=int, default=5)
    parser.add_argument("--series_src", type=str, default="ct")
    parser.add_argument("--series_dst", type=str, default="mask")
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

    for dataset, be_skipped in walk_to_find_rsa_dataset(args.src, [args.series_src], [args.series_dst]):
        relative_path = dataset._config.path.relative_to(args.src)
        logger.info(f"target: {relative_path=}")

        if be_skipped:
            logger.info("skipped: already processed.")
            continue

        proceed(dataset, relative_path, args)


def proceed(dataset: RSA_Dataset, relative_path: Path, args: Args):
    dst_path = dataset.create_new_series(args.series_dst, ".zip" if args.archive else "")
    logger.debug(f"{dst_path=}")

    MaskMaker(
        _src_path=dataset.get_series_path(args.series_src),
        _dst_path=dst_path,
        _label=relative_path,
        _using_gpu_id=0,
    ).run(
        threshold=args.mask_thr,
        closing_iteration=args.closing_itr,
        erosion_iteration=args.erosion_itr,
    )

    dataset.update_log(
        {
            dst_path.stem: {
                "threshold": args.mask_thr,
                "closing_iteration": args.closing_itr,
                "erosion_iteration": args.erosion_itr,
            }
        }
    )


if __name__ == "__main__":
    main()
