import functools
import logging
import shutil
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

# from scipy import ndimage
# from scipy.ndimage import measurements
from skimage import io, morphology, util

from st_modules.config import logger
from st_modules.multi_threading import ForEach, MultiThreading, MultiThreadingGPU
from st_modules.volume import VolumeData, VolumeSeparator

"""
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cp_ndimage

    try:
        from cupyx.scipy.ndimage.measurements import label as cp_label
    except ImportError:
        from cupyx.scipy.ndimage import label as cp_label

    is_cupy_available = True
except ModuleNotFoundError:
    is_cupy_available = False
"""

logging.getLogger("PIL").setLevel(logging.INFO)

warnings.filterwarnings("ignore")


def covert_distance_transformation_value_to_root_diameter(v: float | np.ndarray):
    a = 1.98316325
    b = 0.67376092
    c = -1.99578897
    v = np.asarray(v)
    return v * 2 + (a * (v * 2 + c)) / (b + (v * 2 + c)) - 1


def logging_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import time

        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.debug(f"[elapsed] {func.__name__}: {round(end-start, 1)} sec")
        return result

    return wrapper


class TempDirs:
    def __init__(self) -> None:
        self.original = TemporaryDirectory()
        self.merged = TemporaryDirectory()
        self.distance_map = TemporaryDirectory()

    def __del__(self):
        for t in [self.original, self.merged, self.distance_map]:
            t.cleanup()


@dataclass
class RSApaddy3D(object):
    src_path: str | Path
    dst_path: str | Path
    kernel_radius: int
    size_filter_threshold: int
    mm_resolution: float
    kernel_weight: float = 0.16
    using_gpu_ids: list[int] = None

    def __post_init__(self):
        self.src_path = Path(self.src_path)
        self.dst_path = Path(self.dst_path)

        self.tmp_dirs = TempDirs()

    def prepare_src_imgs_dir(self):
        logger.info("Copying image files to a temporary directory")

        if self.src_path.is_dir():
            for f in VolumeData(self.src_path).image_files:
                shutil.copyfile(f, Path(self.tmp_dirs.original.name, f.name))
        else:
            with zipfile.ZipFile(self.src_path, "r") as zf:
                zf.extractall(self.tmp_dirs.original.name)

    def proceed_filtering(self):
        res_vols = []
        for i, dim in enumerate(["Z", "Y", "X"]):
            logger.info(f"Filtering: {dim}")
            res_vols.append(self.__proceed_filtering(Path(self.tmp_dirs.original.name), dim))

        logger.info("Merging the filtered files")
        res_vols = np.asarray(res_vols).max(axis=0)
        self.save_imgs(res_vols, Path(self.tmp_dirs.merged.name))

    def make_distance_map(self):
        skeleton = self.__make_skeleton(Path(self.tmp_dirs.merged.name))
        distance_map = self.__make_distance_map(skeleton, Path(self.tmp_dirs.original.name), self.mm_resolution)
        self.save_imgs(distance_map, Path(self.tmp_dirs.distance_map.name))

    def save_result(self, archive: bool = False):
        logger.info("Saving result")
        self.copy_from_tmp(Path(self.tmp_dirs.distance_map.name), self.dst_path, archive)

    @logging_time
    def __make_skeleton(self, src_dir: Path):
        logger.info("Making skeleton")
        np_vol = self.load_imgs(sorted(src_dir.glob("*")))
        return morphology.skeletonize_3d(np_vol > 0)

    @logging_time
    def __make_distance_map(self, skeleton: np.ndarray, original_dir: Path, mm_resolution: float):
        logger.info("Making distance map")

        original_volume = self.load_imgs(sorted(original_dir.glob("*")))

        if self.using_gpu_ids is not None:
            import cupy as cp
            from cupyx.scipy.ndimage import binary_dilation

            mask = binary_dilation(cp.asarray(skeleton) > 0, iterations=5, brute_force=True).get()
        else:
            from scipy.ndimage import binary_dilation

            mask = binary_dilation(skeleton > 0, iterations=5, brute_force=True)

        mask = (original_volume * mask) > 8
        del original_volume

        volume_separator = VolumeSeparator()
        separated_vols = volume_separator.separate(mask.astype(np.uint8))
        del mask

        if self.using_gpu_ids is not None:
            from cucim.core.operations.morphology import distance_transform_edt

            with MultiThreadingGPU(self.using_gpu_ids) as mt:
                separated_vols = mt.run([ForEach(separated_vols)], distance_transform_edt)
        else:
            from scipy.ndimage import distance_transform_edt

            with MultiThreading() as mt:
                separated_vols = mt.run([ForEach(separated_vols)], distance_transform_edt)

        separated_vols = np.asarray(np.asarray(separated_vols) * 10, dtype=np.uint8)

        distance_map = volume_separator.assemble(separated_vols)
        distance_map[skeleton == 0] = 0

        return distance_map

    def __proceed_filtering(self, src_dir: Path, dim: str):
        np_vol = self.load_imgs(sorted(src_dir.glob("*"))).astype(np.int16)
        np_vol = self.transpose_volume(np_vol, dim)

        np_vol = self.apply_ring_filter(
            np_ary=np_vol,
            radius=self.kernel_radius,
            kernel_weight=self.kernel_weight,
        )
        np_vol = self.filter_by_size(np_vol, thr=self.size_filter_threshold)
        np_vol = self.transpose_volume(np_vol, dim)
        return np_vol > 0

    @logging_time
    def transpose_volume(self, np_vol, dim: str):
        def transpose(np_vol: np.ndarray, axes: tuple):
            if self.using_gpu_ids is not None:
                import cupy as cp

                return cp.transpose(cp.asarray(np_vol), axes=axes).get()
            else:
                return np.transpose(np_vol, axes=axes)

        dims_map = {"X": (0, 1, 2), "Y": (1, 0, 2), "Z": (2, 1, 0)}
        return transpose(np_vol, dims_map[dim])

    @logging_time
    def apply_ring_filter(self, np_ary: np.ndarray, radius: int, kernel_weight: float):
        kernel = self.get_ring_kernel(radius=radius, kernel_weight=kernel_weight)

        if self.using_gpu_ids is not None:
            from cupyx.scipy.ndimage import convolve

            with MultiThreadingGPU(self.using_gpu_ids) as mt:
                np_ary = mt.run([ForEach(np_ary), kernel], convolve)
        else:
            from scipy.ndimage import convolve

            with MultiThreading() as mt:
                np_ary = mt.run([ForEach(np_ary), kernel], convolve)

        np_ary = np.asarray(np_ary)
        np_ary[np_ary < 0] = 0

        return np_ary

    @logging_time
    def filter_by_size(self, np_ary: np.ndarray, thr: int):
        if self.using_gpu_ids is not None:
            import cupy as cp
            from cupyx.scipy.ndimage import label

            label_image = label(cp.asarray(np_ary) > 0)[0].get()
        else:
            from scipy.ndimage import label

            label_image = label(np_ary > 0)[0]

        areas = cp.bincount(cp.asarray(label_image[label_image > 0])).get()
        all_labels = np.where(areas > 0)[0]
        all_areas = areas[1:]
        area_labels = all_labels * (all_areas > thr)

        if self.using_gpu_ids is not None:
            target_index = np.where(areas <= thr)[0][1:]
            with MultiThreadingGPU(self.using_gpu_ids) as mt:
                label_mask = np.asarray(mt.run([ForEach(label_image), target_index], cp.isin))

            label_image[label_mask] = 0
            return np_ary * (label_image != 0)
        else:
            target_labels = util.map_array(
                label_image,
                all_labels,
                area_labels,
            )
            return np_ary * (target_labels != 0)

    def get_ring_kernel(self, radius: int, kernel_weight: float):
        assert radius > 1
        assert 0.0 <= kernel_weight <= 1.0

        disk_img1 = morphology.disk(radius)
        disk_img2 = np.pad(morphology.disk(radius - 1), 1)

        ring_kernel = np.bitwise_xor(disk_img1, disk_img2) * (-1)
        ring_kernel = ring_kernel.astype(np.float32)
        ring_kernel[radius, radius] = np.sum(ring_kernel < 0) * kernel_weight

        return ring_kernel

    @logging_time
    def merge_and_save_vols(self, src_dirs: list[Path], dst: Path):
        for f in sorted(src_dirs[0].glob("*")):
            imgs = []
            for i in range(3):
                imgs.append(io.imread(Path(src_dirs[i], f.name)).astype(np.uint8))

            merged_vol = np.stack(imgs).max(axis=0).astype(np.uint8)
            io.imsave(Path(dst, f.name), merged_vol)

    @logging_time
    def load_imgs(self, files: list[Path]):
        return np.asarray([io.imread(f).astype(np.uint8) for f in files])

    @logging_time
    def save_imgs(self, imgs, dst_dir: Path, extension=".png"):
        for i, img in enumerate(imgs):
            io.imsave(Path(dst_dir, f"img{i:04}{extension}"), img)

    def copy_from_tmp(self, tmp_dir: Path, dst_dir: Path, archive: bool):
        if archive:
            zip_path = tmp_dir.with_suffix(".zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
                for p in sorted(tmp_dir.glob("*")):
                    zf.write(p, p.name)
            shutil.copyfile(zip_path, dst_dir.with_suffix(".zip"))

        else:
            dst_dir.mkdir(parents=True, exist_ok=True)
            for p in sorted(tmp_dir.glob("*")):
                shutil.copyfile(p, Path(dst_dir, p.name))

        def __del__(self):
            pass
