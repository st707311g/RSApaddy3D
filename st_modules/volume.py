from __future__ import annotations

import itertools
import zipfile
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import imageio
import numpy as np

from .multi_threading import ForEach, MultiThreading


class SliceLoader:
    minimum_file_number: int = 64
    extensions: tuple = (
        ".cb",
        ".png",
        ".tif",
        ".tiff",
        ".jpg",
        ".jpeg",
    )

    def __init__(self, volume_path: str | Path, expand: bool = False):
        self._volume_path = Path(volume_path)
        self._img_files = []
        self._target_dir: Path = None

        if self.is_zip():
            if expand:
                self._tmp_dir = TemporaryDirectory()
                with zipfile.ZipFile(self._volume_path, "r") as zf:
                    zf.extractall(self._tmp_dir.name)
                self._target_dir = Path(self._tmp_dir.name)
        else:
            self._target_dir = self._volume_path

        if self.is_zip() and not expand:
            self._zip_file = zipfile.ZipFile(self._volume_path, "r")
            for info in self._zip_file.infolist():
                if len(Path(info.filename).parents) == 1:
                    self._img_files.append(info.filename)
        else:
            for f in self._target_dir.glob("*"):
                if f.is_file():
                    self._img_files.append(f)

        self._img_files.sort()

        ext_count = []
        for ext in self.extensions:
            ext_count.append(len([f for f in self._img_files if str(f).lower().endswith(ext)]))

        target_extension = self.extensions[ext_count.index(max(ext_count))]
        self._img_files = [f for f in self._img_files if str(f).lower().endswith(target_extension)]

    def is_zip(self):
        return self._volume_path.name.lower().endswith(".zip")

    @property
    def img_count(self):
        return len(self._img_files)

    def is_valid(self):
        return self.img_count >= self.minimum_file_number

    @lru_cache(maxsize=32)
    def load(self, index: int):
        if self._target_dir is None:
            return imageio.imread_v2(self._zip_file.open(self._img_files[index]))
        else:
            return imageio.imread_v2(self._img_files[index])

    def __del__(self):
        if self._target_dir is None:
            self._zip_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def load_all(self):
        with SliceLoader(self._volume_path, expand=True) as loader:
            with MultiThreading(task_name="volume loading") as mt:

                def load_img(i: int):
                    return loader.load(i)

                return np.array(mt.run([ForEach(range(loader.img_count))], load_img, callback=mt.print_callback))


@dataclass
class VolumeData(object):
    path: str | Path
    minimum_file_number: int = 64
    extensions: tuple = (
        ".cb",
        ".png",
        ".tif",
        ".tiff",
        ".jpg",
        ".jpeg",
    )

    def __post_init__(self):
        self.path = Path(self.path)

    @cached_property
    def image_files(self):
        files = list(sorted(self.path.glob("*")))

        ext_count = []
        for ext in self.extensions:
            ext_count.append(len([f for f in files if str(f).lower().endswith(ext)]))

        target_extension = self.extensions[ext_count.index(max(ext_count))]
        image_files = sorted([f for f in files if str(f).lower().endswith(target_extension)])
        return image_files

    @property
    def image_file_number(self):
        return len(self.image_files)

    def is_valid(self):
        return self.image_file_number >= self.minimum_file_number


class VolumeSeparator:
    def __init__(self, block_size: int = 64, overlap: int = 8) -> None:
        self.block_size = block_size
        self.overlap = overlap
        self.__shape = (128, 128, 128)

    def separate(self, np_vol: np.ndarray) -> list[np.ndarray]:
        self.__shape = np_vol.shape

        adjusted_block_size = self.block_size - self.overlap * 2
        pad_size = []

        absize = adjusted_block_size
        for s in self.__shape:
            pad_size.append((0, absize * ((s + absize - 1) // absize) - s))

        np_vol = np.pad(np_vol, pad_size, mode="reflect")
        np_vol = np.pad(np_vol, self.overlap, mode="reflect")

        separated_vols = []
        range_list = [range((s + absize - 1) // absize) for s in self.__shape]
        for i, j, k in itertools.product(*range_list):
            sub = np_vol[
                i * absize : (i + 1) * absize + self.overlap * 2,
                j * absize : (j + 1) * absize + self.overlap * 2,
                k * absize : (k + 1) * absize + self.overlap * 2,
            ]
            separated_vols.append(sub)

        return separated_vols

    def assemble(self, separated_vols: list[np.ndarray]):
        adjusted_block_size = self.block_size - self.overlap * 2
        pad_size = []

        absize = adjusted_block_size
        for s in self.__shape:
            pad_size.append((0, absize * ((s + absize - 1) // absize) - s))

        adjusted_shape = []
        for s, p in zip(self.__shape, pad_size):
            adjusted_shape.append(s + p[1] + self.overlap * 2)

        res = np.zeros(adjusted_shape, dtype=separated_vols[0].dtype)

        index = 0
        range_list = [range((s + absize - 1) // absize) for s in self.__shape]
        for i, j, k in itertools.product(*range_list):
            res[
                i * absize : (i + 1) * absize + self.overlap * 2,
                j * absize : (j + 1) * absize + self.overlap * 2,
                k * absize : (k + 1) * absize + self.overlap * 2,
            ] = separated_vols[index]
            index += 1

        return res[
            self.overlap : self.__shape[0] + self.overlap,
            self.overlap : self.__shape[1] + self.overlap,
            self.overlap : self.__shape[2] + self.overlap,
        ]
