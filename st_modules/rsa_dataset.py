from __future__ import annotations

import datetime
import json
import re
from dataclasses import dataclass
from pathlib import Path


class encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.date):
            return obj.strftime("%Y_%m_%d")
        return json.JSONEncoder.default(self, obj)


@dataclass
class _RSA_DatasetConfig:
    path: Path

    @property
    def log_path(self):
        return Path(self.path, ".log.json")

    @property
    def volume_info_path(self):
        return Path(self.path, ".volume_info.json")


class RSA_Dataset(object):
    def __init__(self, path: str | Path):
        self._config = _RSA_DatasetConfig(path=Path(path))
        self._id_set: set[int] = set()
        self._name_map: dict[str, Path] = {}

        for p in sorted(self._config.path.glob("*")):
            if p.is_dir():
                continue

            suffix = "".join(p.suffixes)
            name = p.name[: -len(suffix)]

            matched = re.findall("^([0-9]{2})_(.*)", name)
            if len(matched) != 1:
                continue

            id, series_name = matched[0]

            self._name_map.update({series_name: p})
            self._id_set.add(int(id))

    @property
    def _next_id(self):
        next_id = 0
        if len(self._id_set) != 0:
            while next_id in self._id_set:
                next_id += 1

        return next_id

    def create_new_series(self, series_name: str, suffix: str = ""):
        id_ = self._next_id
        next_path = Path(self._config.path, f"{id_:02}_{series_name}").with_suffix(suffix)

        self._name_map.update({series_name: next_path})
        self._id_set.add(int(id_))
        return next_path

    def get_series_path(self, series_name: str):
        return self._name_map.get(series_name)

    def does_contain(self, series_name: str):
        return series_name in self._name_map

    def load_volume_info(self):
        if self._config.volume_info_path is None:
            return {}

        return json.loads(self._config.volume_info_path.read_text())

    def load_log(self):
        if self._config.log_path is None:
            return {}

        return json.loads(self._config.log_path.read_text())

    def update_volume_info(self, additional_info: dict):
        if self._config.volume_info_path is None:
            return

        vol_info = self.load_volume_info()
        vol_info.update(additional_info)
        self._config.volume_info_path.write_text(json.dumps(vol_info, indent=2, cls=encoder))

    def update_log(self, additional_info: dict):
        if self._config.log_path is None:
            return

        log = self.load_log()
        log.update(additional_info)
        self._config.log_path.write_text(json.dumps(log, indent=2, cls=encoder))


def walk_to_find_rsa_dataset(
    root_dir: str | Path,
    series_include: list[str] = ["ct"],
    series_exclude: list[str] = None,
    src_dir: str | Path = None,
):
    root_dir = Path(root_dir)
    src_dir = src_dir or root_dir

    dataset = RSA_Dataset(src_dir)

    if series_exclude is not None:
        for series_ in series_exclude:
            if dataset.does_contain(series_):
                yield dataset, True
                return

    is_target = True
    for series_ in series_include:
        is_target &= dataset.does_contain(series_)

    if is_target:
        yield dataset, False
        return

    for d in sorted(Path(src_dir).glob("*/")):
        if d.is_dir():
            yield from walk_to_find_rsa_dataset(root_dir, series_include, series_exclude, d)
