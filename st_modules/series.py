from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Series(object):
    path: str | Path

    def __post_init__(self):
        self.path = Path(self.path)
        self._id_set: set[int] = set()
        self._name_map: dict[str, Path] = {}

        for p in sorted(self.path.glob("*")):
            self.add_item(p)

    def add_item(self, item_path: str | Path):
        item_path = Path(item_path)
        suffix = "".join(item_path.suffixes)
        name = item_path.name[: -len(suffix)]

        matched = re.findall("^([0-9]{2})_(.*)", name)
        if len(matched) != 1:
            return

        id, item_name = matched[0]

        self._name_map.update({item_name: item_path})
        self._id_set.add(int(id))

    @property
    def next_id(self):
        next_id = 0
        if len(self._id_set) != 0:
            while next_id in self._id_set:
                next_id += 1

        return next_id

    def create_and_get_new_path(self, key: str):
        next_path = Path(self.path, f"{self.next_id:02}_{key}")
        self.add_item(next_path)
        return next_path

    def __getitem__(self, key: str):
        return self._name_map.get(key)

    def does_contain(self, key: str):
        return key in self._name_map


def walk_to_find_series_directory(
    root_dir: str | Path,
    series_include: list[str] = ["ct"],
    series_exclude: list[str] = None,
    src_dir: str | Path = None,
):
    root_dir = Path(root_dir)
    src_dir = src_dir or root_dir

    series = Series(src_dir)

    if series_exclude is not None:
        for series_ in series_exclude:
            if series.does_contain(series_):
                yield series, True
                return

    is_target = True
    for series_ in series_include:
        is_target &= series.does_contain(series_)

    if is_target:
        yield series, False
        return

    for d in sorted(Path(src_dir).glob("*/")):
        if d.is_dir():
            yield from walk_to_find_series_directory(root_dir, series_include, series_exclude, d)
