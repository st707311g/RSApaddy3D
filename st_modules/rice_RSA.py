from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from .rsapaddy3d import covert_distance_transformation_value_to_root_diameter


@dataclass(frozen=True)
class Edge:
    path: list[tuple[int]]
    radius: list[int]

    def __post_init__(self):
        assert self.radius is None or len(self.path) == len(self.radius)

    @cached_property
    def z_list(self):
        return [it[0] for it in self.path]

    @cached_property
    def y_list(self):
        return [it[1] for it in self.path]

    @cached_property
    def x_list(self):
        return [it[2] for it in self.path]

    @cached_property
    def length(self):
        return np.linalg.norm(np.diff(self.path, axis=0), axis=1).sum()

    @cached_property
    def distance(self):
        return np.linalg.norm(np.asarray(self.path[0]) - np.asarray(self.path[-1]))

    @cached_property
    def average_radius(self):
        return np.mean(covert_distance_transformation_value_to_root_diameter(np.asarray(self.radius)))


@dataclass()
class RiceRSA:
    edges: list[Edge] = None
    shape: tuple[int] = None
    mm_resolution: float = None
    origin: tuple[int] = None

    class encoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Edge):
                return {"path": obj.path, "radius": obj.radius}
            elif isinstance(obj, np.int16):
                return int(obj)
            else:
                return super().default(obj)

    class decoder(json.JSONDecoder):
        def __init__(self, *args, **kwargs):
            json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

        def object_hook(self, o):
            if "path" in o and "radius" in o:
                return Edge(o["path"], radius=o["radius"])
            else:
                return o

    def save(self, path: str | Path):
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, cls=self.encoder)

    def load(self, path: str | Path):
        path = Path(path)
        with open(path, "r") as f:
            j = json.load(f, cls=self.decoder)

        for key in self.__dict__.keys():
            object.__setattr__(self, key, j[key])

        return self

    def filter_by_distance(self, mm_distance_threshold: float):
        # self.logger.info("filtering by distance")
        oks = []
        for it in self.edges:
            bl = np.asarray(it.path) - np.asarray(self.origin)
            mm_distance_ary = np.linalg.norm(bl, axis=1) * self.mm_resolution
            try:
                index = list((mm_distance_ary <= mm_distance_threshold).tolist()).index(False)
                if index == 0:
                    continue

                e = Edge(it.path[index:], radius=it.radius[index:])
                oks.append(e)

            except ValueError:
                oks.append(it)

        self.edges = oks
        return self
