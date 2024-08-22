from __future__ import annotations

import argparse
import heapq
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from itertools import combinations
from pathlib import Path
from queue import Queue

import coloredlogs
import numpy as np
import sknw

from st_modules.config import logger
from st_modules.multi_threading import ForEach, MultiThreading
from st_modules.rsa_dataset import RSA_Dataset, walk_to_find_rsa_dataset
from st_modules.volume import SliceLoader


def _calculate_angle(a, b, c):
    vec_a = np.asarray(a) - np.asarray(b)
    vec_c = np.asarray(c) - np.asarray(b)

    assert vec_a.ndim == vec_c.ndim
    assert vec_a.ndim == 1 or vec_a.ndim == 2

    ndim = vec_a.ndim

    length_vec_a = np.linalg.norm(vec_a, axis=ndim - 1)
    length_vec_c = np.linalg.norm(vec_c, axis=ndim - 1)

    if ndim == 1:
        inner_product = np.inner(vec_a, vec_c)
    else:
        inner_product = [np.inner(a, b) for a, b in zip(vec_a, vec_c)]

    cos = np.asarray(inner_product) / (length_vec_a * length_vec_c)
    rad = np.arccos(np.clip(cos, -1.0, 1.0))

    degree = np.rad2deg(rad)
    return rad, degree


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
    def average_diameter(self):
        return np.mean(self.radius) * 2


def _get_group_list(edges: dict[tuple, dict[tuple, Edge]]):
    done = set()
    edge_group = []

    for key in edges.keys():
        if key in done:
            continue

        que = Queue()
        que.put(key)

        group_list = []

        while not que.empty():
            from_key = que.get()
            if from_key in done:
                continue
            done.add(from_key)
            group_list.append(from_key)
            for to_key, edge in edges[from_key].items():
                que.put(to_key)

        edge_group.append(group_list)

    return edge_group


def _get_length_and_hist(edges: dict[tuple, dict[tuple, Edge]], start: tuple[int], goal: tuple[int]) -> None | tuple[float, list[tuple[int]]]:
    done = set()

    que = []
    heapq.heappush(que, (0, start, [start]))

    ret = None, None
    while len(que) != 0:
        from_length, from_key, from_hist = heapq.heappop(que)

        if from_key in done:
            continue

        done.add(from_key)

        if len(from_hist) >= 3:
            f = False
            for i in range(1, len(from_hist) - 1):
                rad, degree = _calculate_angle(from_hist[0], from_hist[i], from_hist[-1])
                f |= degree < 135
            if f:
                continue

        if from_key == goal:
            path = [from_hist[0]]
            for to_key in from_hist[1:]:
                path.extend(edges[path[-1]][to_key].path[1:])

            angle = 180
            for i in range(1, len(path) - 1):
                a = np.asarray(path[0])
                b = np.asarray(path[i])
                c = np.asarray(path[-1])
                angle = min(angle, _calculate_angle(a, b, c)[1])
            if 120 <= angle <= 180:
                ret = (from_length, path)
                break
            else:
                continue

        for to_key, edge in edges[from_key].items():
            heapq.heappush(que, (from_length + edge.length, to_key, from_hist + [to_key]))

    return ret


def _get_longest_path(edges: dict[tuple, dict[tuple, Edge]], grouped: list[tuple[int]]):
    counts = defaultdict(list)

    for key in grouped:
        counts[len(edges[key])].append(key)

    if len(counts[1]) < 2:
        return None

    current_length = -1
    current_path = []
    for it in combinations(counts[1], 2):
        length, path = _get_length_and_hist(edges, it[0], it[1])
        if length is not None and current_length < length:
            current_length = length
            current_path = path

    if current_length == -1:
        return None

    return current_path


class RSA_Vect(object):
    def __init__(
        self,
        skeleton_vol: np.ndarray,
        mm_resolution: float,
        logger: logging.Logger = None,
    ) -> None:
        self.logger = logger or logging.getLogger("RSApaddyVis3D")

        self.logger.info("making RSA graph")
        graph = sknw.build_sknw(skeleton_vol)
        edges = defaultdict(dict)
        for s, e in graph.edges():
            ps = [tuple(it) for it in graph[s][e]["pts"]]

            if len(ps) >= 2 and tuple(ps[0]) != tuple(ps[-1]):
                edges[tuple(ps[0])].update({tuple(ps[-1]): Edge(path=ps, radius=None)})
                edges[tuple(ps[-1])].update({tuple(ps[0]): Edge(path=list(reversed(ps)), radius=None)})

        grouped_list = _get_group_list(edges)

        straight_pathes: list[Edge] = []
        for it in grouped_list:
            path = _get_longest_path(edges, it)
            if path:
                radius = [skeleton_vol[p[0], p[1], p[2]] / 10 for p in path]
                straight_pathes.append(Edge(path, radius))

        self.rice_RSA = RiceRSA(straight_pathes, skeleton_vol.shape, mm_resolution)
        self.logger.debug(f"the number of pathes: {len(straight_pathes)}")

    def filter_by_length(self, mm_length_threshold: float):
        self.logger.info("filtering by length")
        self.rice_RSA.edges = list(
            filter(
                lambda x: x.length >= mm_length_threshold / self.rice_RSA.mm_resolution,
                self.rice_RSA.edges,
            )
        )
        self.logger.debug(f"the number of pathes: {len(self.rice_RSA.edges)}")
        return self


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


@dataclass
class Args:
    src: Path
    mm_length_threshold: int
    debug: bool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--mm_length_threshold", type=int, default=10)
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

    for dataset, be_skipped in walk_to_find_rsa_dataset(args.src, ["rsapaddy3d", "mask"], ["root_paths"]):
        relative_path = dataset._config.path.relative_to(args.src)
        logger.info(f"target: {relative_path=}")

        if be_skipped:
            logger.info("skipped: already processed.")
            continue

        proceed(dataset, relative_path, args)


def proceed(dataset: RSA_Dataset, relative_path: Path, args: Args):
    rsapaddy3d_path = dataset.get_series_path("rsapaddy3d")
    logger.debug(f"{rsapaddy3d_path=}")

    with SliceLoader(rsapaddy3d_path, expand=True) as loader:
        with MultiThreading(task_name="volume loading") as mt:

            def load_img(i: int):
                return loader.load(i)

            paddy_vol = np.array(mt.run([ForEach(range(loader.img_count))], load_img, callback=mt.print_callback))

    mask_path = dataset.get_series_path("mask")
    logger.debug(f"{mask_path=}")

    with SliceLoader(mask_path, expand=True) as loader:
        with MultiThreading(task_name="volume loading") as mt:

            def load_img(i: int):
                return loader.load(i)

            mask_vol = np.array(mt.run([ForEach(range(loader.img_count))], load_img, callback=mt.print_callback))

    paddy_vol[mask_vol == 0] = 0
    volume_info = dataset.load_volume_info()
    logger.debug(f"{volume_info=}")

    mm_resolution = volume_info["mm_resolution"]
    mm_length_threshold = args.mm_length_threshold

    rice_rsa = (RSA_Vect(skeleton_vol=paddy_vol, mm_resolution=mm_resolution, logger=logger).filter_by_length(mm_length_threshold)).rice_RSA

    dst_path = dataset.create_new_series("root_paths", ".json")
    logger.debug(f"{dst_path=}")
    rice_rsa.save(dst_path)

    dataset.update_log({dst_path.stem: {"mm_length_threshold": mm_length_threshold}})


if __name__ == "__main__":
    main()
