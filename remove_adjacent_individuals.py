from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import coloredlogs
import numpy as np
from scipy import optimize

from st_modules.config import logger
from st_modules.multi_threading import ForEach, MultiThreading
from st_modules.rice_RSA import Edge, RiceRSA
from st_modules.rsa_dataset import RSA_Dataset, walk_to_find_rsa_dataset
from st_modules.volume import SliceLoader


def _calculate_angle_score(pos: tuple[int], straight_pathes: list[Edge]):
    a = np.asarray(pos)
    b = []
    c = []

    for it in straight_pathes:
        b.append(it.path[0])
        c.append(it.path[-1])

        if np.linalg.norm(np.asarray(b[-1] - a)) > np.linalg.norm(np.asarray(c[-1] - a)):
            b[-1], c[-1] = c[-1], b[-1]

    b = np.asarray(b)
    c = np.asarray(c)

    rad, degree = _calculate_angle(a, b, c)
    return np.sum(abs(degree - 180))


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


def filter_with_angle(rice_rsa: RiceRSA, degree_angle_threshold: int):
    logger.info(f"filtering with angle: {degree_angle_threshold=}")
    origin = rice_rsa.origin

    ok_list = []
    for it in rice_rsa.edges:
        z = [i[0] - origin[0] for i in it.path]
        dy = [y - origin[1] for y in it.y_list]
        dx = [x - origin[2] for x in it.x_list]
        r = [np.linalg.norm([y, x]) for y, x in zip(dy, dx)]

        if r[-1] < r[0]:
            z = list(reversed(z))
            dy = list(reversed(dy))
            dx = list(reversed(dx))
            r = list(reversed(r))

        _, deg1 = _calculate_angle((0, 0), (dy[0], dx[0]), (dy[-1], dx[-1]))
        _, deg2 = _calculate_angle((0, 0), (r[0], z[0]), (r[-1], z[-1]))

        if 180 - degree_angle_threshold <= deg1 <= 180:
            if 180 - degree_angle_threshold <= deg2 <= 180:
                ok_list.append(it)

    rice_rsa.edges = ok_list
    logger.debug(f"the number of pathes: {len(rice_rsa.edges)}")

    return rice_rsa


def find_origin(rice_rsa: RiceRSA):
    logger.info("finding origin")
    shape = rice_rsa.shape

    w_opt = optimize.minimize(
        fun=partial(_calculate_angle_score, straight_pathes=rice_rsa.edges),
        x0=(shape[0] // 2, shape[1] // 2, shape[2] // 2),
        method="L-BFGS-B",
        bounds=[
            (-100, shape[0] // 2),
            (shape[1] // 4, shape[1] // 4 * 3),
            (shape[2] // 4, shape[2] // 4 * 3),
        ],
    )
    rice_rsa.origin = tuple(w_opt.x)
    logger.debug(f"{rice_rsa.origin=}")
    return rice_rsa


def reorder_straight_pathes(rice_rsa: RiceRSA):
    logger.info("reorder straight pathes")
    origin = np.asarray(rice_rsa.origin)

    res = []
    for it in rice_rsa.edges:
        a = np.asarray(it.path[0])
        b = np.asarray(it.path[-1])

        if np.linalg.norm(a - origin) < np.linalg.norm(b - origin):
            path = it.path
            radius = it.radius
        else:
            path = list(reversed(it.path))
            radius = list(reversed(it.radius))

        res.append(Edge(path, radius))

    rice_rsa.edges = res
    return rice_rsa


def remove_items_above_origin(rice_rsa: RiceRSA):
    logger.info("removing items above origin")
    rice_rsa.edges = list(
        filter(
            lambda x: x.path[0][0] >= rice_rsa.origin[0] and x.path[-1][0] >= rice_rsa.origin[0],
            rice_rsa.edges,
        )
    )
    logger.debug(f"the number of pathes: {len(rice_rsa.edges)}")
    return rice_rsa


@dataclass
class Args:
    src: Path
    degree_angle_threshold: int
    debug: bool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--degree_angle_threshold", type=int, default=30)
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

    for dataset, be_skipped in walk_to_find_rsa_dataset(args.src, ["root_paths", "mask"], ["root_paths_filtered"]):
        relative_path = dataset._config.path.relative_to(args.src)
        logger.info(f"target: {relative_path=}")

        if be_skipped:
            logger.info("skipped: already processed.")
            continue

        proceed(dataset, relative_path, args)


def proceed(dataset: RSA_Dataset, relative_path: Path, args: Args):
    mask_path = dataset.get_series_path("mask")
    logger.debug(f"{mask_path=}")

    with SliceLoader(mask_path, expand=True) as loader:
        with MultiThreading(task_name="volume loading") as mt:

            def load_img(i: int):
                return loader.load(i)

            mask_vol = np.array(mt.run([ForEach(range(loader.img_count))], load_img, callback=mt.print_callback))

    shape_range = []
    for i in [(1, 2), (0, 2), (0, 1)]:
        v = np.max(mask_vol, axis=i)
        shape_range.append((min(np.where(v != 0)[0]), max(np.where(v != 0)[0])))

    logger.debug(f"{shape_range=}")

    rice_rsa = RiceRSA()
    rice_rsa.load(dataset.get_series_path("root_paths"))

    # // tentative origin
    rice_rsa.origin = (
        shape_range[0][0],
        shape_range[1][0] // 2 + shape_range[1][1] // 2,
        shape_range[2][0] // 2 + shape_range[2][1] // 2,
    )

    logger.debug(f"tentative: {rice_rsa.origin=}")
    logger.debug(f"the number of pathes: {len(rice_rsa.edges)}")
    rice_rsa = filter_with_angle(rice_rsa, 60)
    rice_rsa = find_origin(rice_rsa)
    rice_rsa = reorder_straight_pathes(rice_rsa)
    rice_rsa = remove_items_above_origin(rice_rsa)
    rice_rsa = filter_with_angle(rice_rsa, args.degree_angle_threshold)

    dst_path = dataset.create_new_series("root_paths_filtered", ".json")
    logger.debug(f"{dst_path=}")
    rice_rsa.save(dst_path)

    dataset.update_log({dst_path.stem: {"degree_angle_threshold": args.degree_angle_threshold}})


if __name__ == "__main__":
    main()
