from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import coloredlogs
import numpy as np
from scipy import interpolate

from st_modules.config import logger
from st_modules.rice_RSA import RiceRSA
from st_modules.rsa_dataset import RSA_Dataset, walk_to_find_rsa_dataset


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
class riceRSAparams:
    rice_rsa: RiceRSA

    @cached_property
    def total_root_length(self):
        total = 0.0
        for e in self.rice_rsa.edges:
            total += e.length

        return total

    @cached_property
    def degrees(self):
        res = []
        origin = self.rice_rsa.origin
        for it in self.rice_rsa.edges:
            z = [i[0] - origin[0] for i in it.path]
            dy = [y - origin[1] for y in it.y_list]
            dx = [x - origin[2] for x in it.x_list]
            r = [np.linalg.norm([y, x]) for y, x in zip(dy, dx)]

            if r[-1] < r[0]:
                z = list(reversed(z))
                r = list(reversed(r))

            _, deg = _calculate_angle((1, 0), (0, 0), (r[-1], z[-1]))
            res.append(deg)

        return res

    @cached_property
    def diameters(self):
        res = []
        for it in self.rice_rsa.edges:
            res.append(it.average_diameter * self.rice_rsa.mm_resolution)

        return res

    @cached_property
    def weighted_average_mm_root_diameter(self):
        root_diameter = 0.0
        for it in self.rice_rsa.edges:
            root_diameter += it.average_radius * self.rice_rsa.mm_resolution * it.length / self.total_root_length

        return root_diameter

    @cached_property
    def weighted_average_RGA(self):
        rga = 0.0

        for it, degree in zip(self.rice_rsa.edges, self.degrees):
            rga += degree * it.length / self.total_root_length

        return rga

    @cached_property
    def _RGA_quartile(self):
        len_deg = []

        for e, degree in zip(self.rice_rsa.edges, self.degrees):
            len_deg.append((e.length, degree))

        len_deg.sort(key=lambda x: x[1])

        len_deg = np.asarray(len_deg)

        len_ary = []
        for i, it in enumerate(len_deg[:, 0]):
            len_ary.append(it)
            if i != 0:
                len_ary[i] += len_ary[i - 1]

        len_ary = np.asarray(len_ary) / len_ary[-1]
        deg_ary = np.asarray([it for it in len_deg[:, 1]])

        interp_func = interpolate.interp1d(len_ary, deg_ary)

        return tuple([float(interp_func(i * 0.25)) for i in range(1, 4)])

    @cached_property
    def RGA25(self):
        return self._RGA_quartile[0]

    @cached_property
    def RGA50(self):
        return self._RGA_quartile[1]

    @cached_property
    def RGA75(self):
        return self._RGA_quartile[2]

    @cached_property
    def RGA_gap(self):
        return self._RGA_quartile[1] - (self._RGA_quartile[2] - self._RGA_quartile[0])

    @cached_property
    def RDI(self) -> float:
        zs_list = []
        for it in self.rice_rsa.edges:
            zs = (np.asarray(it.z_list).astype(np.float32) - self.rice_rsa.origin[0]).tolist()
            zs_list.extend(zs)

        return np.mean(zs_list) * self.rice_rsa.mm_resolution


@dataclass
class Args:
    src: Path
    series_src: str
    series_dst: str
    traits: list[str]
    mm_monolith_diameter: int
    debug: bool


def main():
    parser = argparse.ArgumentParser(description="RSA mask maker")
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--series_src", type=str, default="root_paths_filtered")
    parser.add_argument("--series_dst", type=str, default="rsa_params_filtered")
    parser.add_argument("--traits", type=str, choices=["RGA", "diameter", "length"], nargs="*", default=["RGA", "diameter", "length"])
    parser.add_argument("--mm_monolith_diameter", type=int, default=160)
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
    json_file_path = dataset.get_series_path(args.series_src)
    logger.debug(f"{json_file_path=}")

    rice_rsa = RiceRSA()
    rice_rsa.load(json_file_path)

    trait_dict = {"number_of_segments": len(rice_rsa.edges), "resolution": rice_rsa.mm_resolution}

    for trait_name in args.traits:
        res = None

        match trait_name:
            case "RGA":
                trait_dict.update({"RGA": riceRSAparams(rice_rsa).weighted_average_RGA})
                continue

            case "diameter":
                res = riceRSAparams(rice_rsa).weighted_average_mm_root_diameter
                trait_dict.update({"mm_diameter": res})
                continue

            case "length":
                res = riceRSAparams(rice_rsa).total_root_length * rice_rsa.mm_resolution
                trait_dict.update({"mm_length": res})
                continue

    logger.info(trait_dict)

    dst_path = dataset.create_new_series(args.series_dst, ".json")
    logger.debug(f"{dst_path=}")

    with open(dst_path, "w") as f:
        json.dump(trait_dict, f)


if __name__ == "__main__":
    main()
