from __future__ import annotations

from functools import lru_cache

import numpy as np
import polars as pl
from skimage.morphology import ball
from tqdm import tqdm

from .rice_RSA import Edge, RiceRSA
from .rsapaddy3d import covert_distance_transformation_value_to_root_diameter


@lru_cache(maxsize=None)
def _get_ofset_list(radius: int):
    b = ball(radius=radius)
    return np.array(np.where(b == 1)).transpose() - radius


def _draw(np_vol: np.ndarray, edge: Edge, mm_resolution: float):
    zyx_lists = []
    radius = edge.radius.copy()
    radius = np.convolve(radius, np.ones(5) / 5, mode="same")
    size_list = []
    for pos, rad in zip(edge.path, radius):
        zyx_list = pos + _get_ofset_list(int(np.ceil(rad)) - 1)
        zyx_lists.append(zyx_list)
        size_list.extend([np.clip(int(covert_distance_transformation_value_to_root_diameter(rad) * mm_resolution * 100), 0, 255)] * len(zyx_list))

    np_ary = np.concatenate(zyx_lists, axis=0)

    df = pl.DataFrame(
        {
            "z": np_ary[..., 0],
            "y": np_ary[..., 1],
            "x": np_ary[..., 2],
            "r": np.asarray(size_list),
        }
    )

    df = (
        df.unique(subset=["z", "y", "x"])
        .filter(pl.col("z") >= 0)
        .filter(pl.col("y") >= 0)
        .filter(pl.col("x") >= 0)
        .filter(pl.col("z") < np_vol.shape[0])
        .filter(pl.col("y") < np_vol.shape[1])
        .filter(pl.col("x") < np_vol.shape[2])
    )

    res = df.to_numpy().transpose()
    np_vol[res[0], res[1], res[2]] = res[3]


def draw_and_get_np_vol(rice_rsa: RiceRSA):
    np_vol = np.zeros(rice_rsa.shape, dtype=np.uint8)

    for e in tqdm(rice_rsa.edges):
        _draw(np_vol, e, rice_rsa.mm_resolution)

    return np_vol
