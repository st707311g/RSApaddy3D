from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import coloredlogs
import polars as pl

from st_modules.config import logger
from st_modules.rsa_dataset import walk_to_find_rsa_dataset


@dataclass
class Args:
    src: Path
    series_src: str
    debug: bool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--series_src", type=str, default="rsa_params")
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

    df_list = []

    for dataset, be_skipped in walk_to_find_rsa_dataset(args.src, [args.series_src]):
        relative_path = dataset._config.path.relative_to(args.src)
        logger.info(f"target: {relative_path=}")

        if be_skipped:
            logger.info("skipped: already processed.")
            continue

        with open(dataset.get_series_path(args.series_src)) as f:
            res = json.load(f)

        res = {"file": str(relative_path), **res}

        df = pl.DataFrame(res)
        df_list.append(df)

    df_combined = pl.concat(df_list)
    print(df_combined)

    df_combined.write_csv(Path(args.src, "rsa_params_combined.csv"))


if __name__ == "__main__":
    main()
