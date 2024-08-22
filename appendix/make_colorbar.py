from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


def main():
    fig, cbar = plt.subplots(figsize=(0.3, 2))
    cmap = plt.get_cmap("viridis")
    mpl.colorbar.Colorbar(
        ax=cbar,
        mappable=mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=0, vmax=2.1),
            cmap=cmap,
        ),
        orientation="vertical",
    ).set_label("diameter [mm]")

    plt.savefig("colorbar.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
