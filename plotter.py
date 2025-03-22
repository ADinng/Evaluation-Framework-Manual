import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path


def plot_cdf(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    for desc, col in df.items():
        sorted_data = np.sort(col)
        y = np.arange(len(sorted_data)) / len(sorted_data)
        ax.plot(sorted_data, y, label=desc, lw=2)
    ax.legend(loc="lower right")
    ax.grid()
    ax.set_xlabel("Value")
    ax.set_ylabel("CDF")
    fig.tight_layout()
    return fig


def plot_box(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    df.boxplot(ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel("Value")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CDF and Boxplot from .csv file.")
    parser.add_argument("-d", "--data", type=str, help="Path to the .csv data file", required=True)
    parser.add_argument("-c", "--cdf", type=str, help="Path to the output CDF plot", required=True)
    parser.add_argument("-b", "--box", type=str, help="Path to the output boxplot", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    cdf_fig = plot_cdf(df)
    box_fig = plot_box(df)

    Path(args.cdf).parent.mkdir(parents=True, exist_ok=True)
    Path(args.box).parent.mkdir(parents=True, exist_ok=True)
    cdf_fig.savefig(args.cdf)
    box_fig.savefig(args.box)
    plt.close()


if __name__ == "__main__":
    main()