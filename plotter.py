import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def plot_cdf(data: pd.DataFrame):
    plt.figure(figsize=(8, 4))
    for column in data.columns:
        sorted_data = np.sort(data[column])
        y = np.arange(len(sorted_data)) / len(sorted_data)
        plt.plot(sorted_data, y, label=column, linewidth=2)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.xlabel("Value")
    plt.ylabel("CDF")
    plt.tight_layout()


def plot_box(data: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    data.boxplot()
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Value")
    plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(description="Plot CDF and Boxplot from a CSV file.")
    parser.add_argument("-d", "--data", type=str, required=True, help="Path to the CSV data file")
    parser.add_argument("-c", "--cdf", type=str, required=True, help="Path to save the CDF plot")
    parser.add_argument("-b", "--box", type=str, required=True, help="Path to save the boxplot")
    args = parser.parse_args()

    data = pd.read_csv(args.data)

    plot_cdf(data)
    Path(args.cdf).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.cdf)
    plt.close()

    plot_box(data)
    Path(args.box).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.box)
    plt.close()


if __name__ == "__main__":
    main()