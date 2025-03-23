import argparse
import numpy as np
import pandas as pd
from pathlib import Path


class RNG:
    def __init__(self, seed: int, num_samples: int):
        self.rng = np.random.default_rng(seed)
        self.num_samples = num_samples

    def gamma(self, shape: float, scale: float) -> np.ndarray:
        return self.rng.gamma(shape, scale, self.num_samples)

    def exponential(self, scale: float) -> np.ndarray:
        return self.rng.exponential(scale, self.num_samples)

    def gaussian(self, mean: float, std: float) -> np.ndarray:
        return self.rng.normal(mean, std, self.num_samples)

    def dagum(self, a: float, b: float, p: float) -> np.ndarray:
        if not (a > 0 and b > 0 and p > 0):
            raise ValueError(" a, b, and p must be positive.")
        u = self.rng.uniform(size=self.num_samples)
        return b * (u ** (-1 / p) - 1) ** (-1 / a)

    def geometric(self, p: float) -> np.ndarray:
        return self.rng.geometric(p, self.num_samples)

    def poisson(self, lam: float) -> np.ndarray:
        return self.rng.poisson(lam, self.num_samples)

    def skellam(self, mu1: float, mu2: float) -> np.ndarray:
        if not (mu1 >= 0 and mu2 >= 0):
            raise ValueError(" mu1 and mu2 must be non-negative.")
        if mu2 == 0:
            return self.rng.poisson(mu1, self.num_samples)
        u = self.rng.uniform(size=self.num_samples)
        return self._skellam_ppf(u, mu1, mu2)

    def _skellam_cdf(self, mu1: float, mu2: float) -> tuple[int, np.ndarray]:
        mean = mu1 - mu2
        std = np.sqrt(mu1 + mu2)
        lower = int(np.floor(mean - 10 * std))
        upper = int(np.ceil(mean + 10 * std))
        k = np.arange(lower, upper + 1, dtype=np.float64)
        pmf = np.exp(-mu1 - mu2) * (mu1 / mu2) ** (k / 2) * np.i0(2 * np.sqrt(mu1 * mu2))
        return lower, np.cumsum(pmf)

    def _skellam_ppf(self, q: np.ndarray, mu1: float, mu2: float) -> np.ndarray:
        lower, cdf = self._skellam_cdf(mu1, mu2)
        return np.searchsorted(cdf, q).astype(np.int64) + lower


def main():
    parser = argparse.ArgumentParser(description="Generate random sequences from a .cfg file.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the .cfg file")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output .csv file")
    args = parser.parse_args()

    # Read the configuration file
    with open(args.config) as file:
        lines = [line.strip() for line in file if line.strip()]
        seed = int(lines[0])
        num_samples = int(lines[1])
        rng = RNG(seed, num_samples)

        dist_map = {
            "gam": (rng.gamma, "Gamma"),
            "exp": (rng.exponential, "Exponential"),
            "gau": (rng.gaussian, "Gaussian"),
            "dag": (rng.dagum, "Dagum"),
            "geo": (rng.geometric, "Geometric"),
            "poi": (rng.poisson, "Poisson"),
            "ske": (rng.skellam, "Skellam"),
        }

        # Generate random numbers 
        data = []
        headers = []
        for line in lines[2:]:
            parts = line.split()
            dist = parts[0].lower()
            params = list(map(float, parts[1:]))
            if dist not in dist_map:
                raise ValueError(f"Unsupported distribution: {dist}")
            func, name = dist_map[dist]
            data.append(func(*params))
            headers.append(f"{name}({','.join(map(str, params))})")

    # Save data to a CSV file
    df = pd.DataFrame(data).T
    df.columns = headers
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()