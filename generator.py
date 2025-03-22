import argparse
import numpy as np
import pandas as pd
from pathlib import Path


class Rng:
    def __init__(self, seed: int, count: int) -> None:
        self._rng = np.random.default_rng(seed)
        self._count = count
        self._mapping = {
            "gam": (self.gamma, "Gamma"),
            "exp": (self.exponential, "Exponential"),
            "gau": (self.gaussian, "Gaussian"),
            "dag": (self.dagum, "Dagum"),
            "geo": (self.geometric, "Geometric"),
            "poi": (self.poisson, "Poisson"),
            "ske": (self.skellam, "Skellam"),
        }

    def gamma(self, shape: float, scale: float) -> np.ndarray:
        return self._rng.gamma(shape, scale, self._count)

    def exponential(self, scale: float) -> np.ndarray:
        return self._rng.exponential(scale, self._count)

    def gaussian(self, mean: float, std: float) -> np.ndarray:
        return self._rng.normal(mean, std, self._count)

    def dagum(self, a: float, b: float, p: float) -> np.ndarray:
        if not (a > 0 and b > 0 and p > 0):
            raise ValueError("a, b, and p must be positive")
        u = self._rng.uniform(size=self._count)
        return b * (u ** (-1 / p) - 1) ** (-1 / a)

    def geometric(self, p: float) -> np.ndarray:
        return self._rng.geometric(p, self._count)

    def poisson(self, lam: float) -> np.ndarray:
        return self._rng.poisson(lam, self._count)

    def skellam(self, mu1: float, mu2: float) -> np.ndarray:
        if not (mu1 >= 0 and mu2 >= 0):
            raise ValueError("mu1 and mu2 must be non-negative")
        if mu2 == 0:
            return self._rng.poisson(mu1, self._count)
        u = self._rng.uniform(size=self._count)
        return _skellam_ppf(u, mu1, mu2)

    def __call__(self, distribution: str, *args) -> tuple[str, np.ndarray]:
        f, name = self._mapping[distribution.lower()]
        return f"{name}({','.join(map(str, args))})", f(*map(float, args))


def _skellam_cdf(mu1: float, mu2: float) -> tuple[int, np.ndarray]:
    mean, std = mu1 - mu2, np.sqrt(mu1 + mu2)
    l, r = int(np.floor(mean - 10 * std)), int(np.ceil(mean + 10 * std))
    k = np.arange(l, r + 1, dtype=np.float64)
    pmf = np.exp(-mu1 - mu2) * (mu1 / mu2) ** (k / 2) * np.i0(2 * np.sqrt(mu1 * mu2))
    return l, np.cumsum(pmf)


def _skellam_ppf(q: np.ndarray, mu1: float, mu2: float) -> np.ndarray:
    l, cdf = _skellam_cdf(mu1, mu2)
    return np.searchsorted(cdf, q).astype(np.int64, copy=False) + l


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random sequences from .cfg file.")
    parser.add_argument("-c", "--config", type=str, help="Path to the .cfg file", required=True)
    parser.add_argument("-o", "--output", type=str, help="Path to the output .csv file", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        lines = list(filter(None, map(str.strip, f)))
        seed = int(lines[0])
        count = int(lines[1])
        rng = Rng(seed, count)
        descs, cols = zip(*(rng(*line.split()) for line in lines[2:]))

    df = pd.DataFrame({i: col for i, col in enumerate(cols)})
    df.columns = descs

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)


if __name__ == "__main__":
    main()