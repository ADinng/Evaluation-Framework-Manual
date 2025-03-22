import argparse
import subprocess
import sys
from multiprocessing.pool import ThreadPool
from pathlib import Path

CWD = Path(__file__).parent.resolve()
GENERATOR = CWD / "generator.py"
PLOTTER = CWD / "plotter.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Driver script to process .cfg files and generate plots.")
    parser.add_argument("path", type=str, help="Path to the directory containing .cfg files (default: current directory)", default=".")
    return parser.parse_args()


def process_config(config: Path) -> None:
    name = config.stem
    csv = config.with_name(f"{name}-data.csv")
    cdf = config.with_name(f"{name}-CDF.pdf")
    box = config.with_name(f"{name}-Box.pdf")
    
    # Call generator.py with the .cfg file as an argument
    subprocess.run([sys.executable, GENERATOR, "-c", str(config), "-o", str(csv)], check=True)
    
    # Call plotter.py with the generated .csv file as an argument
    subprocess.run([sys.executable, PLOTTER, "-d", str(csv), "-c", str(cdf), "-b", str(box)], check=True)


def main() -> None:
    args = parse_args()
    path = Path(args.path).resolve()
    configs = list(path.glob("*.cfg"))
    
    # Use ThreadPool to process multiple .cfg files in parallel
    with ThreadPool() as pool:
        pool.map(process_config, configs)


if __name__ == "__main__":
    main()