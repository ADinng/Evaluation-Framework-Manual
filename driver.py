import argparse
import subprocess
import sys
from pathlib import Path


def get_arguments():
    parser = argparse.ArgumentParser(description="Process .cfg files and generate plots.")
    parser.add_argument("path", type=str, help="Path to the directory containing .cfg files", default=".")
    return parser.parse_args()


def process_cfg(config_file: Path):
    base_name = config_file.stem
    csv_file = config_file.with_name(f"{base_name}-data.csv")
    cdf_file = config_file.with_name(f"{base_name}-CDF.pdf")
    box_file = config_file.with_name(f"{base_name}-Box.pdf")

    subprocess.run([sys.executable, "generator.py", "-c", str(config_file), "-o", str(csv_file)], check=True)
    subprocess.run([sys.executable, "plotter.py", "-d", str(csv_file), "-c", str(cdf_file), "-b", str(box_file)], check=True)


def main():
    args = get_arguments()
    config_dir = Path(args.path).resolve()
    config_files = list(config_dir.glob("*.cfg"))
    
    for config_file in config_files:
        print(f"Processing {config_file.name}...")
        process_cfg(config_file)
        print(f"Finished processing {config_file.name}.")


if __name__ == "__main__":
    main()