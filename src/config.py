"""Configuration loading and output locations."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Read a YAML config and resolve its paths against the repository root."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {key: ROOT / value for key, value in cfg["paths"].items()}
    return cfg


def parse_config(description: str | None = None) -> dict:
    """Load the config named on the command line."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    return load_config(parser.parse_args().config)


def results_path(cfg: dict, filename: str) -> Path:
    """Path of an intermediate table."""
    folder = cfg["paths"]["results"]
    folder.mkdir(parents=True, exist_ok=True)
    return folder / filename


def source_data_path(cfg: dict, filename: str) -> Path:
    """Path of a table holding the values of a figure panel."""
    folder = cfg["paths"]["results"] / "source_data"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / filename
