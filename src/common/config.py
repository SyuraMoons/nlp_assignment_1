"""Shared project config loader.

All scripts are run from the repository root as modules, e.g.
``python -m src.preprocessing.clean_text``; paths in config.yaml are relative to
the repository root.
"""
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def data_path(config, key):
    """Absolute path for one of the ``paths:`` entries in config.yaml."""
    return PROJECT_ROOT / config["paths"][key]
