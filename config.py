# my_module/config_loader.py
import yaml
from pathlib import Path

# Path to your config file
CONFIG_PATH = Path(__file__).parent / "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

# Load once at import time (global config)
CONFIG = load_config()
