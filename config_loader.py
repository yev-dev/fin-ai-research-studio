import json
import os

def load_config(config_path='market_analysis/config/distressed_stock.json', overrides=None):
    """
    Loads configuration from a JSON file and applies optional overrides.
    Returns a dictionary representing the configuration.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    if overrides:
        # Perform a shallow merge for top-level keys
        # For deeper structures, more complex merging would be needed
        for key, value in overrides.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                config[key].update(value)
            else:
                config[key] = value
        
    return config
