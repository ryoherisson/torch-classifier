import yaml

def load_yaml(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return cfg