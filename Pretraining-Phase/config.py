import yaml
from types import SimpleNamespace


def _dict_to_namespace(d: dict):
    """Convert dict → SimpleNamespace recursively."""
    ns = {}
    for k, v in d.items():
        if isinstance(v, dict):
            ns[k] = _dict_to_namespace(v)
        else:
            ns[k] = v
    return SimpleNamespace(**ns)


def load_config(path: str = "config.yaml"):
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)
        print("Loading the config file...")
    return _dict_to_namespace(cfg_dict)