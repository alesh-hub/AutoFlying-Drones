import os
ROOT = os.path.dirname(os.path.dirname(__file__))   # .../repo/python
SIM_CONFIG = os.path.join(ROOT, "sim_config")

def scene(name: str) -> str:
    return os.path.join(SIM_CONFIG, name)
