# Modular imports
import importlib
import warnings

# Classic control is in gym and thus necessary for the base version to run
from carl.envs.classic_control import *

# Environment loading
box2d_spec = importlib.util.find_spec("Box2D")
found = box2d_spec is not None
if found:
    from carl.envs.box2d import *
else:
    warnings.warn(
        "Module 'Box2D' not found. If you want to use these environments, please follow the installation guide."
    )
