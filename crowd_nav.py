"""Package-name alias, so `from crowd_nav import CrowdNavController` works.

The implementation lives in `peroi_controller.py` (PeRoI is the method's name in the paper; the
repository is `crowd-nav`). Both spellings refer to the same class -- use whichever reads better in
your codebase.

    from crowd_nav import CrowdNavController      # or
    from peroi_controller import PeRoIController  # identical
"""
from peroi_controller import (  # noqa: F401
    PeRoIController,
    ResidualPredictor,
    load_predictor,
    DT,
    PAST,
    FUTURE,
)

CrowdNavController = PeRoIController

__all__ = ["CrowdNavController", "PeRoIController", "ResidualPredictor",
           "load_predictor", "DT", "PAST", "FUTURE"]
