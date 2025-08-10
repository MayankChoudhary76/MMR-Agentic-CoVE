from .utils import *        # re-export utils (paths, config)
from .data import *         # re-export data (registry, loader)
from .models import *       # re-export models (encoders, fusion)
from .service import *      # re-export service (recommender)

__all__ = []  # namespace passthrough; subpackages manage their own __all__