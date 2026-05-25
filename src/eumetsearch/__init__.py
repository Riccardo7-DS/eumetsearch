from .utils import *
from .transform import FCIMajorTomGrid, extract_fci_patch, latlon_to_fci_pixel

try:
    from .data_collection import *
except ImportError:
    pass  # eumdac / EUMETSAT download deps not installed — download features unavailable

try:
    from . import viz
except ImportError:
    pass  # matplotlib not installed — visualization features unavailable