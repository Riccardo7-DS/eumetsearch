"""
Pre-defined geographic bounding boxes and tile-grid generators.

All values are stored and returned in (W, S, E, N) order, which is the
natural order for most geospatial libraries.  The eumdac API expects
[N, S, W, E]; convert at the call site with:

    W, S, E, N = get_bbox("mtg")
    NSWE = [N, S, W, E]
"""
import itertools
import math

# Registry of named bounding boxes: (W, S, E, N) in decimal degrees.
BBOX_REGISTRY: dict[str, tuple[float, float, float, float]] = {
    # Full MTG-I1 visible disk (conservative, excludes limb)
    "mtg":      (-18.105469, -37.857507,  60.820313,  71.413177),
    # Horn of Africa
    "hoa":      ( 32.016304,  -5.483696,  51.483696,  15.483696),
    # Europe
    "europe":   (-25.0,        34.0,       45.0,       72.0),
    # Africa
    "africa":   (-20.0,       -35.0,       55.0,       38.0),
    # Middle East
    "mideast":  ( 25.0,        12.0,       65.0,       42.0),
    # Brazil
    "brazil":   (-74.0,       -34.0,      -34.0,        5.5),
}


def get_bbox(name: str) -> tuple[float, float, float, float]:
    """
    Return the (W, S, E, N) bounding box for a named region.

    Parameters
    ----------
    name : str
        Region key from BBOX_REGISTRY (case-insensitive).

    Raises
    ------
    KeyError
        If the name is not in the registry.
    """
    key = name.strip().lower()
    if key not in BBOX_REGISTRY:
        available = ", ".join(sorted(BBOX_REGISTRY))
        raise KeyError(f"Unknown bbox '{name}'. Available regions: {available}")
    return BBOX_REGISTRY[key]


def bbox_mtg() -> list[float]:
    """Return [W, S, E, N] for the full MTG visible disk (backwards-compat)."""
    return list(BBOX_REGISTRY["mtg"])


def hoa_bbox(invert: bool = False) -> list[float]:
    """
    Return the Horn of Africa bounding box.

    Parameters
    ----------
    invert : bool
        If False (default) returns [W, S, E, N].
        If True returns [S, W, N, E] (legacy order used by some callers).
    """
    W, S, E, N = BBOX_REGISTRY["hoa"]
    if invert:
        return [S, W, N, E]
    return [W, S, E, N]


def generate_bboxes_fixed(
    bbox: tuple[float, float, float, float],
    n_lat: int,
    n_lon: int,
) -> list[tuple[float, float, float, float]]:
    """
    Divide *bbox* into an n_lat × n_lon regular tile grid.

    Parameters
    ----------
    bbox : (W, S, E, N)
        Outer bounding box in decimal degrees.
    n_lat : int
        Number of rows (latitude splits).
    n_lon : int
        Number of columns (longitude splits).

    Returns
    -------
    list of (W, S, E, N) tuples, row-major order (south → north, west → east).
    """
    W, S, E, N = bbox
    lat_step = (N - S) / n_lat
    lon_step = (E - W) / n_lon

    tiles = []
    for i, j in itertools.product(range(n_lat), range(n_lon)):
        tile_s = S + i * lat_step
        tile_n = tile_s + lat_step
        tile_w = W + j * lon_step
        tile_e = tile_w + lon_step
        tiles.append((tile_w, tile_s, tile_e, tile_n))
    return tiles


def generate_bboxes_from_resolution(
    bbox: tuple[float, float, float, float],
    resolution_deg: float,
) -> list[tuple[float, float, float, float]]:
    """
    Divide *bbox* into tiles of approximately *resolution_deg* × *resolution_deg*.

    The number of tiles in each axis is rounded up so the whole AOI is covered;
    tile edges may extend slightly beyond the outer bbox.

    Parameters
    ----------
    bbox : (W, S, E, N)
    resolution_deg : float
        Approximate tile side length in decimal degrees.

    Returns
    -------
    list of (W, S, E, N) tuples.
    """
    W, S, E, N = bbox
    n_lat = math.ceil((N - S) / resolution_deg)
    n_lon = math.ceil((E - W) / resolution_deg)
    return generate_bboxes_fixed(bbox, n_lat=n_lat, n_lon=n_lon)
