"""
MajorTOM-style grid utilities for FCI geostationary data.

Grid IDs use the official MajorTOM variable-density format ``"{n}U_{m}R"``
(north of equator, east of prime meridian) / ``"{n}D_{m}L"`` etc., exactly
matching the ``majortom.Grid`` library used by MODIS-MajorTOM and other
MajorTOM-family datasets.  This ensures that the same geographic cell has the
same grid_id across all MajorTOM-compatible datasets.

Row labels  : ``{n}U`` = n rows north of equator, ``{n}D`` = n rows south.
Column labels: ``{n}R`` = n cols east  of 0° lon,  ``{n}L`` = n cols west.

An external grid file (GeoParquet / GeoJSON) can still be loaded by passing
``grid_path`` to ``FCIMajorTomGrid``.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)

# Approximate FCI full-disk extent from MTG-1 at 0°E (geostationary).
FCI_LAT_MIN = -80.0
FCI_LAT_MAX = 80.0
FCI_LON_MIN = -80.0
FCI_LON_MAX = 80.0

# Earth radius used by the official majortom library (WGS-84 equatorial, km).
_MAJORTOM_R_KM = 6378.137

# Legacy constant kept for backward-compatible helper functions.
_KM_PER_DEG = 111.32


# ---------------------------------------------------------------------------
# MajorTOM grid geometry helpers (private)
# ---------------------------------------------------------------------------

def _majortom_row_latitudes(spacing_km: float) -> np.ndarray:
    """
    Global MajorTOM row latitudes, south → north.

    Matches the official ``majortom.Grid`` library exactly: -90° is remapped
    to 0° via ``mod(lat, 180) - 90`` so that 0° (equator) is always present
    and ``zeroth_row`` is well-defined regardless of whether *n* is odd or even.
    """
    n = math.ceil(math.pi * _MAJORTOM_R_KM / spacing_km)
    lats = np.linspace(-90.0, 90.0, n + 1)[:-1]
    lats = np.mod(lats, 180.0) - 90.0  # -90 → 0, ensures equator is in the grid
    return np.sort(lats)


def _majortom_col_longitudes(lat: float, spacing_km: float) -> np.ndarray:
    """
    MajorTOM column longitudes at *lat*, west → east.

    The -180° point is remapped to 0° (prime meridian origin) so that
    ``0R`` / ``0L`` always corresponds to lon = 0°.
    """
    circ = 2.0 * math.pi * _MAJORTOM_R_KM * math.cos(math.radians(lat))
    n = math.ceil(circ / spacing_km)
    lons = np.linspace(-180.0, 180.0, n + 1)[:-1]
    lons = np.mod(lons, 360.0) - 180.0  # maps -180 → 0 (prime-meridian origin)
    return np.sort(lons)


def _zeroth_row(lats: np.ndarray) -> int:
    """Index of the 0° (equator) row in *lats* — always present after mod transform."""
    idx = np.where(np.abs(lats) < 1e-9)[0]
    return int(idx[0]) if len(idx) else int(np.searchsorted(lats, 0.0))


def _zeroth_col(lons: np.ndarray) -> int:
    """Index of the 0° (prime meridian) column in *lons* — always present after mod transform."""
    idx = np.where(np.abs(lons) < 1e-9)[0]
    return int(idx[0]) if len(idx) else int(np.argmin(np.abs(lons)))


def _row_label(row_idx: int, zeroth_row: int) -> str:
    n = row_idx - zeroth_row
    return f"{n}U" if n >= 0 else f"{-n}D"


def _col_label(col_idx: int, zeroth_col_idx: int) -> str:
    n = col_idx - zeroth_col_idx
    return f"{n}R" if n >= 0 else f"{-n}L"


# ---------------------------------------------------------------------------
# Public grid-generation functions
# ---------------------------------------------------------------------------

def generate_majortom_grid(
    lat_min: float = FCI_LAT_MIN,
    lat_max: float = FCI_LAT_MAX,
    lon_min: float = FCI_LON_MIN,
    lon_max: float = FCI_LON_MAX,
    spacing_km: float = 100.0,
):
    """
    Generate a MajorTOM-compatible variable-density grid.

    Row spacing is uniform along meridians (~``spacing_km`` between rows).
    Column spacing is uniform along each parallel at each latitude, so the
    number of columns decreases towards the poles — matching the official
    ``majortom.Grid`` library used by MODIS-MajorTOM and other datasets.

    Grid IDs are in ``"{n}U_{m}R"`` / ``"{n}D_{m}L"`` format and are
    geographically identical to those produced by ``majortom.Grid(dist=spacing_km)``.

    Returns a :class:`pandas.DataFrame` with columns
    ``[grid_id, centre_lat, centre_lon]``.
    """
    import pandas as pd

    lats = _majortom_row_latitudes(spacing_km)
    zeroth_row = _zeroth_row(lats)

    records = []
    for row_idx in np.where((lats >= lat_min) & (lats <= lat_max))[0]:
        lat = float(lats[row_idx])
        row_lbl = _row_label(row_idx, zeroth_row)

        lons = _majortom_col_longitudes(lat, spacing_km)
        z_col = _zeroth_col(lons)

        for col_idx in np.where((lons >= lon_min) & (lons <= lon_max))[0]:
            lon = float(lons[col_idx])
            records.append({
                "grid_id": f"{row_lbl}_{_col_label(col_idx, z_col)}",
                "centre_lat": lat,
                "centre_lon": lon,
            })

    df = pd.DataFrame(records)
    logger.info(
        f"Generated MajorTOM grid: {len(df)} cells at {spacing_km:.0f} km spacing "
        f"(bbox lat [{lat_min},{lat_max}] lon [{lon_min},{lon_max}])"
    )
    return df


def generate_regular_grid(
    lat_min: float = FCI_LAT_MIN,
    lat_max: float = FCI_LAT_MAX,
    lon_min: float = FCI_LON_MIN,
    lon_max: float = FCI_LON_MAX,
    spacing_km: float = 100.0,
):
    """
    Generate a *rectangular* lat/lon grid (equal spacing in degrees).

    .. deprecated::
        Use :func:`generate_majortom_grid` for grids compatible with other
        MajorTOM datasets.  This function produces ``"RRRR_CCCC"`` style IDs
        (row 0 = 90°N, col 0 = 180°W) which are **not** geographically
        equivalent to the variable-density MajorTOM IDs.
    """
    import pandas as pd

    spacing_deg = spacing_km / _KM_PER_DEG

    row_start = int(np.ceil((90.0 - lat_max) / spacing_deg))
    row_end   = int(np.floor((90.0 - lat_min) / spacing_deg))
    col_start = int(np.ceil((180.0 + lon_min) / spacing_deg))
    col_end   = int(np.floor((180.0 + lon_max) / spacing_deg))

    records = []
    for r in range(row_start, row_end + 1):
        lat = 90.0 - r * spacing_deg
        for c in range(col_start, col_end + 1):
            lon = -180.0 + c * spacing_deg
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                records.append(
                    {"grid_id": f"{r:04d}_{c:04d}", "centre_lat": lat, "centre_lon": lon}
                )

    df = pd.DataFrame(records)
    logger.info(
        f"Generated regular grid: {len(df)} cells at {spacing_km:.0f} km spacing "
        f"(bbox lat [{lat_min},{lat_max}] lon [{lon_min},{lon_max}])"
    )
    return df


# ---------------------------------------------------------------------------
# Public grid_id ↔ lat/lon conversion utilities
# ---------------------------------------------------------------------------

def grid_id_to_latlon(
    grid_id: str,
    spacing_km: float = 100.0,
) -> tuple[float, float]:
    """
    Convert a MajorTOM ``grid_id`` to ``(lat, lon)`` in decimal degrees.

    Handles both the official variable-density format (``"23D_51L"``,
    ``"0U_0R"``) and the legacy rectangular format (``"0266_0764"``).

    Parameters
    ----------
    grid_id :
        e.g. ``"23D_51L"`` or ``"0U_15R"``.
    spacing_km :
        Grid spacing in km (default 100).  Must match the spacing used when
        the grid was generated.

    Returns
    -------
    (lat, lon) : tuple of float
    """
    parts = grid_id.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse grid_id {grid_id!r}: expected one '_' separator")

    row_str, col_str = parts
    row_m = re.match(r"^(\d+)([UDud]?)$", row_str)
    col_m = re.match(r"^(\d+)([RLrl]?)$", col_str)
    if not row_m or not col_m:
        raise ValueError(f"Cannot parse grid_id {grid_id!r}")

    row_n   = int(row_m.group(1))
    row_dir = row_m.group(2).upper()
    col_n   = int(col_m.group(1))
    col_dir = col_m.group(2).upper()

    lats = _majortom_row_latitudes(spacing_km)
    zeroth_row = _zeroth_row(lats)

    if row_dir == "D":
        row_idx = zeroth_row - row_n
    elif row_dir == "U":
        row_idx = zeroth_row + row_n
    else:
        # Legacy rectangular format: row counts from north pole
        spacing_deg = spacing_km / _KM_PER_DEG
        return 90.0 - row_n * spacing_deg, -180.0 + col_n * spacing_deg

    if not (0 <= row_idx < len(lats)):
        raise ValueError(f"Row index {row_idx} out of bounds for grid_id {grid_id!r}")

    lat = float(lats[row_idx])
    lons = _majortom_col_longitudes(lat, spacing_km)
    z_col = _zeroth_col(lons)

    if col_dir == "L":
        col_idx = z_col - col_n
    else:  # 'R' or empty
        col_idx = z_col + col_n

    if not (0 <= col_idx < len(lons)):
        raise ValueError(f"Column index {col_idx} out of bounds for grid_id {grid_id!r}")

    return lat, float(lons[col_idx])


def latlon_to_grid_id(
    lat: float,
    lon: float,
    spacing_km: float = 100.0,
) -> str:
    """
    Snap ``(lat, lon)`` to the nearest MajorTOM grid cell and return its
    ``grid_id`` (e.g. ``"23D_51L"``).

    Parameters
    ----------
    lat, lon : float
        Geographic coordinates in decimal degrees.
    spacing_km :
        Grid spacing in km (default 100).

    Returns
    -------
    str
        Grid ID in official MajorTOM format.
    """
    lats = _majortom_row_latitudes(spacing_km)
    zeroth_row = _zeroth_row(lats)
    row_idx = int(np.argmin(np.abs(lats - lat)))
    snapped_lat = float(lats[row_idx])

    lons = _majortom_col_longitudes(snapped_lat, spacing_km)
    z_col = _zeroth_col(lons)
    col_idx = int(np.argmin(np.abs(lons - lon)))

    return f"{_row_label(row_idx, zeroth_row)}_{_col_label(col_idx, z_col)}"


def grid_ids_to_latlons(
    grid_ids,
    spacing_km: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised decode of an iterable of ``grid_id`` strings.

    Returns
    -------
    lats, lons : np.ndarray of float, shape (N,)
    """
    lats_out, lons_out = [], []
    for gid in grid_ids:
        lat, lon = grid_id_to_latlon(gid, spacing_km=spacing_km)
        lats_out.append(lat)
        lons_out.append(lon)
    return np.array(lats_out), np.array(lons_out)


# ---------------------------------------------------------------------------
# External grid loader
# ---------------------------------------------------------------------------

def _load_majortom_grid(grid_path: str | Path):
    """
    Load an external MajorTOM grid file (GeoParquet or GeoJSON).

    Returns a DataFrame/GeoDataFrame with ``grid_id``, ``centre_lat``,
    ``centre_lon`` columns (normalised from various naming conventions).
    """
    import geopandas as gpd

    grid_path = Path(grid_path)
    logger.info(f"Loading MajorTOM grid from {grid_path}")
    if grid_path.suffix in (".parquet", ".geoparquet"):
        gdf = gpd.read_parquet(grid_path)
    else:
        gdf = gpd.read_file(grid_path)

    col_map = {}
    for raw, canonical in [
        ("centre_lat", "centre_lat"), ("center_lat", "centre_lat"),
        ("lat", "centre_lat"),        ("latitude", "centre_lat"),
        ("centre_lon", "centre_lon"), ("center_lon", "centre_lon"),
        ("lon", "centre_lon"),        ("longitude", "centre_lon"),
        ("grid_id", "grid_id"),       ("name", "grid_id"),
    ]:
        if raw in gdf.columns and canonical not in col_map.values():
            col_map[raw] = canonical
    gdf = gdf.rename(columns=col_map)

    if "centre_lat" not in gdf.columns and gdf.geometry is not None:
        gdf = gdf.copy()
        gdf["centre_lon"] = gdf.geometry.x
        gdf["centre_lat"] = gdf.geometry.y

    return gdf


# ---------------------------------------------------------------------------
# Legacy pixel-level helpers
# ---------------------------------------------------------------------------

def latlon_to_fci_pixel(area, lat: float, lon: float) -> tuple[int, int] | tuple[None, None]:
    """
    Convert a (lat, lon) coordinate to native FCI scene pixel (row, col).

    Parameters
    ----------
    area : pyresample.AreaDefinition
    lat, lon : float

    Returns
    -------
    (row, col) ints, or (None, None) if outside the scene.
    """
    from pyproj import CRS, Transformer

    try:
        transformer = Transformer.from_crs(
            CRS.from_epsg(4326), area.crs, always_xy=True
        )
        x, y = transformer.transform(lon, lat)

        x_ll, y_ll, x_ur, y_ur = area.area_extent
        col = (x - x_ll) / area.pixel_size_x
        row = (y_ur - y) / area.pixel_size_y

        col_i, row_i = int(col), int(row)
        if 0 <= row_i < area.height and 0 <= col_i < area.width:
            return row_i, col_i
        return None, None
    except Exception:
        return None, None


def extract_fci_patch(
    data: np.ndarray,
    row: int,
    col: int,
    patch_half: int,
    max_null_fraction: float = 0.5,
) -> np.ndarray | None:
    """
    Extract a square patch of shape ``(2*patch_half, 2*patch_half)`` centred
    at ``(row, col)``.

    Returns ``None`` if the patch lies outside the array bounds or contains
    more than ``max_null_fraction`` NaN / fill values.
    """
    h, w = data.shape[-2], data.shape[-1]
    r0, r1 = row - patch_half, row + patch_half
    c0, c1 = col - patch_half, col + patch_half

    if r0 < 0 or r1 > h or c0 < 0 or c1 > w:
        return None

    patch = data[..., r0:r1, c0:c1]
    null_fraction = np.isnan(patch).mean() if np.issubdtype(patch.dtype, np.floating) else 0.0
    if null_fraction > max_null_fraction:
        return None

    return patch


# ---------------------------------------------------------------------------
# FCIMajorTomGrid
# ---------------------------------------------------------------------------

class FCIMajorTomGrid:
    """
    Manages grid cells that fall within the FCI visible disk.

    By default a variable-density MajorTOM-compatible grid is generated at
    ``spacing_km`` spacing (100 km), using the same row/column layout as the
    official ``majortom.Grid`` library so that grid_ids are globally consistent
    across MajorTOM datasets.

    Pass ``grid_path`` to load an external grid file instead.

    Parameters
    ----------
    grid_path :
        Path to an external grid file (GeoParquet or GeoJSON).
    spacing_km :
        Cell spacing in km for the generated grid (default 100).
    lat_min, lat_max, lon_min, lon_max :
        Bounding box to pre-filter grid cells.
    """

    def __init__(
        self,
        grid_path: str | Path | None = None,
        spacing_km: float = 100.0,
        lat_min: float = FCI_LAT_MIN,
        lat_max: float = FCI_LAT_MAX,
        lon_min: float = FCI_LON_MIN,
        lon_max: float = FCI_LON_MAX,
    ):
        if grid_path is not None:
            gdf = _load_majortom_grid(grid_path)
            mask = (
                (gdf["centre_lat"] >= lat_min) & (gdf["centre_lat"] <= lat_max) &
                (gdf["centre_lon"] >= lon_min) & (gdf["centre_lon"] <= lon_max)
            )
            self.grid = gdf[mask].reset_index(drop=True)
            logger.info(f"FCIMajorTomGrid: {len(self.grid)} cells from file within bbox")
        else:
            self.grid = generate_majortom_grid(
                lat_min=lat_min, lat_max=lat_max,
                lon_min=lon_min, lon_max=lon_max,
                spacing_km=spacing_km,
            )
        self._spacing_km = spacing_km

    def visible_pixel_coords(self, area) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return ``(rows, cols, grid_ids)`` for all grid cells visible in ``area``.

        All coordinate transforms are vectorised (one pyproj call for the full
        grid).

        Parameters
        ----------
        area : pyresample.AreaDefinition

        Returns
        -------
        rows, cols : np.ndarray of int, shape (N,)
        grid_ids  : np.ndarray of str, shape (N,)
        """
        from pyproj import CRS, Transformer

        lats = self.grid["centre_lat"].values
        lons = self.grid["centre_lon"].values
        gids = self.grid["grid_id"].values

        transformer = Transformer.from_crs(
            CRS.from_epsg(4326), area.crs, always_xy=True
        )
        xs, ys = transformer.transform(lons, lats)

        x_ll, y_ll, x_ur, y_ur = area.area_extent
        x_min, x_max = min(x_ll, x_ur), max(x_ll, x_ur)
        y_min, y_max = min(y_ll, y_ur), max(y_ll, y_ur)

        in_disk = (
            (xs >= x_min) & (xs <= x_max) &
            (ys >= y_min) & (ys <= y_max) &
            np.isfinite(xs) & np.isfinite(ys)
        )
        xs, ys, gids = xs[in_disk], ys[in_disk], gids[in_disk]

        cols = ((xs - x_ll) / area.pixel_size_x).astype(int)
        rows = ((y_ur - ys) / area.pixel_size_y).astype(int)

        valid = (rows >= 0) & (rows < area.height) & (cols >= 0) & (cols < area.width)
        return rows[valid], cols[valid], gids[valid].astype(str)

    def cells_in_area(self, area) -> Iterator[tuple[str, float, float]]:
        """Yield ``(grid_id, lat, lon)`` for cells visible in ``area``.

        Kept for backward compatibility — new code should use
        ``visible_pixel_coords`` which is fully vectorised.
        """
        rows, cols, gids = self.visible_pixel_coords(area)
        from pyproj import CRS, Transformer
        transformer = Transformer.from_crs(area.crs, CRS.from_epsg(4326), always_xy=True)
        x_ll = area.area_extent[0]
        y_ur = area.area_extent[3]
        for r, c, gid in zip(rows, cols, gids):
            x = x_ll + c * area.pixel_size_x
            y = y_ur - r * area.pixel_size_y
            lon, lat = transformer.transform(x, y)
            yield gid, lat, lon
