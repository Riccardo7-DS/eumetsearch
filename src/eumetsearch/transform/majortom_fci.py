"""
MajorTOM-style grid utilities for FCI geostationary data.

By default a regular lat/lon grid is generated at a fixed spacing (100 km).
Grid IDs use global row/col numbering (row 0 = 90°N, col 0 = 180°W) so IDs
are stable across different AOIs and consistent with the MajorTOM convention
``"RRRR_CCCC"``.

An external grid file (GeoParquet / GeoJSON) can still be loaded by passing
``grid_path`` to ``FCIMajorTomGrid``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)

# Approximate FCI full-disk extent from MTG-1 at 0°E (geostationary).
FCI_LAT_MIN = -80.0
FCI_LAT_MAX = 80.0
FCI_LON_MIN = -80.0
FCI_LON_MAX = 80.0

_KM_PER_DEG = 111.32  # km per degree latitude


def generate_regular_grid(
    lat_min: float = FCI_LAT_MIN,
    lat_max: float = FCI_LAT_MAX,
    lon_min: float = FCI_LON_MIN,
    lon_max: float = FCI_LON_MAX,
    spacing_km: float = 100.0,
):
    """
    Generate a regular lat/lon grid with approximately ``spacing_km`` between
    cell centres.

    Returns a :class:`pandas.DataFrame` with columns
    ``[grid_id, centre_lat, centre_lon]``.

    Grid IDs are derived from global row/col indices so the same geographic
    point always gets the same ID regardless of the AOI used.

    Parameters
    ----------
    lat_min, lat_max, lon_min, lon_max :
        Bounding box for the generated cells.
    spacing_km :
        Approximate spacing between cell centres in kilometres.
    """
    import pandas as pd

    spacing_deg = spacing_km / _KM_PER_DEG

    # Global indices: row 0 → lat = 90°N, col 0 → lon = -180°E
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


def _load_majortom_grid(grid_path: str | Path):
    """
    Load an external MajorTOM grid file.

    Returns a DataFrame/GeoDataFrame with at least ``grid_id``,
    ``centre_lat``, and ``centre_lon`` columns (normalised from the various
    naming conventions used by different MajorTOM releases).
    """
    import geopandas as gpd

    grid_path = Path(grid_path)
    logger.info(f"Loading MajorTOM grid from {grid_path}")
    if grid_path.suffix in (".parquet", ".geoparquet"):
        gdf = gpd.read_parquet(grid_path)
    else:
        gdf = gpd.read_file(grid_path)

    # Normalise column names
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

    # Derive lat/lon from geometry when missing
    if "centre_lat" not in gdf.columns and gdf.geometry is not None:
        gdf = gdf.copy()
        gdf["centre_lon"] = gdf.geometry.x
        gdf["centre_lat"] = gdf.geometry.y

    return gdf


def latlon_to_fci_pixel(area, lat: float, lon: float) -> tuple[int, int] | tuple[None, None]:
    """
    Convert a (lat, lon) coordinate to native FCI scene pixel (row, col).

    Parameters
    ----------
    area : pyresample.AreaDefinition
        The native area of the loaded FCI scene (from ``scn[channel].area``).
    lat, lon : float
        Geographic coordinates in degrees.

    Returns
    -------
    (row, col) ints, or (None, None) if the point is outside the scene.
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
    Extract a square patch of shape ``(2*patch_half, 2*patch_half)`` from
    ``data`` centred at ``(row, col)``.

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


class FCIMajorTomGrid:
    """
    Manages grid cells that fall within the FCI visible disk.

    By default a regular lat/lon grid is generated at ``spacing_km`` spacing
    (100 km).  Pass ``grid_path`` to load an external MajorTOM grid file
    instead (GeoParquet or GeoJSON).

    Parameters
    ----------
    grid_path :
        Path to an external grid file.  When provided, ``spacing_km`` is ignored.
    spacing_km :
        Cell spacing in km for the generated regular grid (default 100).
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
            # Filter loaded grid to requested bbox
            mask = (
                (gdf["centre_lat"] >= lat_min) & (gdf["centre_lat"] <= lat_max) &
                (gdf["centre_lon"] >= lon_min) & (gdf["centre_lon"] <= lon_max)
            )
            self.grid = gdf[mask].reset_index(drop=True)
            logger.info(f"FCIMajorTomGrid: {len(self.grid)} cells from file within bbox")
        else:
            self.grid = generate_regular_grid(
                lat_min=lat_min, lat_max=lat_max,
                lon_min=lon_min, lon_max=lon_max,
                spacing_km=spacing_km,
            )

    def cells_in_area(self, area) -> Iterator[tuple[str, float, float]]:
        """
        Yield ``(grid_id, lat, lon)`` for all grid cells visible in ``area``.

        Uses a fast projection-bounds check before doing per-pixel lookups.
        """
        from pyproj import CRS, Transformer

        transformer = Transformer.from_crs(
            CRS.from_epsg(4326), area.crs, always_xy=True
        )
        x_ll, y_ll, x_ur, y_ur = area.area_extent
        # FCI geostationary area_extent has inverted y-axis (y_ll > y_ur), so
        # use min/max to get the actual bounds regardless of axis orientation.
        x_min, x_max = min(x_ll, x_ur), max(x_ll, x_ur)
        y_min, y_max = min(y_ll, y_ur), max(y_ll, y_ur)

        for _, row in self.grid.iterrows():
            lat, lon = float(row["centre_lat"]), float(row["centre_lon"])
            try:
                x, y = transformer.transform(lon, lat)
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    yield str(row["grid_id"]), lat, lon
            except Exception:
                continue
