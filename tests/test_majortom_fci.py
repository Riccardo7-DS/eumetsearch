"""
Tests for eumetsearch.transform.majortom_fci:
  extract_fci_patch, latlon_to_fci_pixel, FCIMajorTomGrid
"""

import types
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from eumetsearch.transform.majortom_fci import (
    FCIMajorTomGrid,
    extract_fci_patch,
    latlon_to_fci_pixel,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

class _MockArea:
    """Minimal pyresample-like AreaDefinition for testing coordinate math."""

    def __init__(self):
        # Simple 100×200 grid over 0–10°E, 0–5°N in a plain metres-like CRS.
        # We use EPSG:4326 as the "native" CRS so pyproj Transformer is identity.
        from pyproj import CRS
        self.crs = CRS.from_epsg(4326)
        # area_extent: (x_ll, y_ll, x_ur, y_ur) in the native CRS units (degrees for 4326)
        self.area_extent = (0.0, 0.0, 10.0, 5.0)
        self.width = 200   # columns (x dimension)
        self.height = 100  # rows (y dimension)
        self.pixel_size_x = (10.0 - 0.0) / self.width   # 0.05 deg per pixel
        self.pixel_size_y = (5.0  - 0.0) / self.height  # 0.05 deg per pixel


@pytest.fixture
def mock_area():
    return _MockArea()


def _make_gdf(lats, lons, grid_ids=None):
    """Build a minimal GeoDataFrame with centre_lat / centre_lon / grid_id."""
    import geopandas as gpd
    from shapely.geometry import Point

    if grid_ids is None:
        grid_ids = [f"{i:04d}_{i:04d}" for i in range(len(lats))]
    df = pd.DataFrame({
        "centre_lat": lats,
        "centre_lon": lons,
        "grid_id": grid_ids,
        "geometry": [Point(lon, lat) for lat, lon in zip(lats, lons)],
    })
    return gpd.GeoDataFrame(df, crs="EPSG:4326")


# ---------------------------------------------------------------------------
# extract_fci_patch
# ---------------------------------------------------------------------------

class TestExtractFciPatch:
    def test_returns_correct_shape(self):
        data = np.ones((100, 100), dtype=np.float32)
        patch = extract_fci_patch(data, row=50, col=50, patch_half=10)
        assert patch is not None
        assert patch.shape == (20, 20)

    def test_returns_none_when_out_of_bounds_top(self):
        data = np.ones((100, 100))
        assert extract_fci_patch(data, row=5, col=50, patch_half=10) is None

    def test_returns_none_when_out_of_bounds_bottom(self):
        data = np.ones((100, 100))
        assert extract_fci_patch(data, row=95, col=50, patch_half=10) is None

    def test_returns_none_when_out_of_bounds_left(self):
        data = np.ones((100, 100))
        assert extract_fci_patch(data, row=50, col=5, patch_half=10) is None

    def test_returns_none_when_out_of_bounds_right(self):
        data = np.ones((100, 100))
        assert extract_fci_patch(data, row=50, col=95, patch_half=10) is None

    def test_returns_none_when_too_many_nans(self):
        data = np.full((100, 100), np.nan, dtype=np.float32)
        assert extract_fci_patch(data, row=50, col=50, patch_half=10) is None

    def test_returns_patch_when_nan_below_threshold(self):
        data = np.ones((100, 100), dtype=np.float32)
        data[45:55, 45:55] = np.nan  # some NaN in a 10x10 region
        patch = extract_fci_patch(data, row=50, col=50, patch_half=20,
                                  max_null_fraction=0.9)
        assert patch is not None

    def test_works_with_3d_data(self):
        data = np.ones((3, 100, 100), dtype=np.float32)
        patch = extract_fci_patch(data, row=50, col=50, patch_half=8)
        assert patch is not None
        assert patch.shape == (3, 16, 16)

    def test_integer_dtype_does_not_nan_check(self):
        data = np.zeros((100, 100), dtype=np.int32)
        patch = extract_fci_patch(data, row=50, col=50, patch_half=10)
        assert patch is not None

    def test_exact_boundary_valid(self):
        data = np.ones((100, 100), dtype=np.float32)
        # patch_half=10, row=10 → r0=0, r1=20 — exactly on boundary
        patch = extract_fci_patch(data, row=10, col=50, patch_half=10)
        assert patch is not None

    def test_exact_boundary_invalid(self):
        data = np.ones((100, 100), dtype=np.float32)
        # row=9 → r0=-1 — out of bounds
        assert extract_fci_patch(data, row=9, col=50, patch_half=10) is None


# ---------------------------------------------------------------------------
# latlon_to_fci_pixel
# ---------------------------------------------------------------------------

class TestLatLonToFciPixel:
    def test_centre_point_maps_to_centre_pixel(self, mock_area):
        # Centre of the area: lon=5, lat=2.5 → col≈100, row≈50
        row, col = latlon_to_fci_pixel(mock_area, lat=2.5, lon=5.0)
        assert row is not None and col is not None
        assert abs(col - 100) <= 2
        assert abs(row - 50) <= 2

    def test_returns_none_for_point_outside_area(self, mock_area):
        row, col = latlon_to_fci_pixel(mock_area, lat=10.0, lon=20.0)
        assert row is None and col is None

    def test_corner_lower_left(self, mock_area):
        # Lower-left corner (lat≈0, lon≈0) → near col 0, but near the last row
        # because row 0 is the top (y_ur) and row max is the bottom (y_ll).
        row, col = latlon_to_fci_pixel(mock_area, lat=0.025, lon=0.025)
        assert row is not None
        assert 0 <= col < 5        # near left edge
        assert row >= 95           # near bottom row

    def test_types_are_int(self, mock_area):
        row, col = latlon_to_fci_pixel(mock_area, lat=2.5, lon=5.0)
        assert isinstance(row, int) and isinstance(col, int)

    def test_negative_coords_outside(self, mock_area):
        row, col = latlon_to_fci_pixel(mock_area, lat=-1.0, lon=-1.0)
        assert row is None and col is None


# ---------------------------------------------------------------------------
# FCIMajorTomGrid
# ---------------------------------------------------------------------------

class TestFCIMajorTomGrid:
    def _patched_grid(self, gdf, lat_min=-80, lat_max=80, lon_min=-80, lon_max=80):
        with patch(
            "eumetsearch.transform.majortom_fci._load_majortom_grid",
            return_value=gdf,
        ):
            return FCIMajorTomGrid(
                lat_min=lat_min, lat_max=lat_max,
                lon_min=lon_min, lon_max=lon_max,
            )

    def test_filters_cells_outside_bbox(self):
        gdf = _make_gdf(
            lats=[10.0, 50.0, 90.0],
            lons=[5.0, 5.0, 5.0],
        )
        grid = self._patched_grid(gdf, lat_min=-80, lat_max=80)
        assert len(grid.grid) == 2  # 90° latitude is outside ±80

    def test_all_cells_in_bbox_kept(self):
        gdf = _make_gdf(lats=[0.0, 20.0, -20.0], lons=[0.0, 30.0, -30.0])
        grid = self._patched_grid(gdf)
        assert len(grid.grid) == 3

    def test_normalises_center_lat_column_name(self):
        import geopandas as gpd
        from shapely.geometry import Point

        df = pd.DataFrame({
            "center_lat": [10.0],
            "center_lon": [5.0],
            "grid_id": ["0001_0001"],
            "geometry": [Point(5.0, 10.0)],
        })
        gdf = gpd.GeoDataFrame(df, crs="EPSG:4326")
        grid = self._patched_grid(gdf)
        assert "centre_lat" in grid.grid.columns

    def test_normalises_latitude_column_name(self):
        import geopandas as gpd
        from shapely.geometry import Point

        df = pd.DataFrame({
            "latitude": [10.0],
            "longitude": [5.0],
            "name": ["0001_0001"],
            "geometry": [Point(5.0, 10.0)],
        })
        gdf = gpd.GeoDataFrame(df, crs="EPSG:4326")
        grid = self._patched_grid(gdf)
        assert "centre_lat" in grid.grid.columns
        assert "grid_id" in grid.grid.columns

    def test_cells_in_area_yields_visible_cells(self, mock_area):
        # Three cells: two inside mock_area (0–10°lon, 0–5°lat), one outside
        gdf = _make_gdf(
            lats=[2.5, 4.0, 20.0],
            lons=[5.0, 8.0, 50.0],
            grid_ids=["inside_1", "inside_2", "outside"],
        )
        grid = self._patched_grid(gdf, lat_min=-90, lat_max=90,
                                   lon_min=-180, lon_max=180)
        visible = list(grid.cells_in_area(mock_area))
        grid_ids = [v[0] for v in visible]
        assert "inside_1" in grid_ids
        assert "inside_2" in grid_ids
        assert "outside" not in grid_ids

    def test_cells_in_area_yields_lat_lon(self, mock_area):
        gdf = _make_gdf(lats=[2.5], lons=[5.0], grid_ids=["cell_0"])
        grid = self._patched_grid(gdf, lat_min=-90, lat_max=90,
                                   lon_min=-180, lon_max=180)
        cells = list(grid.cells_in_area(mock_area))
        assert len(cells) == 1
        gid, lat, lon = cells[0]
        assert gid == "cell_0"
        assert lat == pytest.approx(2.5)
        assert lon == pytest.approx(5.0)

    def test_cells_in_area_empty_when_no_cells_visible(self, mock_area):
        gdf = _make_gdf(lats=[20.0], lons=[50.0])  # outside mock_area
        grid = self._patched_grid(gdf, lat_min=-90, lat_max=90,
                                   lon_min=-180, lon_max=180)
        assert list(grid.cells_in_area(mock_area)) == []
