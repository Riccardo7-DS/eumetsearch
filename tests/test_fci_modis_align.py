"""
test_fci_modis_align.py
=======================
Unit / integration tests for fci_modis_align.py.

All tests use **synthetic** data so no real Zarr files are required.
Run with:  python test_fci_modis_align.py
"""

import warnings
import numpy as np
from pyproj import CRS, Transformer

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1] / "src" / "eumetsearch" / "transform"))
from fci_modis_align import (
    FCIGeometry, MODISGeometry, TargetGrid,
    _aeqd_to_latlon, _latlon_to_fci_px, _latlon_to_modis_px,
    _sample_array, align_patches,
    latlon_to_fci_fulldisk_rowcol, latlon_to_modis_global_rowcol,
    fci_patch_corner_for_cell, modis_patch_corner_for_cell,
)

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

failures = []

def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}  {detail}")
        failures.append(name)

def section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ─────────────────────────────────────────────────────────────
# 1.  Geometry constants
# ─────────────────────────────────────────────────────────────
section("Geometry constants")

check("FCI SCALE_X ≈ 1000 m/px",
      abs(FCIGeometry.SCALE_X - 1000) < 2,
      f"got {FCIGeometry.SCALE_X:.3f}")

check("MODIS true pixel ≈ 231.656 m",
      abs(MODISGeometry.PIXEL_SIZE_M - 231.65625) < 0.001,
      f"got {MODISGeometry.PIXEL_SIZE_M:.6f}")

check("MODIS patch extent ≈ 118608 m",
      abs(MODISGeometry.PATCH_EXTENT_M - 118608.0) < 0.1,
      f"got {MODISGeometry.PATCH_EXTENT_M:.1f}")


# ─────────────────────────────────────────────────────────────
# 2.  Round-trip: lat/lon → FCI full-disk → lat/lon
# ─────────────────────────────────────────────────────────────
section("FCI coordinate round-trip")

test_points = [
    (0.0,  0.0,   "sub-satellite point"),
    (45.0, 10.0,  "mid-latitude"),
    (-32.0, 30.0, "southern hemisphere, oblique"),
    (51.5, -0.1,  "London"),
]

geo = FCIGeometry
t_fwd = Transformer.from_crs(CRS.from_epsg(4326),
                              CRS.from_proj4(geo.CRS_PROJ_STRING),
                              always_xy=True)
t_inv = Transformer.from_crs(CRS.from_proj4(geo.CRS_PROJ_STRING),
                              CRS.from_epsg(4326),
                              always_xy=True)

for (lat, lon, label) in test_points:
    x, y = t_fwd.transform(lon, lat)
    lon2, lat2 = t_inv.transform(x, y)
    err_m = np.hypot((lat2 - lat) * 111320, (lon2 - lon) * 111320 * np.cos(np.radians(lat)))
    check(f"round-trip {label} < 1 m",
          err_m < 1.0, f"error = {err_m:.2f} m")


# ─────────────────────────────────────────────────────────────
# 3.  latlon_to_fci_fulldisk_rowcol — round() vs int()
# ─────────────────────────────────────────────────────────────
section("Patch centre extraction (round vs int)")

lat0, lon0 = 48.8, 2.35   # Paris
row_r, col_r = latlon_to_fci_fulldisk_rowcol(lat0, lon0)
# int() version (row 0 = south edge)
x, y = t_fwd.transform(lon0, lat0)
col_i = int((x - geo.X_WEST)  / geo.SCALE_X - 0.5)
row_i = int((y  - geo.Y_SOUTH) / geo.SCALE_Y - 0.5)
# round() result must be at most 1 px off (the same or better)
check("round() ≤ int() error for Paris FCI",
      abs(row_r - row_i) <= 1 and abs(col_r - col_i) <= 1)


# ─────────────────────────────────────────────────────────────
# 4.  TargetGrid shape and extent
# ─────────────────────────────────────────────────────────────
section("TargetGrid construction")

lat0, lon0 = 45.0, 10.0
g = TargetGrid(cell_lat=lat0, cell_lon=lon0)

check("Grid square", g.n_px == g.n_px and len(g.xs) == len(g.ys))
check("Grid covers ~118.6 km",
      abs(g.xs[-1] - g.xs[0] + g.resolution_m - MODISGeometry.PATCH_EXTENT_M) < 500,
      f"extent = {g.xs[-1]-g.xs[0]+g.resolution_m:.0f} m")
check("First xs is negative (west half)", g.xs[0] < 0)
check("First ys is positive (north half)", g.ys[0] > 0)

g2 = TargetGrid(cell_lat=lat0, cell_lon=lon0, n_pixels=256)
check("n_pixels=256 override", g2.n_px == 256)
check("n_pixels=256 resolution scales correctly",
      abs(g2.resolution_m - MODISGeometry.PATCH_EXTENT_M / 256) < 1)


# ─────────────────────────────────────────────────────────────
# 5.  _aeqd_to_latlon centre pixel is the cell centre
# ─────────────────────────────────────────────────────────────
section("AEQD → lat/lon: centre pixel accuracy")

for (lat0, lon0, label) in [(0, 0, "equator/prime"), (48.8, 2.35, "Paris"),
                              (-33.9, 18.4, "Cape Town")]:
    g = TargetGrid(cell_lat=lat0, cell_lon=lon0)
    lat_grid, lon_grid = _aeqd_to_latlon(g)
    # Centre pixel
    cx = g.n_px // 2
    lat_c = lat_grid[cx, cx]
    lon_c = lon_grid[cx, cx]
    err_km = np.hypot(lat_c - lat0, lon_c - lon0) * 111.32
    check(f"Centre pixel < 1 km from cell centre ({label})",
          err_km < 1.0, f"err = {err_km:.3f} km")


# ─────────────────────────────────────────────────────────────
# 6.  _latlon_to_fci_px: centre of patch maps back to centre
# ─────────────────────────────────────────────────────────────
section("FCI inverse mapping: centre → patch centre")

for (lat0, lon0, label) in [(0, 0, "nadir"), (48.8, 2.35, "Paris"),
                              (-32.0, 30.0, "oblique")]:
    row0, col0 = fci_patch_corner_for_cell(lat0, lon0, 128)
    # The exact cell centre in lat/lon
    lat_arr = np.array([[lat0]])
    lon_arr = np.array([[lon0]])
    fr, fc = _latlon_to_fci_px(lat_arr, lon_arr, row0, col0)
    # Should land near 63.5 (centre of 128 px patch)
    target = 63.5
    err_r = abs(float(fr[0, 0]) - target)
    err_c = abs(float(fc[0, 0]) - target)
    check(f"FCI patch centre mapping < 1.0 px ({label})",
          err_r < 1.0 and err_c < 1.0,
          f"row={fr[0,0]:.2f} col={fc[0,0]:.2f} (target≈63.5)")


# ─────────────────────────────────────────────────────────────
# 7.  _latlon_to_modis_px: centre of patch maps back to centre
# ─────────────────────────────────────────────────────────────
section("MODIS inverse mapping: centre → patch centre")

mg = MODISGeometry
t_sin_fwd = Transformer.from_crs(CRS.from_epsg(4326),
                                   CRS.from_proj4(mg.CRS_PROJ_STRING),
                                   always_xy=True)

for (lat0, lon0, label) in [(0, 20, "equatorial Africa"),
                              (45.0, 10.0, "northern Italy"),
                              (-20.0, 130.0, "Australia")]:
    row0, col0 = modis_patch_corner_for_cell(lat0, lon0, 512)
    lat_arr = np.array([[lat0]])
    lon_arr = np.array([[lon0]])
    fr, fc = _latlon_to_modis_px(lat_arr, lon_arr, row0, col0)
    target = 255.5
    err_r = abs(float(fr[0, 0]) - target)
    err_c = abs(float(fc[0, 0]) - target)
    check(f"MODIS patch centre mapping < 1.0 px ({label})",
          err_r < 1.0 and err_c < 1.0,
          f"row={fr[0,0]:.2f} col={fc[0,0]:.2f} (target≈255.5)")


# ─────────────────────────────────────────────────────────────
# 8.  _sample_array: flat patch → constant output
# ─────────────────────────────────────────────────────────────
section("_sample_array: flat-value correctness")

flat = np.full((128, 128), 1000.0, dtype=np.float32)
rows = np.array([[20.3, 63.5, 100.7]])
cols = np.array([[20.3, 63.5, 100.7]])
out = _sample_array(flat, rows, cols)
check("Flat array returns constant value",
      np.allclose(out, 1000.0, atol=1e-3))

# NaN passthrough
rows_nan = np.array([[np.nan, 63.5]])
cols_nan = np.array([[20.3,   np.nan]])
out_nan = _sample_array(flat, rows_nan, cols_nan)
check("NaN coords produce NaN output",
      np.isnan(out_nan[0, 0]) and np.isnan(out_nan[0, 1]))


# ─────────────────────────────────────────────────────────────
# 9.  align_patches: output shape and coverage
# ─────────────────────────────────────────────────────────────
section("align_patches: output shape, coverage, no NaN bleed")

lat0, lon0 = 45.0, 10.0
fci_r0, fci_c0 = fci_patch_corner_for_cell(lat0, lon0, 128)
mod_r0, mod_c0 = modis_patch_corner_for_cell(lat0, lon0, 512)

rng = np.random.default_rng(42)
fci_patch  = rng.integers(0, 10000, (128, 128), dtype=np.uint16)
modis_patch = rng.integers(0, 10000, (512, 512), dtype=np.uint16)

result = align_patches(
    fci_patch=fci_patch,
    fci_patch_row0=fci_r0, fci_patch_col0=fci_c0,
    modis_patch=modis_patch,
    modis_patch_row0=mod_r0, modis_patch_col0=mod_c0,
    cell_lat=lat0, cell_lon=lon0,
)

check("Output FCI shape is (n_px, n_px)",
      result.fci_reprojected.shape == (result.grid.n_px, result.grid.n_px))
check("Output MODIS shape is (n_px, n_px)",
      result.modis_reprojected.shape == (result.grid.n_px, result.grid.n_px))

fci_cov  = result.fci_coverage.mean()
mod_cov  = result.modis_coverage.mean()
check("FCI coverage > 10% for mid-lat European cell",
      fci_cov > 0.10, f"got {fci_cov:.1%}")
check("MODIS coverage > 80% for mid-lat cell",
      mod_cov > 0.80, f"got {mod_cov:.1%}")

# Values inside coverage should not be NaN
fci_in_cov  = result.fci_reprojected[result.fci_coverage]
mod_in_cov  = result.modis_reprojected[result.modis_coverage]
check("No NaN inside FCI coverage mask",
      not np.isnan(fci_in_cov).any())
check("No NaN inside MODIS coverage mask",
      not np.isnan(mod_in_cov).any())


# ─────────────────────────────────────────────────────────────
# 10.  align_patches: oblique FCI cell (high-distortion regime)
# ─────────────────────────────────────────────────────────────
section("align_patches: oblique FCI cell (lat=-32, lon=30)")

lat0, lon0 = -32.0, 30.0
fci_r0, fci_c0 = fci_patch_corner_for_cell(lat0, lon0, 128)
mod_r0, mod_c0 = modis_patch_corner_for_cell(lat0, lon0, 512)

result2 = align_patches(
    fci_patch=fci_patch,
    fci_patch_row0=fci_r0, fci_patch_col0=fci_c0,
    modis_patch=modis_patch,
    modis_patch_row0=mod_r0, modis_patch_col0=mod_c0,
    cell_lat=lat0, cell_lon=lon0,
)
fci_cov2 = result2.fci_coverage.mean()
# Coverage will be lower here (only part of FCI 128px patch overlaps AEQD grid)
check("Oblique FCI: some valid output pixels exist",
      fci_cov2 > 0.01, f"got {fci_cov2:.1%}")
check("Oblique MODIS: coverage > 80%",
      result2.modis_coverage.mean() > 0.80)


# ─────────────────────────────────────────────────────────────
# 11.  align_patches with n_pixels override
# ─────────────────────────────────────────────────────────────
section("align_patches: target_n_pixels override")

result3 = align_patches(
    fci_patch=fci_patch,
    fci_patch_row0=fci_r0, fci_patch_col0=fci_c0,
    modis_patch=modis_patch,
    modis_patch_row0=mod_r0, modis_patch_col0=mod_c0,
    cell_lat=45.0, cell_lon=10.0,
    target_n_pixels=256,
)
check("n_pixels=256 output shape",
      result3.fci_reprojected.shape == (256, 256))
check("n_pixels=256 MODIS shape",
      result3.modis_reprojected.shape == (256, 256))


# ─────────────────────────────────────────────────────────────
# 12.  align_patches: GSD sanity — ground extent via AEQD
# ─────────────────────────────────────────────────────────────
section("Target grid GSD sanity check")

g = result.grid
# xs[-1] - xs[0] + resolution = full span
full_span = g.xs[-1] - g.xs[0] + g.resolution_m
check("Full span ≈ 118.6 km (MODIS patch extent)",
      abs(full_span - MODISGeometry.PATCH_EXTENT_M) < 500,
      f"got {full_span/1000:.3f} km")


# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
if failures:
    print(f"  {FAIL}  {len(failures)} test(s) FAILED: {', '.join(failures)}")
    sys.exit(1)
else:
    total = 0  # counted by check() calls above
    print(f"  {PASS}  All tests passed.")
print(f"{'='*60}\n")
