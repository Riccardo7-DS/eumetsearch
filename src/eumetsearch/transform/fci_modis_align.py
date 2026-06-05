"""
fci_modis_align.py
==================
Geometrically correct alignment of FCI (geostationary) and MODIS MOD09GQ
(polar-orbit sinusoidal) patches onto a common Earth-fixed reference grid.

Design principle
----------------
Do NOT align in patch/pixel space.  Instead:

    Earth-fixed target grid (AEQD, centred on MajorTOM cell)
           │
           ├─► project each pixel into FCI geostationary space  → sample FCI
           └─► project each pixel into MODIS sinusoidal space   → sample MODIS

Every output pixel is located independently; no affine approximation is used.

Key constants (FCI MTG-I1)
--------------------------
Full-disk:  11136 × 11136 px
Projection: +proj=geos +lon_0=0 +h=35785831 +a=6378169 +b=6356583.8
Extent:     ±5 571 248 m (x),  ±5 570 248 m (y)   [EUMETSAT ICD values]

Key constants (MODIS MOD09GQ)
-----------------------------
Tile pixels: 4800 px per MODIS tile side
Tile extent: 1 111 950 m  (1 × sin-proj unit per tile)
True pixel:  231.656 25 m/px  (NOT the nominal 250 m label)
Patch size:  512 px  →  118 607.2 m ≈ 118.6 km
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from pyproj import CRS, Transformer
from scipy.ndimage import map_coordinates


# ---------------------------------------------------------------------------
# 1.  Sensor geometry constants
# ---------------------------------------------------------------------------

class FCIGeometry:
    """MTG FCI full-disk geostationary geometry (EUMETSAT ICD-SBV-0064).

    Scan order note
    ---------------
    The FCI L1C product (as read by satpy) stores rows south→north: row 0 is
    the southernmost line of the visible disk.  The satpy pyresample area
    therefore has a *negative* pixel_size_y (y_ll = north edge, y_ur = south
    edge in the area_extent tuple).

    Consequence for row mapping
    ---------------------------
    Given a geostationary y coordinate (positive = northward, standard PROJ
    convention), the full-disk pixel row is:

        row = (y_geos - Y_SOUTH) / SCALE_Y - 0.5

    where Y_SOUTH is the *geographic* y value of the south edge of the disk
    (a negative number in standard geos coordinates).
    """

    FULL_DISK_PX: int = 11136

    # Projection semi-axes from the official CRS
    A_METERS: float = 6_378_169.0       # semi-major axis
    B_METERS: float = 6_356_583.8       # semi-minor axis
    H_METERS: float = 35_785_831.0      # satellite altitude above ellipsoid
    LON_0_DEG: float = 0.0

    # Full-disk extent from areas.yaml (EUMETSAT ICD values)
    # X_WEST / X_EAST : western / eastern edge x in geos metres
    # Y_NORTH / Y_SOUTH: northern / southern edge y in geos metres
    # (Y_SOUTH is negative because south of the equator)
    X_WEST:  float = -5_571_248.412732527
    X_EAST:  float =  5_566_247.740968115
    Y_NORTH: float =  5_571_248.412732527
    Y_SOUTH: float = -5_566_247.740968115

    SCALE_X: float = (X_EAST  - X_WEST)  / FULL_DISK_PX   # m per pixel (col)
    SCALE_Y: float = (Y_NORTH - Y_SOUTH) / FULL_DISK_PX   # m per pixel (row)

    CRS_PROJ_STRING: str = (
        "+proj=geos +lon_0=0 +h=35785831 "
        "+a=6378169 +b=6356583.8 +units=m"
    )

    @classmethod
    def crs(cls) -> CRS:
        return CRS.from_proj4(cls.CRS_PROJ_STRING)


class MODISGeometry:
    """MODIS MOD09GQ sinusoidal grid geometry."""

    # True pixel size derived from tile dimensions
    TILE_SIZE_M: float = 1_111_950.0    # one MODIS tile side in projection m
    PIXELS_PER_TILE: int = 4_800
    PIXEL_SIZE_M: float = TILE_SIZE_M / PIXELS_PER_TILE  # 231.65625 m

    # MODIS sinusoidal origin (upper-left of tile h00v00)
    XMIN_GLOBAL: float = -20_015_109.354  # m
    YMAX_GLOBAL: float =  10_007_554.677  # m

    PATCH_PX: int = 512
    PATCH_EXTENT_M: float = PATCH_PX * PIXEL_SIZE_M   # ≈ 118 607.2 m

    CRS_PROJ_STRING: str = (
        "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 "
        "+a=6371007.181 +b=6371007.181 +units=m"
    )

    @classmethod
    def crs(cls) -> CRS:
        return CRS.from_proj4(cls.CRS_PROJ_STRING)


# ---------------------------------------------------------------------------
# 2.  Target (output) grid definition
# ---------------------------------------------------------------------------

@dataclass
class TargetGrid:
    """
    An Earth-fixed AEQD raster centred on a MajorTOM cell.

    Parameters
    ----------
    cell_lat, cell_lon : float
        MajorTOM cell centre in WGS-84 degrees.
    extent_m : float
        Half-size of the square grid in metres.
        Default matches the MODIS patch (118 607.2 / 2 ≈ 59 304 m).
    resolution_m : float
        Ground sampling distance in metres for the output grid.
    n_pixels : int  (optional override)
        If provided, overrides resolution_m with extent*2/n_pixels.
    """
    cell_lat: float
    cell_lon: float
    extent_m: float = field(default=MODISGeometry.PATCH_EXTENT_M / 2)
    resolution_m: float = MODISGeometry.PIXEL_SIZE_M
    n_pixels: Optional[int] = None

    # Derived, filled in __post_init__
    n_px: int = field(init=False)
    xs: np.ndarray = field(init=False, repr=False)   # 1-D AEQD x coords
    ys: np.ndarray = field(init=False, repr=False)   # 1-D AEQD y coords (N→S)
    xx: np.ndarray = field(init=False, repr=False)   # 2-D meshgrid
    yy: np.ndarray = field(init=False, repr=False)
    crs: CRS = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_pixels is not None:
            self.resolution_m = 2 * self.extent_m / self.n_pixels
            self.n_px = self.n_pixels
        else:
            self.n_px = int(round(2 * self.extent_m / self.resolution_m))

        self.crs = CRS.from_proj4(
            f"+proj=aeqd +lat_0={self.cell_lat} +lon_0={self.cell_lon} "
            "+datum=WGS84 +units=m"
        )
        # Pixel centres in AEQD metres; y decreasing (North-up raster)
        half = self.extent_m
        coords = (np.arange(self.n_px) + 0.5) * self.resolution_m - half
        self.xs = coords
        self.ys = coords[::-1].copy()
        self.xx, self.yy = np.meshgrid(self.xs, self.ys)


# ---------------------------------------------------------------------------
# 3.  Reprojection helpers
# ---------------------------------------------------------------------------

def _aeqd_to_latlon(grid: TargetGrid) -> Tuple[np.ndarray, np.ndarray]:
    """Convert every AEQD output pixel to WGS-84 lat/lon."""
    t = Transformer.from_crs(grid.crs, CRS.from_epsg(4326), always_xy=True)
    lon, lat = t.transform(grid.xx, grid.yy)
    return lat, lon


def _latlon_to_fci_px(
    lat: np.ndarray, lon: np.ndarray,
    fci_patch_row0: int,
    fci_patch_col0: int,
    fci_patch_h: int = 128,
    fci_patch_w: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map (lat, lon) → fractional pixel coordinates inside the FCI patch.

    Parameters
    ----------
    lat, lon : ndarray  (H_out × W_out)
    fci_patch_row0, fci_patch_col0 : int
        Top-left corner of the extracted FCI patch in full-disk pixel coords.
        (Row 0 = north edge of full disk.)
    fci_patch_h, fci_patch_w : int
        Size of the FCI patch (default 128×128).

    Returns
    -------
    frac_row, frac_col : ndarray  (H_out × W_out)
        Fractional index into the FCI patch; NaN where outside or over the
        Earth limb (geostationary projection is undefined beyond the disk).
    """
    geo = FCIGeometry
    t = Transformer.from_crs(
        CRS.from_epsg(4326),
        CRS.from_proj4(geo.CRS_PROJ_STRING),
        always_xy=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x_geos, y_geos = t.transform(lon, lat)

    # Over-the-limb pixels become ±inf or nan from pyproj
    limb_mask = ~np.isfinite(x_geos) | ~np.isfinite(y_geos)

    # Convert geostationary metres → full-disk pixel index (0-based, float)
    # Column:  x increases eastward, col increases eastward
    # Row:     FCI scans south→north so row 0 = south edge; y increases northward
    col_full = (x_geos - geo.X_WEST)  / geo.SCALE_X - 0.5
    row_full = (y_geos  - geo.Y_SOUTH) / geo.SCALE_Y - 0.5

    # Convert to patch-local fractional index
    frac_col = col_full - fci_patch_col0
    frac_row = row_full - fci_patch_row0

    # Mask out-of-bounds and limb.
    # For bilinear interpolation (order=1), scipy.map_coordinates returns NaN
    # if the fractional index is strictly > size-1 (it would need to sample the
    # out-of-bounds neighbour pixel).  The safe range is exactly [0, size-1].
    outside = (
        limb_mask
        | (frac_row < 0) | (frac_row > fci_patch_h - 1)
        | (frac_col < 0) | (frac_col > fci_patch_w - 1)
    )
    frac_row[outside] = np.nan
    frac_col[outside] = np.nan

    return frac_row, frac_col


def _latlon_to_modis_px(
    lat: np.ndarray, lon: np.ndarray,
    modis_patch_row0: int,
    modis_patch_col0: int,
    modis_patch_h: int = 512,
    modis_patch_w: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map (lat, lon) → fractional pixel coordinates inside the MODIS patch.

    Parameters
    ----------
    modis_patch_row0, modis_patch_col0 : int
        Top-left corner of the MODIS patch in the global sinusoidal grid
        (i.e. the absolute pixel index, not within a single MODIS tile).
        Row 0 = northernmost row of the global grid.
    """
    mg = MODISGeometry
    t = Transformer.from_crs(
        CRS.from_epsg(4326),
        CRS.from_proj4(mg.CRS_PROJ_STRING),
        always_xy=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x_sin, y_sin = t.transform(lon, lat)

    # Sinusoidal global pixel index (0-based, float)
    # x increases eastward; col increases eastward
    # y increases northward; row increases southward
    col_global = (x_sin - mg.XMIN_GLOBAL) / mg.PIXEL_SIZE_M - 0.5
    row_global = (mg.YMAX_GLOBAL - y_sin) / mg.PIXEL_SIZE_M - 0.5

    frac_col = col_global - modis_patch_col0
    frac_row = row_global - modis_patch_row0

    outside = (
        ~np.isfinite(frac_row) | ~np.isfinite(frac_col)
        | (frac_row < 0) | (frac_row > modis_patch_h - 1)
        | (frac_col < 0) | (frac_col > modis_patch_w - 1)
    )
    frac_row[outside] = np.nan
    frac_col[outside] = np.nan

    return frac_row, frac_col


def _sample_array(
    source: np.ndarray,
    frac_row: np.ndarray,
    frac_col: np.ndarray,
    order: int = 1,
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Bilinear (order=1) sampling of *source* at fractional indices.

    NaN coordinates produce fill_value in the output.
    The output dtype is float32 regardless of the input dtype.
    """
    out = np.full(frac_row.shape, fill_value, dtype=np.float32)
    valid = np.isfinite(frac_row) & np.isfinite(frac_col)
    if not valid.any():
        return out

    out[valid] = map_coordinates(
        source.astype(np.float32),
        [frac_row[valid], frac_col[valid]],
        order=order,
        mode="constant",
        cval=fill_value,
        prefilter=False,
    )
    return out


# ---------------------------------------------------------------------------
# 4.  Public API
# ---------------------------------------------------------------------------

@dataclass
class AlignedPatches:
    """Result container returned by :func:`align_patches`."""
    fci_reprojected: np.ndarray      # (n_px, n_px) float32
    modis_reprojected: np.ndarray    # (n_px, n_px) float32
    grid: TargetGrid
    fci_coverage: np.ndarray         # bool mask: True where FCI data present
    modis_coverage: np.ndarray       # bool mask: True where MODIS data present


def align_patches(
    fci_patch: np.ndarray,
    fci_patch_row0: int,
    fci_patch_col0: int,
    modis_patch: np.ndarray,
    modis_patch_row0: int,
    modis_patch_col0: int,
    cell_lat: float,
    cell_lon: float,
    target_resolution_m: float = MODISGeometry.PIXEL_SIZE_M,
    target_n_pixels: Optional[int] = None,
    target_extent_m: Optional[float] = None,
    interpolation_order: int = 1,
) -> AlignedPatches:
    """
    Reproject FCI and MODIS patches onto a common Earth-fixed AEQD grid.

    Both sensors are sampled independently by projecting each output pixel
    into the respective sensor's native coordinate system; no affine
    approximation or patch-space cropping is used.

    Parameters
    ----------
    fci_patch : ndarray, shape (128, 128)
        Raw FCI patch in geostationary pixel space (any band, uint16 or float).
    fci_patch_row0, fci_patch_col0 : int
        Top-left corner of the FCI patch in the **full-disk** pixel grid
        (row 0 = north edge, col 0 = west edge of the 11136×11136 disk).
        Use ``round()`` – not ``int()`` – when computing these from lat/lon.
    modis_patch : ndarray, shape (512, 512)
        Raw MODIS MOD09GQ patch in sinusoidal pixel space.
    modis_patch_row0, modis_patch_col0 : int
        Top-left corner of the MODIS patch in the **global sinusoidal** pixel
        grid (row 0 = north edge of YMAX_GLOBAL, col 0 = west edge of
        XMIN_GLOBAL).  Use ``round()`` – not ``int()``.
    cell_lat, cell_lon : float
        MajorTOM cell centre (WGS-84 degrees).
    target_resolution_m : float
        Output pixel size in metres (default: true MODIS pixel, 231.656 m).
    target_n_pixels : int, optional
        If set, forces the output to exactly this many pixels per side
        (overrides target_resolution_m).
    target_extent_m : float, optional
        Half-width of the output square in metres.
        Defaults to half the MODIS patch extent (≈59 304 m).
    interpolation_order : int
        scipy map_coordinates order: 0=nearest, 1=bilinear (default), 3=cubic.

    Returns
    -------
    AlignedPatches
        Both reprojected arrays on the same grid plus coverage masks.
    """
    # --- build output grid ---------------------------------------------------
    extent = target_extent_m if target_extent_m is not None else MODISGeometry.PATCH_EXTENT_M / 2
    grid = TargetGrid(
        cell_lat=cell_lat,
        cell_lon=cell_lon,
        extent_m=extent,
        resolution_m=target_resolution_m,
        n_pixels=target_n_pixels,
    )

    # --- lat/lon for every output pixel --------------------------------------
    lat, lon = _aeqd_to_latlon(grid)

    # --- FCI reprojection ----------------------------------------------------
    fci_frac_row, fci_frac_col = _latlon_to_fci_px(
        lat, lon,
        fci_patch_row0, fci_patch_col0,
        fci_patch.shape[0], fci_patch.shape[1],
    )
    fci_out = _sample_array(fci_patch, fci_frac_row, fci_frac_col,
                            order=interpolation_order)

    # --- MODIS reprojection --------------------------------------------------
    modis_frac_row, modis_frac_col = _latlon_to_modis_px(
        lat, lon,
        modis_patch_row0, modis_patch_col0,
        modis_patch.shape[0], modis_patch.shape[1],
    )
    modis_out = _sample_array(modis_patch, modis_frac_row, modis_frac_col,
                              order=interpolation_order)

    return AlignedPatches(
        fci_reprojected=fci_out,
        modis_reprojected=modis_out,
        grid=grid,
        fci_coverage=np.isfinite(fci_frac_row) & np.isfinite(fci_frac_col),
        modis_coverage=np.isfinite(modis_frac_row) & np.isfinite(modis_frac_col),
    )


# ---------------------------------------------------------------------------
# 4b.  Multi-band convenience wrapper
# ---------------------------------------------------------------------------

def align_patches_multiband(
    fci_bands: dict,
    fci_patch_row0: int,
    fci_patch_col0: int,
    modis_bands: dict,
    modis_patch_row0: int,
    modis_patch_col0: int,
    cell_lat: float,
    cell_lon: float,
    target_resolution_m: float = MODISGeometry.PIXEL_SIZE_M,
    target_n_pixels: Optional[int] = None,
    target_extent_m: Optional[float] = None,
    interpolation_order: int = 1,
) -> dict:
    """
    Reproject multiple FCI and MODIS bands onto a common AEQD grid,
    computing the projection coordinates only once per sensor.

    Parameters
    ----------
    fci_bands : dict[str, ndarray]
        e.g. ``{"red": arr_128x128, "nir": arr_128x128}``
    modis_bands : dict[str, ndarray]
        e.g. ``{"red": arr_512x512, "nir": arr_512x512}``
    fci_patch_row0, fci_patch_col0 : int
        Top-left corner of the FCI patch in full-disk pixel coordinates,
        as stored in the ``patch_origins`` Zarr array.
    modis_patch_row0, modis_patch_col0 : int
        Top-left corner of the MODIS patch in global sinusoidal pixel
        coordinates, computed via ``modis_patch_corner_for_cell``.
    (remaining parameters identical to ``align_patches``)

    Returns
    -------
    dict with keys:
        ``"fci"``            – dict[str, ndarray] aligned FCI bands
        ``"modis"``          – dict[str, ndarray] aligned MODIS bands
        ``"grid"``           – TargetGrid
        ``"fci_coverage"``   – bool ndarray
        ``"modis_coverage"`` – bool ndarray
    """
    extent = target_extent_m if target_extent_m is not None else MODISGeometry.PATCH_EXTENT_M / 2
    grid = TargetGrid(
        cell_lat=cell_lat, cell_lon=cell_lon,
        extent_m=extent, resolution_m=target_resolution_m, n_pixels=target_n_pixels,
    )

    lat, lon = _aeqd_to_latlon(grid)

    # FCI fractional coordinates — computed once for all FCI bands
    fci_h, fci_w = next(iter(fci_bands.values())).shape[-2:]
    fci_fr, fci_fc = _latlon_to_fci_px(
        lat, lon, fci_patch_row0, fci_patch_col0, fci_h, fci_w,
    )
    fci_coverage = np.isfinite(fci_fr) & np.isfinite(fci_fc)

    # MODIS fractional coordinates — computed once for all MODIS bands
    mod_h, mod_w = next(iter(modis_bands.values())).shape[-2:]
    mod_fr, mod_fc = _latlon_to_modis_px(
        lat, lon, modis_patch_row0, modis_patch_col0, mod_h, mod_w,
    )
    modis_coverage = np.isfinite(mod_fr) & np.isfinite(mod_fc)

    fci_out   = {k: _sample_array(v, fci_fr, fci_fc, order=interpolation_order)
                 for k, v in fci_bands.items()}
    modis_out = {k: _sample_array(v, mod_fr, mod_fc, order=interpolation_order)
                 for k, v in modis_bands.items()}

    return {
        "fci":            fci_out,
        "modis":          modis_out,
        "grid":           grid,
        "fci_coverage":   fci_coverage,
        "modis_coverage": modis_coverage,
    }


# ---------------------------------------------------------------------------
# 5.  Coordinate helpers  (use these when extracting patches from Zarr)
# ---------------------------------------------------------------------------

def latlon_to_fci_fulldisk_rowcol(lat: float, lon: float) -> Tuple[int, int]:
    """
    Convert a geographic point to the nearest FCI full-disk pixel.

    Returns (row, col) using round() to avoid subpixel bias.
    Row 0 = south edge of full disk (FCI scans south→north).
    """
    geo = FCIGeometry
    t = Transformer.from_crs(
        CRS.from_epsg(4326),
        CRS.from_proj4(geo.CRS_PROJ_STRING),
        always_xy=True,
    )
    x_geos, y_geos = t.transform(lon, lat)
    col = int(round((x_geos - geo.X_WEST)  / geo.SCALE_X - 0.5))
    row = int(round((y_geos  - geo.Y_SOUTH) / geo.SCALE_Y - 0.5))
    return row, col


def latlon_to_modis_global_rowcol(lat: float, lon: float) -> Tuple[int, int]:
    """
    Convert a geographic point to the nearest MODIS global sinusoidal pixel.

    Returns (row, col) using round() to avoid subpixel bias.
    Row 0 = north edge of the global YMAX_GLOBAL boundary.
    """
    mg = MODISGeometry
    t = Transformer.from_crs(
        CRS.from_epsg(4326),
        CRS.from_proj4(mg.CRS_PROJ_STRING),
        always_xy=True,
    )
    x_sin, y_sin = t.transform(lon, lat)
    col = int(round((x_sin - mg.XMIN_GLOBAL) / mg.PIXEL_SIZE_M - 0.5))
    row = int(round((mg.YMAX_GLOBAL - y_sin) / mg.PIXEL_SIZE_M - 0.5))
    return row, col


def fci_patch_corner_for_cell(
    cell_lat: float, cell_lon: float, patch_px: int = 128
) -> Tuple[int, int]:
    """
    Return the top-left (row0, col0) in full-disk coords for a centred patch.

    The centre pixel is computed with round(), not int().
    """
    row_c, col_c = latlon_to_fci_fulldisk_rowcol(cell_lat, cell_lon)
    return row_c - patch_px // 2, col_c - patch_px // 2


def modis_patch_corner_for_cell(
    cell_lat: float, cell_lon: float, patch_px: int = 512
) -> Tuple[int, int]:
    """
    Return the top-left (row0, col0) in global sinusoidal coords.

    The centre pixel is computed with round(), not int().
    """
    row_c, col_c = latlon_to_modis_global_rowcol(cell_lat, cell_lon)
    return row_c - patch_px // 2, col_c - patch_px // 2
