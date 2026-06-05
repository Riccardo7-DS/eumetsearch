"""
viz.py
======
Visualization utilities for comparing FCI and MODIS MajorTOM patches.

Requires matplotlib; all other dependencies (numpy, zarr) are assumed to be
available.  The module is safe to import without matplotlib — functions will
raise ``ImportError`` only when called.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from eumetsearch.transform import (
    align_patches_multiband,
    grid_id_to_latlon,
    grid_ids_to_latlons,
    modis_patch_corner_for_cell,
)
from eumetsearch.utils import compute_ndvi
from eumetsearch.transform.ndvi import (
    _build_fci_index,
    _fci_ndvi_scalars_fast,
    _modis_load_parallel,
)


# ---------------------------------------------------------------------------
# Map-level plots
# ---------------------------------------------------------------------------

def plot_ndvi_map(
    ndvi_dict: dict[str, float],
    title: str,
    ax=None,
    vmin: float = -0.2,
    vmax: float = 0.8,
    cmap: str = "RdYlGn",
    s: int = 8,
):
    """Scatter-plot NDVI values as coloured dots on a lat/lon axes.

    Parameters
    ----------
    ndvi_dict : dict[str, float]
        ``{grid_id: ndvi_value}`` as returned by :func:`fci_ndvi_daily_max`
        or :func:`modis_ndvi_one_date`.
    title : str
        Axes title.
    ax : matplotlib Axes, optional
        If *None*, a new figure is created.
    """
    import matplotlib.pyplot as plt

    gids = sorted(ndvi_dict.keys())
    vals = np.array([ndvi_dict[g] for g in gids])
    lats, lons = grid_ids_to_latlons(gids)

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    sc = ax.scatter(lons, lats, c=vals, cmap=cmap, vmin=vmin, vmax=vmax,
                    s=s, linewidths=0)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, label="NDVI")
    return ax


def compare_ndvi_scatter(
    fci_ndvi: dict[str, float],
    modis_ndvi: dict[str, float],
) -> None:
    """Scatter FCI daily-max NDVI vs MODIS NDVI for overlapping MajorTOM cells.

    Prints Pearson r, mean bias, and RMSE.  Does nothing if there are no
    overlapping cells.
    """
    import matplotlib.pyplot as plt

    common = sorted(set(fci_ndvi) & set(modis_ndvi))
    if not common:
        print("No overlapping grid cells — nothing to compare.")
        return

    fci_vals   = np.array([fci_ndvi[g]   for g in common], dtype=np.float32)
    modis_vals = np.array([modis_ndvi[g] for g in common], dtype=np.float32)

    valid = np.isfinite(fci_vals) & np.isfinite(modis_vals)
    n_dropped = (~valid).sum()
    if n_dropped:
        print(f"Dropped {n_dropped} cells with NaN/Inf ({valid.sum()} remaining)")
    fci_vals, modis_vals = fci_vals[valid], modis_vals[valid]

    if len(fci_vals) == 0:
        print("No valid (non-NaN) overlapping cells to plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(modis_vals, fci_vals, s=15, alpha=0.6)
    lim = (-0.2, 1.0)
    ax.plot(lim, lim, "k--", lw=1, label="1:1")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("MODIS NDVI (MOD09GQ)")
    ax.set_ylabel("FCI daily-max NDVI")
    ax.set_title(f"NDVI correspondence — {len(fci_vals)} MajorTOM tiles")
    ax.legend()

    r    = np.corrcoef(modis_vals, fci_vals)[0, 1]
    bias = np.mean(fci_vals - modis_vals)
    rmse = np.sqrt(np.mean((fci_vals - modis_vals) ** 2))
    print(f"Pearson r = {r:.3f},  mean bias FCI−MODIS = {bias:+.3f},  RMSE = {rmse:.3f}")
    plt.show()


# ---------------------------------------------------------------------------
# Tile-level comparison
# ---------------------------------------------------------------------------

def compare_tiles(
    fci_store,
    modis_store,
    date: str = "2025-06-01",
    n: int = 6,
    sort_by: str = "ndvi_diff",
    gid_list: Optional[list[str]] = None,
    target_n_pixels: int = 256,
) -> None:
    """Show *n* overlapping MajorTOM tiles side-by-side using AEQD reprojection.

    Both sensors are independently reprojected onto a common Earth-fixed grid
    centred on each MajorTOM cell.  FCI patch origins are read directly from
    the Zarr (exact corners stored at download time); MODIS corners are
    computed deterministically from the cell lat/lon.

    Parameters
    ----------
    fci_store : zarr group
        Opened FCI MajorTOM zarr store.
    modis_store : zarr group
        Opened MODIS MajorTOM zarr store.
    date : str
        ISO date string, e.g. ``"2025-06-01"``.
    n : int
        Number of tiles to show (ignored when *gid_list* is given).
    sort_by : str
        Tile selection strategy:

        * ``"ndvi_diff"`` — largest |FCI − MODIS| NDVI first (default)
        * ``"ndvi"``      — highest MODIS NDVI first
        * ``"random"``    — random sample (seed 42)
        * ``"first"``     — alphabetical order
    gid_list : list[str], optional
        Show exactly these grid IDs (overrides *sort_by* and *n*).
    target_n_pixels : int
        Output grid size in pixels (same for both sensors).
    """
    import matplotlib.pyplot as plt
    from IPython.display import display

    fci_dates = [t for t in fci_store["patches"]["vis_06"].keys() if t.startswith(date)]
    modis_date_gids = (
        set(modis_store["patches"]["sur_refl_b01"][date].keys())
        if date in modis_store["patches"]["sur_refl_b01"]
        else set()
    )

    all_fci_gids: set[str] = set()
    for ts in fci_dates:
        all_fci_gids.update(str(g) for g in fci_store["patches"]["vis_06"][ts]["grid_ids"][:])

    common = sorted(all_fci_gids & modis_date_gids)
    if not common:
        print(f"No overlapping cells for date {date}.")
        return

    fci_info: dict   = {}
    modis_cache: dict = {}

    if gid_list is not None:
        tiles = [g for g in gid_list if g in set(common)]
    elif sort_by == "random":
        rng = np.random.default_rng(42)
        tiles = rng.choice(common, size=min(n, len(common)), replace=False).tolist()
    elif sort_by == "ndvi":
        print(f"Loading MODIS for {len(common)} cells (parallel)…")
        modis_cache = _modis_load_parallel(modis_store, date, common)
        mod_sc: dict[str, float] = {}
        for g, (r, ni) in modis_cache.items():
            ndvi = np.where((r + ni) > 0.01, (ni - r) / (ni + r), np.nan)
            mod_sc[g] = float(np.nanmedian(ndvi))
        tiles = sorted(mod_sc, key=mod_sc.get, reverse=True)[:n]
    elif sort_by == "ndvi_diff":
        print(f"Ranking {len(common)} overlapping cells by |FCI − MODIS| NDVI…")
        fci_info    = _fci_ndvi_scalars_fast(fci_store, date, gids_filter=set(common))
        modis_cache = _modis_load_parallel(modis_store, date, common)
        fci_sc  = {g: fci_info[g][0] for g in common if g in fci_info}
        mod_sc = {}
        for g, (r, ni) in modis_cache.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ndvi = np.where((r + ni) > 0.01, (ni - r) / (ni + r), np.nan)
            mod_sc[g] = float(np.nanmedian(ndvi))
        valid_tiles = [g for g in common if g in fci_sc and g in mod_sc]
        tiles = sorted(valid_tiles, key=lambda g: abs(fci_sc[g] - mod_sc[g]), reverse=True)[:n]
    else:
        tiles = common[:n]

    fci_index = _build_fci_index(fci_store, date)

    COLS = [
        "FCI vis_06\n(Red)", "FCI vis_08\n(NIR)", "FCI NDVI",
        "MODIS b01\n(Red)", "MODIS b02\n(NIR)", "MODIS NDVI",
    ]
    fig, axes = plt.subplots(len(tiles), 6, figsize=(18, 3 * len(tiles)), squeeze=False)
    ndvi_kw = dict(cmap="RdYlGn", vmin=-0.1, vmax=0.8)
    band_kw = dict(cmap="gray")

    for row_idx, gid in enumerate(tiles):
        lat, lon = grid_id_to_latlon(gid)

        # Best FCI timestamp for this cell
        if gid in fci_info:
            _, best_ts, best_row = fci_info[gid]
        else:
            best_med, best_ts, best_row = -np.inf, None, None
            for ts, idx in fci_index[gid].items():
                red = fci_store["patches"]["vis_06"][ts]["data"][idx].astype(np.float32)
                nir = fci_store["patches"]["vis_08"][ts]["data"][idx].astype(np.float32)
                med = float(np.nanmedian(compute_ndvi(red, nir)))
                if med > best_med:
                    best_med, best_ts, best_row = med, ts, idx

        f_red    = fci_store["patches"]["vis_06"][best_ts]["data"][best_row].astype(np.float32)
        f_nir    = fci_store["patches"]["vis_08"][best_ts]["data"][best_row].astype(np.float32)
        fci_row0, fci_col0 = fci_store["patches"]["vis_06"][best_ts]["patch_origins"][best_row]

        if gid in modis_cache:
            m_red, m_nir = modis_cache[gid]
        else:
            m_red = modis_store["patches"]["sur_refl_b01"][date][gid][:].astype(np.float32)
            m_nir = modis_store["patches"]["sur_refl_b02"][date][gid][:].astype(np.float32)
        mod_row0, mod_col0 = modis_patch_corner_for_cell(lat, lon, patch_px=512)

        result = align_patches_multiband(
            fci_bands={"red": f_red, "nir": f_nir},
            fci_patch_row0=int(fci_row0), fci_patch_col0=int(fci_col0),
            modis_bands={"red": m_red, "nir": m_nir},
            modis_patch_row0=mod_row0, modis_patch_col0=mod_col0,
            cell_lat=lat, cell_lon=lon,
            target_n_pixels=target_n_pixels,
        )

        fa_red = result["fci"]["red"]
        fa_nir = result["fci"]["nir"]
        ma_red = result["modis"]["red"]
        ma_nir = result["modis"]["nir"]

        fa_ndvi = compute_ndvi(fa_red, fa_nir)
        ma_ndvi = np.where((ma_red + ma_nir) > 0.01, compute_ndvi(ma_red, ma_nir), np.nan)

        def _norm(a: np.ndarray) -> np.ndarray:
            lo, hi = np.nanpercentile(a, 2), np.nanpercentile(a, 98)
            return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)

        panels = [_norm(fa_red), _norm(fa_nir), fa_ndvi,
                  _norm(ma_red), _norm(ma_nir), ma_ndvi]
        kwlist  = [band_kw, band_kw, ndvi_kw, band_kw, band_kw, ndvi_kw]

        for col, (panel, kw) in enumerate(zip(panels, kwlist)):
            ax = axes[row_idx, col]
            ax.imshow(panel, **kw, interpolation="nearest")
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(COLS[col], fontsize=9)

        axes[row_idx, 0].set_ylabel(
            f"{gid}\n({lat:.1f}°, {lon:.1f}°)\n{best_ts[11:16]} UTC",
            fontsize=8, rotation=0, labelpad=80, va="center",
        )

    fig.suptitle(
        f"MajorTOM tile comparison  —  FCI vs MODIS (AEQD reprojection)  —  {date}",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    display(fig)
    plt.close(fig)
    print(f"Showed {len(tiles)} tiles  ({len(common)} overlapping cells available)")
