"""
ndvi.py
=======
NDVI computation from MajorTOM-structured Zarr stores (FCI and MODIS).

All public functions accept an open zarr group as returned by
``zarr.open(path, mode='r')`` and return plain dicts
``{grid_id: float}`` or ``{grid_id: ndarray}`` that are easy to
compare, serialise, or hand off to the viz module.

Band conventions
----------------
FCI NDVI   : ``(vis_08 − vis_06) / (vis_08 + vis_06)``  (NIR 865 nm, Red 600 nm)
MODIS NDVI : ``(sur_refl_b02 − sur_refl_b01) / (sur_refl_b02 + sur_refl_b01)``
             (NIR 841–876 nm, Red 620–670 nm)
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from eumetsearch.utils import compute_ndvi  # noqa: F401 – re-exported for convenience

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FCI helpers
# ---------------------------------------------------------------------------

def fci_ndvi_one_timestamp(
    fci_store,
    timestamp: str,
) -> dict[str, np.ndarray]:
    """Return ``{grid_id: ndvi_patch (H, W)}`` for every cell at one FCI timestamp."""
    red  = fci_store["patches"]["vis_06"][timestamp]["data"][:].astype(np.float32)
    nir  = fci_store["patches"]["vis_08"][timestamp]["data"][:].astype(np.float32)
    gids = fci_store["patches"]["vis_06"][timestamp]["grid_ids"][:]

    ndvi = compute_ndvi(red, nir)
    return {str(g): ndvi[i] for i, g in enumerate(gids)}


def fci_ndvi_daily_max(
    fci_store,
    date: str,
) -> dict[str, float]:
    """Per-cell daily-max NDVI from all FCI timestamps matching *date*.

    Each patch is summarised by its spatial median (robust to cloud edges),
    then the daily max is taken across all matching timestamps.

    Parameters
    ----------
    fci_store : zarr group
        Opened zarr store with structure ``patches/vis_06/<timestamp>/…``.
    date : str
        ISO date prefix, e.g. ``"2025-06-01"``.

    Returns
    -------
    dict[str, float]
        ``{grid_id: daily_max_ndvi}``.  Cells with no finite value are excluded.
    """
    timestamps = [t for t in sorted(fci_store["patches"]["vis_06"].keys())
                  if t.startswith(date)]
    logger.debug("FCI daily max: %d timestamps for %s: %s", len(timestamps), date, timestamps)

    per_ts: list[dict[str, float]] = []
    for ts in timestamps:
        patch_ndvi = fci_ndvi_one_timestamp(fci_store, ts)
        per_ts.append({gid: float(np.nanmedian(arr)) for gid, arr in patch_ndvi.items()})

    all_gids = set().union(*[d.keys() for d in per_ts])
    daily_max: dict[str, float] = {}
    for gid in all_gids:
        vals = [d[gid] for d in per_ts if gid in d]
        v = float(np.nanmax(vals)) if vals else np.nan
        if np.isfinite(v):
            daily_max[gid] = v
    return daily_max


def _build_fci_index(fci_store, date: str) -> dict[str, dict[str, int]]:
    """Build ``{grid_id: {timestamp: row_index}}`` for fast patch lookup."""
    timestamps = sorted(
        t for t in fci_store["patches"]["vis_06"].keys() if t.startswith(date)
    )
    index: dict[str, dict[str, int]] = {}
    for ts in timestamps:
        gids = fci_store["patches"]["vis_06"][ts]["grid_ids"][:]
        for i, g in enumerate(gids):
            index.setdefault(str(g), {})[ts] = i
    return index


def _fci_ndvi_scalars_fast(
    fci_store,
    date: str,
    gids_filter: Optional[set] = None,
) -> dict[str, tuple[float, str, int]]:
    """Vectorised FCI NDVI, returning ``{gid: (best_median, best_ts, best_row)}``.

    Reads all patches per timestamp in one batch; much faster than looping
    over individual cells when processing many tiles at once.
    """
    timestamps = sorted(
        t for t in fci_store["patches"]["vis_06"].keys() if t.startswith(date)
    )
    gids_filter_set = set(gids_filter) if gids_filter is not None else None
    best: dict[str, tuple[float, str, int]] = {}

    for ts in timestamps:
        red_all = fci_store["patches"]["vis_06"][ts]["data"][:].astype(np.float32)
        nir_all = fci_store["patches"]["vis_08"][ts]["data"][:].astype(np.float32)
        gids_all = [str(g) for g in fci_store["patches"]["vis_06"][ts]["grid_ids"][:]]
        ndvi_all = compute_ndvi(red_all, nir_all)
        medians = np.nanmedian(ndvi_all.reshape(len(gids_all), -1), axis=1)

        for i, (gid, med) in enumerate(zip(gids_all, medians)):
            if gids_filter_set is not None and gid not in gids_filter_set:
                continue
            med = float(med)
            if gid not in best or med > best[gid][0]:
                best[gid] = (med, ts, i)

    return best


# ---------------------------------------------------------------------------
# MODIS helpers
# ---------------------------------------------------------------------------

def modis_ndvi_one_date(
    modis_store,
    date: str,
) -> dict[str, float]:
    """Return ``{grid_id: median_ndvi}`` for all cells in MOD09GQ on *date*.

    ``sur_refl_b01`` = Red (620–670 nm), ``sur_refl_b02`` = NIR (841–876 nm).
    Values are expected in reflectance [0, 1].
    Cells where the entire patch is NaN/invalid are excluded.
    """
    b01_group = modis_store["patches"]["sur_refl_b01"][date]
    b02_group = modis_store["patches"]["sur_refl_b02"][date]
    gids = sorted(b01_group.keys())

    result: dict[str, float] = {}
    for gid in gids:
        red = b01_group[gid][:].astype(np.float32)
        nir = b02_group[gid][:].astype(np.float32)
        ndvi = np.where((nir + red) > 0.01, compute_ndvi(red, nir), np.nan)
        val = float(np.nanmedian(ndvi))
        if np.isfinite(val):
            result[gid] = val
    return result


def _modis_load_parallel(
    modis_store,
    date: str,
    gids: list[str],
    max_workers: int = 8,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load ``(red, nir)`` arrays for multiple MODIS cells concurrently.

    Returns
    -------
    dict[str, (np.ndarray, np.ndarray)]
        ``{grid_id: (red_patch, nir_patch)}``.
    """
    b01 = modis_store["patches"]["sur_refl_b01"][date]
    b02 = modis_store["patches"]["sur_refl_b02"][date]

    def _one(gid: str):
        return gid, (b01[gid][:].astype(np.float32), b02[gid][:].astype(np.float32))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return dict(ex.map(_one, gids))
