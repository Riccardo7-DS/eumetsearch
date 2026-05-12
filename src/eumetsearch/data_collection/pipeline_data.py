"""
Tiled download pipeline for EUMETSAT FCI / MSG SEVIRI data.

Divides an area of interest into n_lat × n_lon tiles, optionally filters
tiles by minimum land fraction (via Google Earth Engine), then runs
EUMDownloader + ZarrExport for each tile.

Usage
-----
    python -m eumetsearch.data_collection.pipeline_data \\
        --product MTG-FCI-L1C-FDHSI \\
        --channels vis_06 vis_08 \\
        --region europe \\
        --n_lat 4 --n_lon 4 \\
        --start_date 2025-06-26T08:00:00 \\
        --end_date   2025-06-26T13:00:00 \\
        --observations_per_day 6 \\
        --jump_minutes 60 \\
        --majortom

Tile bboxes that pass the land-fraction filter are cached in
<DATA_PATH>/water_min.npy so subsequent runs skip the Earth Engine call.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

sys.dont_write_bytecode = True
load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
from eumetsearch import init_logging  # noqa: E402

logger = init_logging(log_file="pipeline_data.log", verbose=False)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
from eumetsearch import BBOX_REGISTRY, products_list  # noqa: E402
from eumetsearch.utils.bbox import generate_bboxes_fixed, get_bbox  # noqa: E402

parser = argparse.ArgumentParser(description="Tiled EUMETSAT data download pipeline")

parser.add_argument(
    "--product",
    type=str,
    default=os.getenv("product", "MTG-FCI-L1C-FDHSI"),
    choices=list(products_list.keys()),
    help="EUMETSAT product key (from products_list)",
)
parser.add_argument(
    "--channels",
    nargs="+",
    default=os.getenv("channels", "vis_06 vis_08").split(),
    help="FCI / SEVIRI channel names to export",
)
parser.add_argument(
    "--region",
    default=os.getenv("EUMETS_REGION", "mtg"),
    choices=sorted(BBOX_REGISTRY),
    help="Named AOI region from BBOX_REGISTRY (default: mtg full disk)",
)
# Custom bbox overrides --region when all four are provided
parser.add_argument("--lon_min", type=float, default=None)
parser.add_argument("--lat_min", type=float, default=None)
parser.add_argument("--lon_max", type=float, default=None)
parser.add_argument("--lat_max", type=float, default=None)

parser.add_argument("--n_lat", type=int, default=int(os.getenv("n_lat", 4)), help="Tile rows")
parser.add_argument("--n_lon", type=int, default=int(os.getenv("n_lon", 4)), help="Tile columns")

parser.add_argument("--start_date", type=str, default=os.getenv("start_date", "2025-06-26T08:00:00"))
parser.add_argument("--end_date",   type=str, default=os.getenv("end_date",   "2025-06-26T13:00:00"))
parser.add_argument(
    "--observations_per_day",
    type=int,
    default=int(os.getenv("observations_per_day", 6)),
)
parser.add_argument("--jump_minutes", type=int, default=int(os.getenv("jump_minutes", 60)))
parser.add_argument("--batch_days",   type=int, default=int(os.getenv("batch_days", 1)))

parser.add_argument("--resampler", default=os.getenv("resampler", "nearest"))
parser.add_argument(
    "--area_reprojection",
    default=os.getenv("area_reprojection", "mtg_fci_latlon_1km"),
    help="pyresample area name for regular-grid export (ignored when --majortom)",
)
parser.add_argument(
    "--majortom",
    action="store_true",
    default=False,
    help="Export to MajorTOM sparse Zarr instead of regular lat/lon grid",
)
parser.add_argument(
    "--also_majortom",
    action="store_true",
    default=False,
    help="Export both regular grid Zarr AND MajorTOM sparse Zarr (single download)",
)
parser.add_argument(
    "--majortom_patch_size",
    type=int,
    default=int(os.getenv("majortom_patch_size", 64)),
)
parser.add_argument("--majortom_grid_path", type=str, default=None,
                    help="Path to external MajorTOM grid file. When omitted a regular grid is generated.")
parser.add_argument(
    "--majortom_spacing_km",
    type=float,
    default=float(os.getenv("majortom_spacing_km", 100.0)),
    help="Cell spacing in km for the generated regular grid (default 100). Ignored when --majortom_grid_path is set.",
)

parser.add_argument(
    "--min_land_fraction",
    type=float,
    default=float(os.getenv("min_land_fraction", 0.0)),
    help="Minimum land fraction (0–100) for tile filtering via GEE. 0 = keep all tiles.",
)
parser.add_argument(
    "--water_min_path",
    type=str,
    default=None,
    help="Path to water_min.npy cache file. Defaults to <DATA_PATH>/water_min.npy.",
)

parser.add_argument("-y", "--yes",    action="store_true", help="Auto-confirm zarr deletion")
parser.add_argument("-r", "--remove", action="store_true", help="Delete source files after processing")
parser.add_argument("-t", "--threading", action="store_false", help="Enable threading for I/O")
parser.add_argument("--dask_threads", type=int, default=int(os.getenv("dask_threads", 0)))
parser.add_argument(
    "--max_parallel_conns",
    type=int,
    default=int(os.getenv("max_parallel_conns", 10)),
)
parser.add_argument(
    "--dtype",
    type=str,
    default=os.getenv("dtype", "float32"),
    choices=["float16", "float32", "float64"],
    help="NumPy dtype for patch arrays written to MajorTOM Zarr (default: float32)",
)
parser.add_argument(
    "-d", "--delete_temp",
    action="store_true",
    default=False,
    help="Delete temporary download folders after each tile",
)
parser.add_argument(
    "--consolidated_label",
    type=str,
    default=None,
    help=(
        "Stem for the merged output zarr, e.g. 'mtg_june_2025'. "
        "Written to <DATA_PATH>/datastore_data/<label>_majortom.zarr. "
        "If omitted, no consolidation is performed."
    ),
)

args = parser.parse_args()

# ---------------------------------------------------------------------------
# Dask scheduler
# ---------------------------------------------------------------------------
import dask  # noqa: E402

if args.dask_threads > 0:
    dask.config.set(scheduler="threads", num_workers=args.dask_threads)
else:
    dask.config.set(scheduler="single-threaded")

# ---------------------------------------------------------------------------
# Resolve AOI bbox
# ---------------------------------------------------------------------------
_custom = [args.lon_min, args.lat_min, args.lon_max, args.lat_max]
if all(v is not None for v in _custom):
    aoi_bbox = (args.lon_min, args.lat_min, args.lon_max, args.lat_max)
    logger.info(f"Using custom AOI bbox (W,S,E,N): {aoi_bbox}")
else:
    aoi_bbox = get_bbox(args.region)
    logger.info(f"Using named region '{args.region}' bbox (W,S,E,N): {aoi_bbox}")

# ---------------------------------------------------------------------------
# Generate tiles
# ---------------------------------------------------------------------------
all_tiles = generate_bboxes_fixed(aoi_bbox, n_lat=args.n_lat, n_lon=args.n_lon)
logger.info(f"Generated {len(all_tiles)} tiles ({args.n_lat} × {args.n_lon})")

# ---------------------------------------------------------------------------
# Land-fraction filter (with water_min.npy cache)
# ---------------------------------------------------------------------------
from definitions import DATA_PATH  # noqa: E402

_water_min_path = Path(args.water_min_path) if args.water_min_path else Path(DATA_PATH) / "water_min.npy"

if args.min_land_fraction > 0:
    if _water_min_path.exists():
        logger.info(f"Loading precomputed filtered tiles from {_water_min_path}")
        tiles = np.load(_water_min_path, allow_pickle=True).tolist()
    else:
        logger.info(
            f"Filtering {len(all_tiles)} tiles by min land fraction "
            f">= {args.min_land_fraction}% (using Google Earth Engine)"
        )
        try:
            import ee
            ee.Authenticate()
            ee.Initialize(project=os.environ["EE_PROJECT"])

            def _tile_has_min_land(bbox: tuple, min_pct: float) -> bool:
                W, S, E, N = bbox
                region = ee.Geometry.Rectangle([W, S, E, N])
                land_mask = ee.Image("MODIS/006/MOD44W/2015_01_01").select("water_mask")
                land_binary = land_mask.eq(0).toFloat()
                stats = land_binary.reduceRegion(
                    reducer=ee.Reducer.sum().combine(
                        reducer2=ee.Reducer.count(), sharedInputs=True
                    ),
                    geometry=region,
                    scale=500,
                    maxPixels=1e9,
                ).getInfo()
                if not stats:
                    return False
                land_sum = stats.get("water_mask_sum") or stats.get("sum") or stats.get("constant_sum")
                total = stats.get("water_mask_count") or stats.get("count") or stats.get("constant_count")
                if land_sum is None or not total:
                    return False
                return (land_sum / total) * 100 >= min_pct

            tiles = [
                tile
                for tile in tqdm(all_tiles, desc="Filtering tiles by land fraction")
                if _tile_has_min_land(tile, args.min_land_fraction)
            ]
        except Exception as e:
            logger.error(f"GEE land-fraction filter failed: {e}. Using all tiles.")
            tiles = all_tiles

        np.save(_water_min_path, np.array(tiles, dtype=object))
        logger.info(f"Saved filtered tiles to {_water_min_path}")
else:
    tiles = all_tiles

logger.info(f"Processing {len(tiles)} tiles out of {len(all_tiles)} total")

# ---------------------------------------------------------------------------
# Main download loop
# ---------------------------------------------------------------------------
from eumetsearch import EUMDownloader, ZarrExport  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402

product_id = products_list[args.product]["product_id"]
start_dt = datetime.fromisoformat(args.start_date)
end_dt = datetime.fromisoformat(args.end_date)
datastore_dir = Path(DATA_PATH) / "datastore_data"

# Build the list of time batches upfront so every tile iterates the same windows.
_batch_end = start_dt
_time_batches: list[tuple[datetime, datetime]] = []
while _batch_end <= end_dt:
    _batch_start = _batch_end
    _batch_end = min(
        _batch_start + timedelta(days=args.batch_days),
        end_dt + timedelta(seconds=1),  # inclusive end
    )
    _time_batches.append((_batch_start, _batch_end - timedelta(seconds=1)))
    _batch_end = _batch_end  # already advanced

logger.info(
    f"Date range split into {len(_time_batches)} batch(es) of up to {args.batch_days} day(s)"
)

# Resolve the single output zarr path upfront so all tiles write into it directly.
# For MajorTOM the sparse structure (patches/channel/timestamp/grid_id) is naturally
# append-friendly: require_group/require_dataset are idempotent, no axes to pre-allocate,
# and no spatial chunks are shared between tiles.
use_majortom = args.majortom or args.also_majortom
if use_majortom and args.consolidated_label:
    suffix = "_majortom" if use_majortom else ""
    consolidated_zarr_path = str(datastore_dir / f"{args.consolidated_label}{suffix}.zarr")
    logger.info(f"All tiles will write directly into: {consolidated_zarr_path}")
else:
    consolidated_zarr_path = None

for i, tile_bbox in tqdm(enumerate(tiles), desc="Processing tiles", total=len(tiles)):
    W, S, E, N = tile_bbox
    NSWE = [N, S, W, E]
    tile_label = f"tile_{i:04d}_W{W:.2f}_S{S:.2f}_E{E:.2f}_N{N:.2f}"
    logger.info(f"Tile {i+1}/{len(tiles)}: (W={W:.3f}, S={S:.3f}, E={E:.3f}, N={N:.3f})")

    for b, (batch_start, batch_end) in enumerate(
        tqdm(_time_batches, desc=f"  Tile {i+1} batches", leave=False)
    ):
        batch_start_iso = batch_start.isoformat()
        batch_end_iso = batch_end.isoformat()
        logger.info(
            f"  Batch {b+1}/{len(_time_batches)}: {batch_start.date()} → {batch_end.date()}"
        )

        downloader = EUMDownloader(
            product_id=product_id,
            output_dir=str(datastore_dir),
            max_parallel_conns=args.max_parallel_conns,
        )

        try:
            downloader.download_interval(
                start_time=batch_start_iso,
                end_time=batch_end_iso,
                bounding_box=NSWE,
                observations_per_day=args.observations_per_day,
                jump_minutes=args.jump_minutes,
                start_hour=start_dt.hour,
            )

            ZarrExport(
                args=args,
                downloader=downloader,
                label=tile_label,
                channels=args.channels,
                area_reprojection=None if args.majortom else args.area_reprojection,
                reprojection=args.resampler,
                majortom=args.majortom,
                also_majortom=args.also_majortom,
                majortom_patch_size=args.majortom_patch_size,
                majortom_grid_path=args.majortom_grid_path,
                majortom_spacing_km=args.majortom_spacing_km,
                bbox_filter=(S, N, W, E),  # (lat_min, lat_max, lon_min, lon_max)
                dtype=args.dtype,
                zarr_path=consolidated_zarr_path,  # None → per-tile file
            )

            if args.delete_temp:
                nat_path = datastore_dir / "disk" / "Data" / "natfolder"
                zip_path = datastore_dir / "disk" / "Data" / "zipfolder"
                for folder in (nat_path, zip_path):
                    if folder.exists():
                        shutil.rmtree(folder)

        except Exception as e:
            logger.error(
                f"Error on tile {i} batch {b} ({batch_start.date()}→{batch_end.date()}): {e}"
            )
            import traceback
            logger.error(traceback.format_exc())
            continue  # skip to next batch, not next tile

if consolidated_zarr_path:
    import zarr as _zarr
    _zarr.consolidate_metadata(consolidated_zarr_path)
    logger.info(f"Pipeline complete. Output: {consolidated_zarr_path}")
