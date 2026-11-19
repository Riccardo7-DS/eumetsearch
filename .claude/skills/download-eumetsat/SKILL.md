---
name: download-eumetsat
description: Downloading eumetsat data with eumetsearch. Use this skill when the user wants to download EUMETSAT satellite data (MTG FCI L1C/L2, MSG SEVIRI), search the EUMETSAT DataStore, export to Zarr, or use MajorTOM tiling. Triggers on phrases like "download FCI data", "get MTG data", "fetch MSG SEVIRI", "download satellite data from EUMETSAT", "export to zarr from eumetsat", "MajorTOM patches from FCI".
---

# Downloading EUMETSAT Data with eumetsearch

The `eumetsearch` package downloads EUMETSAT satellite data via the `eumdac` API (DataStore or DataTailor) and exports it to Zarr, either as a regular lat/lon grid or as a MajorTOM sparse patch store.

## Setup

Required environment variables (in `.env` at the repo root, see `.env.example`):
```
EUMETSAT_CONSUMER_KEY=<your_key>
EUMETSAT_CONSUMER_SECRET=<your_secret>
```

Get credentials at: https://data.eumetsat.int → Login → API Key. Never print or commit these values.

Install: `pip install -e .` (full) or `pip install -e ".[transform]"` (analysis/alignment only, no download deps — `EUMDownloader`/`ZarrExport` are then unavailable).

## Supported Products

All entries come from the `products_list` dict in `src/eumetsearch/data_collection/downloader.py`. Each entry has `product_id`, `product_name`, `satpy_reader`, `bands`, `reflectance_bands`, `bt_bands` (and `datatailor_bands` for SEVIRI).

| Key | Product ID | Description | Satpy reader | Bands |
|-----|-----------|-------------|-------------|-------|
| `MTG-FCI-L1C-FDHSI` | `EO:EUM:DAT:0662` | FCI L1C Full Disk HSR | `fci_l1c_nc` | `FCI_CHANNELS_ALL` (16) |
| `MTG-FCI-L1C-HRFI` | `EO:EUM:DAT:0665` | FCI L1C High-Res Full Disk | `fci_l1c_nc` | `vis_06_hr`, `vis_08_hr` |
| `MTG-FCI-L2-CLM` | `EO:EUM:DAT:0678` | FCI L2 Cloud Mask | `fci_l2_nc` | `cloud_state` |
| `MTG-FCI-L2-AFM` | `EO:EUM:DAT:0801` | FCI L2 Active Fire Monitoring | `fci_l2_nc` | `fire_probability` |
| `MTG-FCI-L2-OCA` | `EO:EUM:DAT:0668` | FCI L2 Optimal Cloud Analysis | `fci_l2_nc` | `retrieved_cloud_optical_thickness`, `retrieved_cloud_top_pressure_upper_layer`, `retrieved_cloud_top_pressure_lower_layer` |
| `MTG-FCI-L2-GII` | `EO:EUM:DAT:0683` | FCI L2 Global Instability Indices | `fci_l2_nc` | `FCI_GII_BANDS` |
| `MTG-FCI-L2-AMV` | `EO:EUM:DAT:0676` | FCI L2 Atmospheric Motion Vectors | `fci_l2_nc` | `FCI_AMV_BANDS` |
| `MSG-SEVIRI-L1-HRY` | `EO:EUM:DAT:MSG:HRSEVIRI` | SEVIRI L1.5 Prime (0°E) | `seviri_l1b_hrit` | `SEVIRI_CHANNELS_HRV` |
| `MSG-SEVIRI-L1-IODC` | `EO:EUM:DAT:MSG:HRSEVIRI-IODC` | SEVIRI L1.5 IODC (41.5°E) | `seviri_l1b_hrit` | `SEVIRI_CHANNELS_HRV` |
| `MSG-SEVIRI-RSS` | `EO:EUM:DAT:MSG:MSG15-RSS` | SEVIRI Rapid Scan (3.5°E) | `seviri_l1b_hrit` | `SEVIRI_CHANNELS_ALL` (no HRV) |

Lookup by code instead of key: `EUMDownloader.get_product_metadata("EO:EUM:DAT:0668")` (static, no auth needed). OCA has no unlayered `cloud_top_pressure` — use the per-layer variants.

## Channel / Band Constants

All importable from `eumetsearch`:

```python
# FCI L1C
FCI_CHANNELS_VIS_NIR = ["vis_04","vis_05","vis_06","vis_08","vis_09","nir_13","nir_16","nir_22"]
FCI_CHANNELS_IR      = ["ir_38","wv_63","wv_73","ir_87","ir_97","ir_105","ir_123","ir_133"]
FCI_CHANNELS_ALL     = FCI_CHANNELS_VIS_NIR + FCI_CHANNELS_IR
FCI_HR_CHANNELS      = ["vis_06_hr", "vis_08_hr"]          # HRFI only

# FCI L2
FCI_GII_BANDS = ["k_index","lifted_index","prec_water_high","prec_water_low","prec_water_mid","prec_water_total"]
FCI_AMV_BANDS = ["speed","direction","pressure","temperature","wind_method","qi"]

# MSG SEVIRI (satpy names)
SEVIRI_CHANNELS_VIS_NIR = ["VIS006","VIS008","IR_016"]
SEVIRI_CHANNELS_IR      = ["IR_039","WV_062","WV_073","IR_087","IR_097","IR_108","IR_120","IR_134"]
SEVIRI_CHANNELS_ALL     = SEVIRI_CHANNELS_VIS_NIR + SEVIRI_CHANNELS_IR
SEVIRI_CHANNELS_HRV     = SEVIRI_CHANNELS_ALL + ["HRV"]
```

DataTailor does not accept satpy SEVIRI names; `SEVIRI_CHANNEL_TO_DATATAILOR_ID` maps them to `channel_1`…`channel_11` / `hrv`. The downloader uses `datatailor_bands` automatically — always pass satpy names to `ZarrExport`.

`ZarrExport` asserts every requested channel is in the allowed set (all of the above + CLM/AFM/OCA bands).

## Bounding Boxes

`BBOX_REGISTRY` (in `src/eumetsearch/utils/bbox.py`) stores named regions in **(W, S, E, N)**: `mtg`, `hoa`, `europe`, `africa`, `mideast`, `brazil`. eumdac expects **[N, S, W, E]** — convert at the call site:

```python
from eumetsearch import get_bbox
W, S, E, N = get_bbox("europe")
NSWE = [N, S, W, E]
```

`bbox_mtg()` and `hoa_bbox()` remain for backwards compatibility. `generate_bboxes_fixed` / `generate_bboxes_from_resolution` split an AOI into tiles.

## EUMDownloader

```python
from eumetsearch import EUMDownloader, products_list

downloader = EUMDownloader(
    product_id=products_list["MTG-FCI-L1C-FDHSI"]["product_id"],
    output_dir="./data/datastore_data",
    format="netcdf4",            # or "geotiff" (DataTailor)
    max_parallel_conns=10,
)
```

Methods:
- `search(start_time, end_time, bounding_box=NSWE)` → `{"count", "products": [{"identifier","date"}]}`; no download.
- `download_interval(start_time, end_time, bounding_box, method=None, observations_per_day=1, start_hour=12, interval_minutes=10, jump_minutes=0)` → **only plans** the time windows (`self.intervals`); nothing is downloaded yet. Files are fetched later by `initiate_download()`, which `ZarrExport` / `MajorTomZarrExport` call for you.
- `download_single(start_time, end_time, bounding_box, method="datatailor")` → one-shot download of a single window (event-based), returns `file_list`.
- `initiate_download(aggregated_file=None)` → collects products, skips timestamps already in `aggregated_file`, and downloads when `method` is `"datatailor"` (customised, reprojected to geographic + band filter) or `"datastore"` (raw).

Times are `'%Y-%m-%dT%H:%M:%S'` strings, interpreted as UTC.

## Usage — Regular Grid (lat/lon Zarr)

```python
from eumetsearch import EUMDownloader, ZarrExport, products_list, get_bbox

downloader = EUMDownloader(product_id=products_list["MTG-FCI-L1C-FDHSI"]["product_id"],
                           output_dir="./data/datastore_data")
W, S, E, N = get_bbox("mtg")
downloader.download_interval(
    start_time="2025-08-01T08:00:00",
    end_time="2025-08-01T18:00:00",
    bounding_box=[N, S, W, E],
    observations_per_day=5,
    jump_minutes=60,
    start_hour=8,
)

ZarrExport(
    args,                                   # argparse Namespace; needs .threading (and .yes/.remove)
    downloader,
    label="mtg_2025_08",
    channels=["vis_06", "vis_08", "ir_105"],
    area_reprojection="mtg_fci_latlon_1km",
    reprojection="nearest",                 # pyresample: nearest / bilinear
    chunks={"time": 1, "lat": 500, "lon": 500},   # lat/lon auto-computed (~128 MB chunks)
    custom_size={"time": 150},              # pre-allocate time slots
)
```

Output: `./data/datastore_data/MTG_FCI_<label>.zarr` (+ `MTG_FCI_<label>.status.json`). The `MTG_FCI_` prefix is applied for every product, including SEVIRI.

`area_reprojection` values supported by `ZarrExport._get_size` (other names in `areas.yaml` raise `NotImplementedError`):

| Name | Shape (lat × lon) |
|------|-------|
| `mtg_fci_latlon_1km` | 14000 × 14000 |
| `worldeqc3km` | 2048 × 4096 |
| `worldeqc3km70` | 4096 × 8192 |
| `worldeqc1km70` | 15585 × 40075 (not defined in `areas.yaml`) |
| `EPSG_4326_36000x18000` | 18000 × 36000 |

## Usage — MajorTOM Sparse Zarr

`ZarrExport(..., majortom=True)` routes to `MajorTomZarrExport` (can also be called directly). Patches are cut in the native geostationary projection; no regular-grid reprojection.

```python
ZarrExport(
    args, downloader,
    label="mtg_2025_08",
    channels=["vis_06", "vis_08"],
    majortom=True,
    majortom_patch_size=64,        # half-side → 128×128 patches
    majortom_grid_path=None,       # None = generate grid locally at majortom_spacing_km
    majortom_spacing_km=100.0,
    bbox_filter=(lat_min, lat_max, lon_min, lon_max),   # optional, restricts grid cells
    dtype="float16",               # default float32
    zarr_path=None,                # set to a shared path to append several tiles into one store
)
```

- `also_majortom=True` writes **both** the regular-grid zarr and the MajorTOM zarr from a single download.
- Patches with > 50% NaN are rejected (`max_null_fraction`).
- Re-running with the same label appends new timestamps / grid cells; existing `grid_ids` are skipped (no double-append).
- `merge_majortom_zarrs(tile_paths, output_path)` merges per-tile stores with dedup by `grid_id`.

Output: `./data/datastore_data/<label>_majortom.zarr` (or `zarr_path`). Batched layout:
```
patches/
  <channel>/                 e.g. vis_06
    <timestamp>/             e.g. 2025-08-01T12:00:00
      data            (N, H, W)  chunks (1, H, W)
      grid_ids        (N,)       MajorTOM cell IDs, e.g. "266U_764R" (U/D rows, R/L cols)
      patch_origins   (N, 2)     [row0, col0] top-left pixel in the full disk
```

Grid helpers (in `eumetsearch.transform`): `FCIMajorTomGrid`, `generate_majortom_grid`, `generate_regular_grid`, `grid_id_to_latlon`, `grid_ids_to_latlons`, `latlon_to_grid_id`, `latlon_to_fci_pixel`, `extract_fci_patch`.

## Reading Back

```python
from eumetsearch import load_zarr_preprocess
ds = load_zarr_preprocess("./data/datastore_data/MTG_FCI_mtg_2025_08.zarr", storage_options={})
# ds.vis_06 → (time, lat, lon)

import zarr
z = zarr.open_group("./data/datastore_data/mtg_2025_08_majortom.zarr", mode="r")
grp = z["patches"]["vis_06"]["2025-08-01T12:00:00"]
ids = grp["grid_ids"][:]
patch = grp["data"][list(ids).index(ids[0])]        # (128, 128)
```

## Running from the CLI

**Tiled pipeline (recommended)** — `src/eumetsearch/data_collection/pipeline_data.py`:
```bash
python -m eumetsearch.data_collection.pipeline_data \
    --product MTG-FCI-L1C-FDHSI --channels vis_06 vis_08 \
    --region europe --n_lat 4 --n_lon 4 \
    --start_date 2025-06-26T08:00:00 --end_date 2025-06-26T13:00:00 \
    --observations_per_day 6 --jump_minutes 60 \
    --majortom --dtype float16 --consolidated_label mtg_june_2025
```
Key flags: `--product` (any `products_list` key), `--region` or `--lon_min/--lat_min/--lon_max/--lat_max`, `--n_lat/--n_lon` (tiles), `--batch_days`, `--area_reprojection`, `--majortom` / `--also_majortom`, `--majortom_patch_size`, `--majortom_grid_path`, `--majortom_spacing_km`, `--min_land_fraction` (GEE land filter, cached in `water_min.npy`), `--max_parallel_conns`, `--dtype`, `-d/--delete_temp`, `--consolidated_label` (merge tiles into `<label>_majortom.zarr`). Most flags can also be set via env vars of the same name.

**Legacy monthly batch** — `main.py` (FCI FDHSI, `vis_06`/`vis_08`, monthly `mtg_YYYY_MM` zarrs):
```bash
python main.py --resampler nearest --region mtg --yes
```

Common flags (both scripts):
- `-y` / `--yes`: auto-confirm deletion of an existing zarr (3 s countdown)
- `-r` / `--remove`: auto-delete source files after processing
- `-t` / `--threading`: threading is **on by default**; passing `-t` turns it **off** (thread-safe single-thread mode, use if satpy/dask crashes)
- `--resampler`: `nearest` or `bilinear`
- `--dask_threads`: 0 = single-threaded dask

## Uploading to HuggingFace

```bash
python src/eumetsearch/utils/upload_to_hf.py <local_zarr_path> <hf_folder_name>
```
Requires `HF_TOKEN` in `.env`. The target dataset repo is set by `HF_REPO` in the script.

## Implementation Files

| File | Role |
|------|------|
| `src/eumetsearch/data_collection/downloader.py` | `products_list`, channel constants, `EUMDownloader`, `ZarrExport` |
| `src/eumetsearch/data_collection/majortom_export.py` | `MajorTomZarrExport`, `merge_majortom_zarrs` |
| `src/eumetsearch/data_collection/pipeline_data.py` | Tiled CLI pipeline |
| `src/eumetsearch/transform/majortom_fci.py` | MajorTOM grid + FCI pixel/patch utilities |
| `src/eumetsearch/transform/fci_modis_align.py` | FCI ↔ MODIS patch alignment (`align_patches`, `align_patches_multiband`) |
| `src/eumetsearch/transform/ndvi.py` | `compute_ndvi`, `fci_ndvi_one_timestamp`, `fci_ndvi_daily_max`, `modis_ndvi_one_date` |
| `src/eumetsearch/utils/bbox.py` | `BBOX_REGISTRY`, `get_bbox`, tile generators |
| `src/eumetsearch/utils/zarr.py` | `ZarrStore`, `extend_time_dim`, `add_channels_to_zarr`, `compute_auto_chunks` |
| `src/eumetsearch/utils/general.py` | `JsonDataResponse`, `aggregate_status_jsons`, `load_zarr_preprocess`, `init_logging` |
| `src/eumetsearch/utils/upload_to_hf.py` | HuggingFace upload |
| `src/eumetsearch/viz.py` | NDVI plots / tile comparison |
| `src/eumetsearch/utils/areas.yaml` | pyresample area definitions |
| `main.py` | Legacy monthly batch script |
