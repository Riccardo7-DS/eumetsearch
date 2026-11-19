---
name: download-eumetsat
description: Downloading eumetsat data with eumetsearch. Use this skill when the user wants to download EUMETSAT satellite data (MTG FCI, MSG SEVIRI), export to Zarr, or use MajorTOM tiling. Triggers on phrases like "download FCI data", "get MTG data", "fetch MSG SEVIRI", "download satellite data from EUMETSAT", "export to zarr from eumetsat".
---

# Downloading EUMETSAT Data with eumetsearch

The `eumetsearch` package downloads EUMETSAT satellite data via the `eumdac` API and exports it to Zarr, either as a regular lat/lon grid or as a MajorTOM sparse grid.

## Setup

Required environment variables (in `.env`):
```
EUMETSAT_CONSUMER_KEY=<your_key>
EUMETSAT_CONSUMER_SECRET=<your_secret>
```

Get credentials at: https://data.eumetsat.int → Login → API Key.

## Supported Products

All product IDs come from the `products_list` dict in `src/eumetsearch/data_collection/downloader.py`.

| Key | Product ID | Description | Satpy reader |
|-----|-----------|-------------|-------------|
| `MTG-FCI-L1C-FDHSI` | `EO:EUM:DAT:0662` | MTG-I1 FCI L1C Full Disk HSR | `fci_l1c_nc` |
| `MTG-FCI-L1C-HRFI` | `EO:EUM:DAT:0665` | MTG-I1 FCI L1C High-Res Full Disk | `fci_l1c_nc` |
| `MTG-FCI-L2-CLM` | `EO:EUM:DAT:0666` | MTG-I1 FCI L2 Cloud Mask | `fci_l2_nc` |
| `MTG-FCI-L2-AFM` | `EO:EUM:DAT:0733` | MTG-I1 FCI L2 Active Fire | `fci_l2_nc` |
| `MTG-FCI-L2-OCA` | `EO:EUM:DAT:0668` | MTG-I1 FCI L2 Optimal Cloud Analysis | `fci_l2_nc` |
| `MSG-SEVIRI-L1-HRY` | `EO:EUM:DAT:MSG:HRSEVIRI` | MSG SEVIRI L1.5 Prime (0°E) | `seviri_l1b_hrit` |
| `MSG-SEVIRI-L1-IODC` | `EO:EUM:DAT:MSG:HRSEVIRI-IODC` | MSG SEVIRI L1.5 IODC (41.5°E) | `seviri_l1b_hrit` |
| `MSG-SEVIRI-RSS` | `EO:EUM:DAT:MSG:RSS` | MSG SEVIRI Rapid Scan (3.5°E) | `seviri_l1b_hrit` |

## FCI Channels

```python
# All 16 FCI L1C channels:
FCI_CHANNELS_VIS_NIR = ["vis_04","vis_05","vis_06","vis_08","vis_09","nir_13","nir_16","nir_22"]
FCI_CHANNELS_IR      = ["ir_38","wv_63","wv_73","ir_87","ir_97","ir_105","ir_123","ir_133"]
FCI_CHANNELS_ALL     = FCI_CHANNELS_VIS_NIR + FCI_CHANNELS_IR
```

## MSG SEVIRI Channels

```python
SEVIRI_CHANNELS_ALL = ["VIS006","VIS008","NIR016","IR039","WV062","WV073","IR087","IR097","IR108","IR120","IR134"]
SEVIRI_CHANNELS_HRV = SEVIRI_CHANNELS_ALL + ["HRV"]
```

## Usage — Regular Grid (lat/lon Zarr)

```python
from eumetsearch import EUMDownloader, ZarrExport, products_list, bbox_mtg

# 1. Build the downloader
product_id = products_list["MTG-FCI-L1C-FDHSI"]["product_id"]
downloader = EUMDownloader(
    product_id=product_id,
    output_dir="./data/datastore_data",
    max_parallel_conns=10,
)

# 2. Specify time range and bounding box
W, S, E, N = bbox_mtg()          # pre-defined MTG bounding box
downloader.download_interval(
    start_time="2025-08-01T08:00:00",
    end_time="2025-08-01T18:00:00",
    bounding_box=[N, S, W, E],
    observations_per_day=5,
    jump_minutes=60,
)

# 3. Export to Zarr (regular lat/lon grid)
ZarrExport(
    args,                               # argparse Namespace
    downloader,
    label="mtg_2025_08",
    channels=["vis_06", "vis_08", "ir_105"],
    area_reprojection="mtg_fci_latlon_1km",  # see areas.yaml
    reprojection="nearest",
    chunks={"time": 1, "lat": 500, "lon": 500},
    custom_size={"time": 150},           # pre-allocate 150 time slots
)
```

The zarr is written to `./data/datastore_data/MTG_FCI_mtg_2025_08.zarr`.

## Usage — MajorTOM Sparse Grid

Pass `majortom=True` to `ZarrExport`. No reprojection to a regular grid is performed; instead, 128×128-pixel patches are extracted at each MajorTOM grid cell location in the native FCI geostationary projection.

```python
ZarrExport(
    args,
    downloader,
    label="mtg_2025_08",
    channels=["vis_06", "vis_08", "ir_105"],
    majortom=True,
    majortom_patch_size=64,   # full patch = 128×128 pixels
    majortom_grid_path=None,  # None = auto-download MajorTOM grid
)
```

Output: `./data/datastore_data/mtg_2025_08_majortom.zarr`  
Structure:
```
patches/
  vis_06/
    2025-08-01T12:00:00/
      0266_0764/    # array shape (128, 128)
      0267_0765/
  vis_08/
    ...
```

## Zarr Append Behaviour

- **New dates on an existing store**: `ZarrStore` detects how many time slots are filled and automatically extends the time dimension (`zarr.Array.resize`).
- **New channels on an existing store**: New channel arrays are added without touching existing data (`add_channels_to_zarr`).
- **Skipping already-processed dates**: The aggregated status JSON (`aggregated_data.json`) tracks processed timestamps; `files_exclusion()` filters those out before downloading.

## Area Definitions

Available `area_reprojection` values (see `src/eumetsearch/utils/areas.yaml`):

| Name | Resolution | Shape |
|------|-----------|-------|
| `mtg_fci_latlon_1km` | ~1 km | 14000×14000 |
| `worldeqc3km` | ~3 km | 2048×4096 |
| `worldeqc3km70` | ~3 km | 4096×8192 |
| `worldeqc1km70` | ~1 km | 15585×40075 |
| `EPSG_4326_36000x18000` | ~1 km | 18000×36000 |

## Reading Back the Zarr

```python
import xarray as xr
from eumetsearch import load_zarr_preprocess

# Regular grid zarr
ds = load_zarr_preprocess("./data/datastore_data/MTG_FCI_mtg_2025_08.zarr", storage_options={})
# ds.vis_06 → shape (time, lat, lon)

# MajorTOM sparse zarr
import zarr
z = zarr.open("./data/datastore_data/mtg_2025_08_majortom.zarr", mode="r")
patch = z["patches"]["vis_06"]["2025-08-01T12:00:00"]["0266_0764"][:]  # (128, 128) array
```

## Running from the CLI

```bash
cd eumetsearch  # repo root
python main.py --resampler nearest --yes
```

Key CLI flags:
- `--yes` / `-y`: auto-confirm deletion of existing zarr
- `--remove` / `-r`: auto-delete temp source files after processing
- `-t` / `--threading`: enable threading for I/O (default: single-thread safe mode)
- `--resampler`: pyresample resampler (`nearest`, `bilinear`)

## Implementation Files

| File | Role |
|------|------|
| `src/eumetsearch/data_collection/downloader.py` | `EUMDownloader` (auth, search, download) + `ZarrExport` (regular grid pipeline) |
| `src/eumetsearch/data_collection/majortom_export.py` | `MajorTomZarrExport` (sparse MajorTOM pipeline) |
| `src/eumetsearch/transform/majortom_fci.py` | `FCIMajorTomGrid`, `latlon_to_fci_pixel`, `extract_fci_patch` |
| `src/eumetsearch/utils/zarr.py` | `ZarrStore`, `extend_time_dim`, `add_channels_to_zarr`, `compute_auto_chunks` |
| `src/eumetsearch/utils/general.py` | `JsonDataResponse`, `aggregate_status_jsons`, `load_zarr_preprocess` |
| `src/eumetsearch/utils/areas.yaml` | pyresample area definitions |
| `main.py` | Batch download orchestration script |
