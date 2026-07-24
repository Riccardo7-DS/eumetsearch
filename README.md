<p align="center">
  <img src="assets/mtg_first_full_disk.jpg" alt="Meteosat Third Generation — first full-disk FCI image" width="480">
</p>

<p align="center">
  <em>Meteosat-12 (MTG-I1) first full-disk image, Flexible Combined Imager — © EUMETSAT, via <a href="https://commons.wikimedia.org/wiki/File:Meteosat_12_First_Full_Disk_image.jpg">Wikimedia Commons</a></em>
</p>

# EUMETSearch

[![PyPI version](https://img.shields.io/pypi/v/eumetsearch.svg)](https://pypi.org/project/eumetsearch/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.txt)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

Query and download data from the EUMETSAT DataStore and DataTailor — MTG FCI and MSG SEVIRI products — and export it to regular-grid or [MajorTOM](https://github.com/ESA-PhiLab/Major-TOM) sparse Zarr stores.

## Features

- Search EUMETSAT collections by time window and bounding box, with no download.
- One-off, single-window downloads (`download_single`) for event-based use cases.
- Batched, multi-day interval downloads (`download_interval`) with threaded I/O for larger jobs.
- Export downloaded data straight to a regular lat/lon Zarr store or a MajorTOM sparse Zarr store.
- Look up product metadata either by the package's short product key (e.g. `"MTG-FCI-L2-AFM"`) or directly by EUMETSAT product code (e.g. `"EO:EUM:DAT:0733"`).
- A tiled CLI pipeline (`pipeline_data.py`) for large-area, multi-tile batch downloads.

## Installation

```bash
pip install eumetsearch
```

From source, for development:

```bash
git clone https://github.com/Riccardo7-DS/eumetsearch.git
cd eumetsearch
pip install -e .
```

### Optional extras

| Extra | Installs | Used for |
|---|---|---|
| `transform` | numpy, zarr, xarray, dask, pyproj, scipy, pandas, geopandas, rioxarray | Lightweight analysis/alignment only, no download deps |
| `gee` | earthengine-api | Land-fraction tile filtering in `pipeline_data.py --min_land_fraction` |

```bash
pip install "eumetsearch[transform]"
```

## Authentication

Downloading requires an EUMETSAT Data Store API key/secret — register at the [EUMETSAT API key portal](https://api.eumetsat.int/api-key/) to get one. Copy `.env.example` to `.env` and fill in:

```bash
EUMETSAT_CONSUMER_KEY="..."
EUMETSAT_CONSUMER_SECRET="..."
```

`EUMDownloader` loads this automatically (`python-dotenv`), as long as the `.env` file is somewhere above your working directory.

## Quickstart

### 1. Look up a product

Products are keyed by a short name in `products_list` (see [Supported products](#supported-products)), each mapping to metadata including the real EUMETSAT product code:

```python
from eumetsearch import products_list, EUMDownloader

product_id = products_list["MTG-FCI-L2-AFM"]["product_id"]   # "EO:EUM:DAT:0733"

# Or go the other way — resolve metadata directly from a product code,
# without needing to know (or look up) the dictionary key:
meta = EUMDownloader.get_product_metadata("EO:EUM:DAT:0733")
print(meta["product_name"], meta["bands"])
```

### 2. Search without downloading

```python
downloader = EUMDownloader(
    product_id=product_id,
    output_dir="./data/datastore_data",
)

results = downloader.search(
    start_time="2024-07-23T12:00:00",
    end_time="2024-07-23T12:10:00",
    bounding_box=[41.20, 41.02, -8.70, -8.48],  # [N, S, W, E]
)
print(results["count"], results["products"])
```

### 3. Download a single time window

For one-off, event-based downloads — no daily-interval splitting or thread-chunking, just one window in one shot:

```python
file_list = downloader.download_single(
    start_time="2024-07-23T12:00:00",
    end_time="2024-07-23T12:10:00",
    bounding_box=[41.20, 41.02, -8.70, -8.48],  # [N, S, W, E]
    method="datatailor",  # or "datastore"
)
```

Files land in `output_dir`. See `notebook/quickstart.ipynb` for a full worked example, from search through to plotting the downloaded data.

### 4. Batch downloads over multiple days

```python
downloader.download_interval(
    start_time="2025-06-01T08:00:00",
    end_time="2025-06-05T08:00:00",
    bounding_box=[41.20, 41.02, -8.70, -8.48],
    observations_per_day=6,
    jump_minutes=60,
)
```

`download_interval` sets up the daily-interval splitting; combine it with `ZarrExport` (regular lat/lon grid) or `MajorTomZarrExport` (sparse MajorTOM grid) to trigger the actual download and write results to Zarr. For large, multi-tile areas, the CLI pipeline wraps this end-to-end:

```bash
python -m eumetsearch.data_collection.pipeline_data \
    --product MTG-FCI-L1C-FDHSI \
    --channels vis_06 vis_08 \
    --region europe \
    --n_lat 4 --n_lon 4 \
    --start_date 2025-06-26T08:00:00 \
    --end_date   2025-06-26T13:00:00 \
    --observations_per_day 6 \
    --jump_minutes 60 \
    --majortom
```

## Supported products

| Key | Product | EUMETSAT code |
|---|---|---|
| `MTG-FCI-L1C-FDHSI` | MTG FCI L1C — Full Disk High Spectral Res. | `EO:EUM:DAT:0662` |
| `MTG-FCI-L1C-HRFI` | MTG FCI L1C — High Resolution Full Disk | `EO:EUM:DAT:0665` |
| `MTG-FCI-L2-CLM` | MTG FCI L2 — Cloud Mask | `EO:EUM:DAT:0666` |
| `MTG-FCI-L2-AFM` | MTG FCI L2 — Active Fire Monitoring | `EO:EUM:DAT:0733` |
| `MTG-FCI-L2-OCA` | MTG FCI L2 — Optimal Cloud Analysis | `EO:EUM:DAT:0668` |
| `MTG-FCI-L2-GII` | MTG FCI L2 — Global Instability Indices | `EO:EUM:DAT:0683` |
| `MTG-FCI-L2-AMV` | MTG FCI L2 — Atmospheric Motion Vectors | `EO:EUM:DAT:0676` |
| `MSG-SEVIRI-L1-HRY` | MSG SEVIRI L1.5 — Prime (0°E) | `EO:EUM:DAT:MSG:HRSEVIRI` |
| `MSG-SEVIRI-L1-IODC` | MSG SEVIRI L1.5 — Indian Ocean (41.5°E) | `EO:EUM:DAT:MSG:HRSEVIRI-IODC` |
| `MSG-SEVIRI-RSS` | MSG SEVIRI — Rapid Scan Service (3.5°E) | `EO:EUM:DAT:MSG:MSG15-RSS` |

## License

Apache License 2.0 — see [LICENSE.txt](LICENSE.txt).
