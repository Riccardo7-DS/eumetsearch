
import os
from dotenv import load_dotenv
import xarray as xr
from eumetsearch import prepare
from eumetsearch import hoa_bbox, load_zarr_preprocess
from dask.diagnostics import ProgressBar
import numpy as np

from dask.distributed import Client, LocalCluster


def main():
    # ------------------------------------------------------
    # START LOCAL DASK CLUSTER
    # ------------------------------------------------------
    cluster = LocalCluster(
        n_workers=8,
        threads_per_worker=1,
        memory_limit="20GB",
    )
    client = Client(cluster)
    print(client)
    print("Dashboard:", client.dashboard_link)

load_dotenv()
t = ProgressBar().register()

access_key = os.environ["MINIO_ACCESS_KEY"]
access_secret = os.environ["MINIO_SECRET_KEY"]


storage_options={
        "key": access_key,
        "secret": access_secret,
        "client_kwargs": {"endpoint_url": "https://object-storage.esrin.it.esa.int/"}
    }


from eumetsearch import extract_mtg_coords, compute_ndvi
# from dask.distributed import Client
# client = Client()   # optional


# ======================================================
# 1. Your updated load_zarr_preprocess (optimized)
# ======================================================

def load_zarr_preprocess(path: str, storage_options: dict) -> xr.Dataset:
    """
    Load and minimally preprocess the raw MTG Zarr dataset:
    - lazy open
    - assign external coordinates
    - flip latitude
    - decode timeStart into datetime64
    - compute NDVI if vis_06 and vis_08 exist
    """
    ds = xr.open_zarr(
        path,
        decode_times=False,
        storage_options=storage_options,
    )

    lon_1d, lat_1d = extract_mtg_coords()

    ds = ds.assign_coords(
        lat=("lat", lat_1d),
        lon=("lon", lon_1d)
    )

    # Flip latitude dimension
    ds = ds.isel(lat=slice(None, None, -1))

    # Decode time from timeStart
    time_array = np.array(ds["timeStart"].values, dtype="datetime64[ns]")
    ds = ds.assign_coords(time=("time", time_array))

    # Compute NDVI if reflectances exist
    if {"vis_06", "vis_08"}.issubset(ds.data_vars):
        ds = compute_ndvi(ds, "vis_06", "vis_08")
        return ds[["ndvi"]]

    return ds


# ======================================================
# 2. Loader for ONE month with spatial subset + dedup + daily max
# ======================================================

TARGET_CHUNKS = {"time": 100, "lat": 500, "lon": 500}

def load_zarr_minio_subset(path, storage_options, lat_slice, lon_slice, chunks=TARGET_CHUNKS):
    """
    1) open with preprocess function (lazy)
    2) remove NaT times
    3) spatial subset
    4) dedup times (fast np.unique)
    5) resample daily max (lazy)
    6) apply chunking
    """
    ds = load_zarr_preprocess(path, storage_options)

    # Remove NaT times
    times = ds["time"].values  # time axis is small → OK to load
    good_mask = ~np.isnat(times)
    if not np.all(good_mask):
        ds = ds.isel(time=np.where(good_mask)[0])

    # Spatial subset BEFORE resample = HUGE speedup
    ds = ds.sel(lat=lat_slice, lon=lon_slice)

    # Fast dedup (keep first occurrence)
    times = ds["time"].values
    _, unique_idx = np.unique(times, return_index=True)
    ds = ds.isel(time=np.sort(unique_idx))
    ds = ds.sortby("time").drop_duplicates('time')

    # Daily aggregation
    ds = ds.resample(time="1D").max()

    # Apply chunking once
    ds = ds.chunk(chunks)

    return ds


# ======================================================
# 3. Brazil bounding box coordinates and slices
# ======================================================

if __name__=="__main__":

    main()

    brazil_box = {
        "min_lon": -74.0,
        "max_lon": -34.0,
        "min_lat": -34.0,
        "max_lat":  5.5
    }
    
    lon_1d, lat_1d = extract_mtg_coords()
    
    lat_min, lat_max = brazil_box["min_lat"], brazil_box["max_lat"]
    lon_min, lon_max = brazil_box["min_lon"], brazil_box["max_lon"]
    
    lat_idx = np.where((lat_1d >= lat_min) & (lat_1d <= lat_max))[0]
    lon_idx = np.where((lon_1d >= lon_min) & (lon_1d <= lon_max))[0]
    
    i_min, i_max = lat_idx.min(), lat_idx.max()
    j_min, j_max = lon_idx.min(), lon_idx.max()
    
    lat_slice = slice(lat_1d[i_max], lat_1d[i_min])   # note reversed order because lat is flipped
    lon_slice = slice(lon_1d[j_min], lon_1d[j_max])
    
    
    # ======================================================
    # 4. Load all months lazily with spatial subset
    # ======================================================
    
    
    paths = [
        "s3://mtg-fci-data/MTG_FCI_mtg_2025_06.zarr/MTG_FCI_mtg_2025_06.zarr",
        "s3://mtg-fci-data/MTG_FCI_mtg_2025_07.zarr",
        "s3://mtg-fci-data/MTG_FCI_mtg_2025_08.zarr",
        "s3://mtg-fci-data/MTG_FCI_mtg_2025_09.zarr",
    ]
    
    datasets = []
    for p in paths:
        print("Opening:", p)
        datasets.append(
            load_zarr_minio_subset(
                p, storage_options, lat_slice, lon_slice, chunks=TARGET_CHUNKS
            )
        )
    
    
    # ======================================================
    # 5. Concatenate lazily
    # ======================================================
    
    ds_all = xr.concat(datasets, dim="time", combine_attrs="drop_conflicts")
    ds_all = ds_all.chunk(TARGET_CHUNKS)  # final unified chunking
    
    
    # ======================================================
    # 6. Ensure time sorted (cheap)
    # ======================================================
    
    times = ds_all["time"].values
    if not np.all(np.diff(times) >= np.timedelta64(0, "s")):
        ds_all = ds_all.sortby("time")
    
    
    # ======================================================
    # 7. Write output to Zarr (compute here)
    # ======================================================
    
    out_zarr = "/home/riccardo/Desktop/Riccardo/Projects/mtg-fci-query/data/ndvi_brazil_daily_max_2025_06_09.zarr"
    
    if os.path.exists(out_zarr):
        import shutil
        shutil.rmtree(out_zarr)
    
    print("Writing final dataset to:", out_zarr)
    with ProgressBar():
        ds_all.to_zarr(out_zarr, mode="w")
    
    print("Done!")