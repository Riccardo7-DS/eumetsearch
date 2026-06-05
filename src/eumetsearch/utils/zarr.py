import zarr
import dask.array as da
import xarray as xr
import os
from pydantic import PositiveInt
import math
import logging
import shutil
import numpy as np

logger = logging.getLogger(__name__)

class ZarrStore:
    def __init__(self, 
                 folder_path, 
                 size:list,
                 file_list:list,
                 channels:list, 
                 ds:xr.Dataset=None,
                 label:str="VIS",
                 chunks: dict = {"time": 1, "lat": "auto", "lon": "auto"}, 
                 n_workers:PositiveInt=4,
                 custom_size:dict | None = None,
                 remove_flag:bool=False):
        
        self.folder_path = folder_path
        self._num_timesteps = len(file_list) if not custom_size or custom_size.get("time") is None else custom_size["time"]
        self._size = size  # (height, width)
        self._chunks = chunks
        self._num_timechunks = chunks.get("time", 1)
        self._num_ychunks = chunks.get("lat", size[0])
        self._num_xchunks = chunks.get("lon", size[1])
        self._n_workers = n_workers

        zarr_path = self.zarr_store_create(
            label=label, 
            channels=channels, 
            size=size,
            ds_example=ds,
            remove_flag=remove_flag
        )

        self.path = zarr_path

    def zarr_store_create(self, label, channels, size, ds_example=None, remove_flag=False):
        def on_rm_error(func, path, exec_info):
            import stat
            os.chmod(path, stat.S_IWRITE)
            func(path)

        zarr_path = os.path.join(
            self.folder_path,
            f'MTG_FCI_{label}.zarr'
        )

        if os.path.exists(zarr_path):
            if remove_flag:
                response = 'yes'

            elif not remove_flag:
                response = "no"
                
            else:
                response = input(
                    f"The folder '{os.path.basename(zarr_path)}' already exists. Do you want to delete it? (yes/no): "
                ).strip().lower()

            if response == 'yes':
                try:
                    shutil.rmtree(zarr_path, onerror=on_rm_error)
                    logger.info(f"Deleted existing folder: {zarr_path}")
                except Exception as e:
                    logger.error(f"Error deleting folder {zarr_path}: {e}")
                    raise FileExistsError(
                        f"Could not delete folder '{os.path.basename(zarr_path)}'. Please delete it manually or choose a different folder."
                    )
            elif response == "no":
                # Existing store: check whether we need to extend time or add channels
                logger.info("The zarr store already exists — checking for extension needs...")
                existing = zarr.open(zarr_path, mode="a")
                existing_time_size = existing["filled_flag"].shape[0]
                need_time = self._num_timesteps > existing_time_size
                existing_vars = set(existing.keys())
                need_channels = [c for c in channels if c not in existing_vars]

                if need_time:
                    n_new = self._num_timesteps - existing_time_size
                    logger.info(f"Extending time dimension by {n_new} slots...")
                    extend_time_dim(zarr_path, n_new, self._num_timechunks)

                if need_channels:
                    logger.info(f"Adding new channels: {need_channels}")
                    height, width = size
                    lat_chunk = (self._num_ychunks if isinstance(self._num_ychunks, int)
                                 else height)
                    lon_chunk = (self._num_xchunks if isinstance(self._num_xchunks, int)
                                 else width)
                    add_channels_to_zarr(
                        zarr_path, need_channels,
                        time_size=self._num_timesteps,
                        height=height, width=width,
                        time_chunk=self._num_timechunks,
                        lat_chunk=lat_chunk,
                        lon_chunk=lon_chunk,
                    )

                return zarr_path
            else:
                raise ValueError(f"Invalid response: '{response}'. Expected 'yes' or 'no'.")
                

        # All recognised channel names and their dtypes
        _VIS_NIR_CHANNELS = {
            "vis_04", "vis_05", "vis_06", "vis_08", "vis_09",
            "nir_13", "nir_16", "nir_22",
            "vis_06_hr", "vis_08_hr",
            "VIS006", "VIS008", "NIR016", "HRV",
        }
        _IR_CHANNELS = {
            "ir_38", "wv_63", "wv_73", "ir_87", "ir_97",
            "ir_105", "ir_123", "ir_133",
            "IR039", "WV062", "WV073", "IR087", "IR097", "IR108", "IR120", "IR134",
        }
        _L2_CHANNELS = {
            "cloud_state", "fire_probability", "cloud_top_pressure", "cloud_optical_thickness",
        }
        _ALL_VARS = _VIS_NIR_CHANNELS | _IR_CHANNELS | _L2_CHANNELS

        # Dimensions
        num_time = self._num_timesteps
        height, width = size
        time_coord = np.full(num_time, np.datetime64("NaT", "ns"))

        meta_vars = ["identifier", "timeStart", "timeEnd"]

        assert all(v in _ALL_VARS for v in channels), \
            f"One or more channels are invalid. Allowed: {sorted(_ALL_VARS)}"
        selected_vars = list(channels) + meta_vars

        def _channel_dtype(var):
            if var in _VIS_NIR_CHANNELS:
                return "float32"
            if var in _IR_CHANNELS:
                return "int32"
            return "float32"  # L2 float by default

        data_vars = {}
        encoding = {}
        compressor = zarr.Blosc(cname="zstd", clevel=4)

        for var in selected_vars:
            if var in meta_vars:
                shape = (num_time,)
                chunks = (self._num_timechunks,)
                dtype = "S143" if var == "identifier" else "datetime64[ns]"
                fill_value = b"" if var == "identifier" else np.datetime64("NaT")
                dims = ("time",)
            else:
                shape = (num_time, height, width)
                chunks = (self._num_timechunks, self._num_ychunks, self._num_xchunks)
                dtype = _channel_dtype(var)
                fill_value = np.nan
                dims = ("time", "lat", "lon")

            data_vars[var] = (dims, da.full(shape, fill_value, dtype=dtype, chunks=chunks))
            encoding[var] = {"compressor": compressor, "chunks": chunks}
            

        filled_flag = da.zeros((num_time,), dtype=bool, chunks=(self._num_timechunks,))
        data_vars["filled_flag"] = (("time",), filled_flag)
        encoding["filled_flag"] = {"compressor": compressor, "chunks": (self._num_timechunks,)}
    
        ds_empty = xr.Dataset(data_vars=data_vars, coords={"time": time_coord})

        if ds_example is not None:
            merged_attrs = ds_example.attrs.copy()
            merged_attrs.update(ds_example[channels[1]].attrs)
            ds_empty = ds_empty.assign_attrs(merged_attrs)
            to_del = ["time_parameters", "ancillary_variables", "end_time", "start_time"]
            for attr in to_del:
                ds_empty.attrs.pop(attr, None)

        ds_empty.to_zarr(zarr_path, mode='w', compute=True, consolidated=True)

        return zarr_path


def extend_time_dim(zarr_path: str, n_new: int, time_chunk: int = 1) -> None:
    """Grow the time axis of every array in an existing zarr store by n_new slots.

    New slots are filled with the array's fill_value (NaN / NaT / False / empty).
    """
    z = zarr.open(zarr_path, mode="a")
    for name, arr in z.items():
        if not isinstance(arr, zarr.Array):
            continue
        old_shape = arr.shape
        new_shape = (old_shape[0] + n_new,) + old_shape[1:]
        arr.resize(*new_shape)
        logger.debug(f"Extended '{name}' from {old_shape} to {new_shape}")

    # Rebuild consolidated metadata so xr.open_zarr sees the new sizes
    zarr.consolidate_metadata(zarr_path)
    logger.info(f"Time dimension extended by {n_new} in {zarr_path}")


def add_channels_to_zarr(
    zarr_path: str,
    channels: list[str],
    time_size: int,
    height: int,
    width: int,
    time_chunk: int = 1,
    lat_chunk: int = 500,
    lon_chunk: int = 500,
) -> None:
    """Add new channel arrays to an existing zarr store without touching existing data."""
    _VIS_NIR = {
        "vis_04", "vis_05", "vis_06", "vis_08", "vis_09",
        "nir_13", "nir_16", "nir_22", "vis_06_hr", "vis_08_hr",
        "VIS006", "VIS008", "NIR016", "HRV",
    }
    compressor = zarr.Blosc(cname="zstd", clevel=4)
    z = zarr.open(zarr_path, mode="a")

    for ch in channels:
        if ch in z:
            logger.debug(f"Channel '{ch}' already in store, skipping.")
            continue
        dtype = "float32" if ch in _VIS_NIR else "int32"
        fill_value = np.nan if dtype == "float32" else 0
        shape = (time_size, height, width)
        chunks = (time_chunk, lat_chunk, lon_chunk)
        z.require_dataset(
            ch,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            compressor=compressor,
            fill_value=fill_value,
        )
        logger.info(f"Added channel '{ch}' ({dtype}, shape={shape}) to {zarr_path}")

    zarr.consolidate_metadata(zarr_path)


def compute_auto_chunks(shape, dtype_size=4, fixed_chunks={"time": 1}, target_chunk_bytes=128 * 2**20):
    """
    Compute approximate Dask chunk sizes for 'auto' chunks given fixed chunks along one dimension.
    
    Parameters:
    - shape: dict with dimension names and sizes, e.g., {"time": 1000, "lat": 2000, "lon": 2000}
    - dtype_size: size in bytes (e.g., 4 for float32, 8 for float64)
    - fixed_chunks: dict like {"time": 1}, others will be considered "auto"
    - target_chunk_bytes: default 128 MiB

    Returns:
    - dict with chunk sizes, including computed sizes for "auto" dimensions
    """
    # Separate fixed and auto dimensions
    fixed_volume = dtype_size
    chunk_shape = {}
    
    for dim, size in shape.items():
        if dim in fixed_chunks:
            chunk_shape[dim] = fixed_chunks[dim]
            fixed_volume *= fixed_chunks[dim]
        else:
            chunk_shape[dim] = None  # mark as "auto"

    # Remaining bytes for auto dimensions
    remaining_bytes = target_chunk_bytes / fixed_volume
    
    # Determine number of elements needed for auto dims
    auto_dims = [dim for dim in chunk_shape if chunk_shape[dim] is None]
    if len(auto_dims) != 2:
        raise ValueError("Function currently supports exactly 2 auto dimensions (e.g., lat, lon)")
    
    dim1, dim2 = auto_dims
    size1, size2 = shape[dim1], shape[dim2]

    # Solve: chunk1 * chunk2 ≈ remaining_elements
    target_elements = remaining_bytes / dtype_size
    chunk1 = int(math.sqrt(target_elements * size1 / size2))
    chunk2 = int(target_elements / chunk1)

    # Clip to max dimension size
    chunk_shape[dim1] = min(chunk1, size1)
    chunk_shape[dim2] = min(chunk2, size2)

    return chunk_shape