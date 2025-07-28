import zarr 
import queue
import dask.array as da
import xarray as xr
import os
from pydantic import PositiveInt

class ZarrStore:
    def __init__(self, 
                 folder_path, 
                 size:list,
                 file_list:list,
                 channels:list, 
                 chunks: dict = {"time": 1, "y": 500, "x": 500}, 
                 n_workers:PositiveInt=4):
        
        self.folder_path = folder_path
        self._num_timesteps = len(file_list)
        self._size = size  # (height, width)
        self._chunks = chunks
        self._num_timechunks = chunks.get("time", 1)
        self._num_ychunks = chunks.get("y", 500)
        self._num_xchunks = chunks.get("x", 500)
        self._n_workers = n_workers

        zarr_path , encoding = self.zarr_store_create(
            label='VIS', 
            channels=channels, 
            size=size
        )

        self.path = zarr_path

    def zarr_store_create(self, label, channels, size):
        zarr_path = os.path.join(
            self.folder_path,
            f'MTG_FCI_{label}.zarr'
        )

        # Dimensions
        num_time = self._num_timesteps  
        height, width = size
        time_coord = list(range(num_time))

        all_vars = {
            "vis_06", "vis_08", "vis_09",
            "ir_105", "ir_123", "ir_133",
            "ir_38", "ir_87", "ir_97",
            "wv_63", "wv_73"
        }

        meta_vars = ["identifier", "unixTimeStart", "unixTimeEnd"]

        assert all(v in all_vars for v in channels), "One or more channels are invalid"
        selected_vars = set(channels)
        selected_vars.update(meta_vars)

        data_vars = {}
        encoding = {}
        compressor = zarr.Blosc(cname="zstd", clevel=4)

        for var in selected_vars:
            if var in meta_vars:
                shape = (num_time,)
                chunks = (self._num_timechunks,)
                dtype = 'S143' if var == "identifier" else "datetime64[ns]"
                dims = ("time",)
            else:
                shape = (num_time, height, width)
                chunks = (self._num_timechunks, self._num_ychunks, self._num_xchunks)
                dtype = 'float32' if var.startswith("vis_") else 'int32'
                dims = ("time", "y", "x")

            data_vars[var] = (dims, da.empty(shape, dtype=dtype, chunks=chunks))
            encoding[var] = {"compressor": compressor, 
                             "chunks": chunks}
    
        ds_empty = xr.Dataset(data_vars=data_vars, coords={"time": time_coord})
        ds_empty.to_zarr(zarr_path, mode='w', compute=True)

        return zarr_path, encoding
