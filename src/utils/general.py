import logging
import pandas as pd
from datetime import datetime
import xarray as xr
import re
import os
import zipfile
import logging
from glob import glob
from typing import TypeVar, Literal
from pydantic import validate_call
from datetime import UTC

T = TypeVar("T")


logger = logging.getLogger(__name__)

def bbox_mtg():
    return [-18.105469,-37.857507,60.820313,71.413177]


def coords_mtg_grid(resolution_deg = 0.08789, full_grid:bool = True):
    from pyresample import geometry, kd_tree
    # Approximate 1 km resolution in degrees (at equator)
    # ~1.1 km at equator
    if full_grid:
        lat_min = -90
        lat_max = 90
        lon_min = -180
        lon_max = 180
    else:
        # Unpack
        bbox = bbox_mtg()
        lon_min, lat_min, lon_max, lat_max = bbox

    height = int((lat_max - lat_min) / resolution_deg) 
    width = int((lon_max - lon_min) / resolution_deg)

    lats = np.linspace(lat_max, lat_min, height)  # Flip to ensure north-to-south order
    lons = np.linspace(lon_min, lon_max, width)

    # Create 2D lat/lon grid
    lon2d, lat2d = np.meshgrid(lons, lats)

    return lats, lons, lon2d, lat2d

    # # Wrap into xarray Dataset (optional)
    # grid_ds = xr.Dataset(
    #     coords={
    #         "lat": ("lat", lats),
    #         "lon": ("lon", lons)
    #     }
    # )

    # logger.info("Grid dimensions: ", lat2d.shape)
    # logger.info("Lat range: ", lats.min(), "to", lats.max())
    # logger.info("Lon range: ", lons.min(), "to", lons.max())

    # # Define source geometry (satellite scan)
    # source_swath = geometry.SwathDefinition(lons=lons, lats=lats)

    # # Define target grid (regular lat/lon grid)
    # target_grid = geometry.GridDefinition(lons=lon2d, lats=lat2d)

    # # Resample radiance to lat/lon grid
    # result = kd_tree.resample_nearest(source_swath, data, target_grid,
    #                                    radius_of_influence=50000, fill_value=None)
    # return result


def unzip_files(destination_folder, filename):
    with zipfile.ZipFile(os.path.join(destination_folder, filename), 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith(".nc"):
                zip_ref.extract(file, destination_folder)
        logger.info(f'Unzipping of product {filename} finished.')
    
    os.remove(os.path.join(destination_folder, filename))


def init_logging(log_file:str=None, verbose:bool=False)-> logging.Logger:
    import os
    # Determine the logging level
    if verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    # Define the logging format
    formatter = "%(asctime)s : %(levelname)s : [%(filename)s:%(lineno)s - %(funcName)s()] : %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    
    # Setup basic configuration for logging
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.basicConfig(
            level=level,
            format=formatter,
            datefmt=datefmt,
            handlers=[
                logging.FileHandler(log_file, "w"),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=level,
            format=formatter,
            datefmt=datefmt,
            handlers=[
                logging.StreamHandler()
            ]
        )

    logger = logging.getLogger()
    return logger


def assert_(item: T, message: str, exception: type[Exception] = ValueError, silent: bool = True) -> T:
    """Assert the truth value of the item, and return the item or raise ``exception``.

    Args:
        item:
            The item to assert. It does not have to be a boolean. For example, given an empty list, ``assert []`` fails
            as an empty list evaluates to ``False``.
        message:
            The exception message, which will be shown when the exception is raised.
        exception:
            The exception to raise if both the ``item`` and ``silent`` evaluate to ``False``.
            Defaults to ``ValueError``.
        silent:
            A boolean indicating whether to return the item silently or raise an exception in the case of assertion
            failure. Defaults to ``True``, which means the item will be silently returned.

    Returns:
        Return the item if it evaluates to ``True``. If the item evaluates to ``False``, return it only if ``silent``
        is ``True``.

    Raises:
        ``exception``:
            If the ``item`` evaluates to ``False`` and ``silent`` is ``False``.
    """
    if item or silent:
        return item
    raise exception(message)

def assert_datetime_is_timezone_aware(datetime_object: datetime, silent: bool = False) -> bool:
    """Assert that the ``datetime_object`` is timezone-aware.

    Note:
        This function relies on :func:`~monkey_wrench.generic.assert_`.

    Examples:
        >>> assert_datetime_is_timezone_aware(datetime.now(), silent=True)
        False

        >>> assert_datetime_is_timezone_aware(datetime.now(UTC))
        True
    """
    try:
        result = None not in [datetime_object.tzinfo, datetime_object.tzinfo.utcoffset(datetime_object)]
    except AttributeError:
        result = False

    return assert_(result, f"{datetime_object} is not timezone-aware!", silent=silent)


@validate_call
def assert_start_precedes_end(start_datetime: datetime, end_datetime: datetime, silent: bool = False) -> bool:
    """Assert that the ``start_datetime`` is not later than the ``end_datetime``.

    Note:
        This function relies on :func:`~monkey_wrench.generic.assert_`.

    Examples:
        >>> # The following will not raise an exception.
        >>> assert_start_precedes_end(datetime(2020, 1, 1), datetime(2020, 12, 31))
        True

        >>> # The following will raise an exception!
        >>> assert_start_precedes_end(datetime(2020, 1, 2), datetime(2020, 1, 1), silent=True)
        False

        >>> # The following will raise an exception!
        >>> # assert_start_precedes_end(datetime(2020, 1, 2), datetime(2020, 1, 1))
    """
    return assert_(
        start_datetime <= end_datetime,
        f"start_datetime='{start_datetime}' is later than end_datetime='{end_datetime}'.",
        silent=silent
    )


def assert_datetime_has_past(datetime_instance: datetime, silent: bool = False) -> bool:
    """Assert that the ``datetime_instance`` is in not in the future.

    Note:
        This function relies on :func:`~monkey_wrench.generic.assert_`.

    Examples:
        >>> # The following will not raise an exception.
        >>> assert_datetime_has_past(datetime(2020, 1, 1, tzinfo=UTC))
        True

        >>> assert_datetime_has_past(datetime(2100, 1, 2, tzinfo=UTC), silent=True)
        False

        >>> # The following will raise an exception!
        >>> # assert_has_datetime_past(datetime(2100, 1, 2, tzinfo=UTC))
    """
    assert_datetime_is_timezone_aware(datetime_instance, silent=False)
    return assert_(
        datetime_instance <= datetime.now(UTC),
        "The given datetime instance is in the future!",
        silent=silent
    )



def add_time(dataset: xr.Dataset) -> xr.Dataset:
    """Adds 'time' coordinate to a dataset based on filename or metadata."""
    try:
        if 'EPCT_start_sensing_time' in dataset.attrs:
            my_date_string = dataset.attrs['EPCT_start_sensing_time']
            date_xr = datetime.strptime(my_date_string, '%Y%m%dT%H%M%SZ')
            date_xr = pd.to_datetime(date_xr)
        else:
            source_name = dataset.encoding.get("source", "")
            string_time = os.path.basename(source_name)
            pattern = r'(\d{8})(\d{6})'
            match = re.search(pattern, string_time)
            if match:
                start_date = match.group(1)
                start_time = match.group(2)
                date_xr = pd.to_datetime(start_date + start_time, format='%Y%m%d%H%M%S')
            else:
                logger.warning(f"Could not parse time in: {string_time}")
                return None
        dataset = dataset.assign_coords(time=date_xr).expand_dims("time")
        return dataset
    except Exception as e:
        logger.warning(f"Failed to assign time: {e}")
        return None

def load_clean_xarray_dataset(path_pattern: str) -> xr.Dataset:
    """
    Loads multiple NetCDF files, applies preprocessing, and safely skips corrupt files.

    Parameters:
        path_pattern (str): Glob-style path to NetCDF files, e.g., "../data/*.nc"

    Returns:
        xr.Dataset: Concatenated dataset along the time dimension.
    """
    valid_datasets = []
    filepaths = sorted(glob(path_pattern))

    for path in filepaths:
        try:
            ds = xr.open_dataset(path)
            ds = add_time(ds)
            if ds is not None:
                valid_datasets.append(ds)
            else:
                logger.warning(f"Skipping file due to preprocessing failure: {path}")
        except Exception as e:
            logger.warning(f"Skipping corrupt or unreadable file: {path} — {e}")
            continue

    if valid_datasets:
        return xr.concat(valid_datasets, dim="time", coords="minimal")
    else:
        raise RuntimeError("No valid NetCDF files could be opened.")
    

import xarray as xr
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def convert_mtg_fci_with_offsets(ds: xr.Dataset, selected_channels=["vis_06", "vis_08"]) -> xr.Dataset:
    """
    Convert an MTG FCI Level 1C dataset with body chunks and index offsets
    into a regular (time, channel, lat, lon) xarray.Dataset for selected channels.

    Parameters:
        ds (xr.Dataset): The raw MTG FCI dataset
        selected_channels (list of str): Channels to extract (e.g., ['vis_06', 'vis_08'])

    Returns:
        xr.Dataset with dims: time, channel, lat, lon
    """

    # Step 1: Map channel names to indices
    if "l1c_channels_present" not in ds:
        raise ValueError("Missing 'l1c_channels_present' in dataset.")

    all_channels = ds["l1c_channels_present"].values.astype(str)[0].tolist()
    channel_indices = [i for i, ch in enumerate(all_channels) if ch in selected_channels]
    if not channel_indices:
        raise ValueError(f"Selected channels {selected_channels} not found in dataset.")

    # Step 2: Slice the dataset to keep only selected channels
    ds = ds.isel(number_of_l1c_channels=channel_indices)
    ds = ds.assign_coords(channel=("number_of_l1c_channels", selected_channels))
    ds = ds.drop_dims("number_of_l1c_channels")

    # Step 3: Reconstruct global index
    # body_chunk x index → global flat index (e.g., 111600)
    if "index_offset" not in ds:
        raise ValueError("Missing 'index_offset' for reconstructing global index.")

    offsets = ds["index_offset"]  # shape: (body_chunk)
    body_chunks = ds["body_chunk"]
    n_chunks = ds.dims["body_chunk"]
    n_index = ds.dims["index"]

    global_index = []
    for i in range(n_chunks):
        offset = offsets[i].item()
        global_index.append(np.arange(offset, offset + n_index))

    global_index = np.stack(global_index)  # shape: (body_chunk, index)

    # Step 4: Assign global_index to the dataset
    ds = ds.assign_coords(global_index=(("body_chunk", "index"), global_index))

    # Step 5: Flatten body_chunk and index → 1D spatial dimension
    ds_flat = ds.stack(pixel=("body_chunk", "index")).swap_dims({"pixel": "global_index"}).drop_vars(["body_chunk", "index"])

    # Step 6: Use lat/lon per pixel
    lat = ds_flat["latitude"]
    lon = ds_flat["longitude"]
    ds_flat = ds_flat.assign_coords(lat=("global_index", lat), lon=("global_index", lon))

    # Step 7: Try reshaping into 2D grid (assumes regular grid)
    n = ds_flat.dims["global_index"]
    try:
        height = int(np.sqrt(n))
        width = n // height
        if height * width != n:
            raise ValueError("Cannot reshape global_index into square grid.")
    except Exception as e:
        raise ValueError("Cannot infer spatial grid shape.") from e

    # Step 8: Reshape variables
    data_vars = {}
    for var in ds_flat.data_vars:
        if set(ds_flat[var].dims) >= {"time", "channel", "global_index"}:
            reshaped = ds_flat[var].data.reshape((ds.dims["time"], len(selected_channels), height, width))
            data_vars[var] = (("time", "channel", "lat", "lon"), reshaped)

    # Reshape lat/lon
    lat2d = lat.data.reshape((height, width))
    lon2d = lon.data.reshape((height, width))

    # Step 9: Final dataset
    ds_out = xr.Dataset(
        data_vars=data_vars,
        coords=dict(
            time=ds["time"],
            channel=("channel", selected_channels),
            lat=(("lat", "lon"), lat2d),
            lon=(("lat", "lon"), lon2d),
        )
    )

    return ds_out

def compute_ndvi(dataset:xr.Dataset, channel_1:str, channel_2:str) -> xr.Dataset:
    return dataset.assign(ndvi=(
        dataset[channel_2] - dataset[channel_1]) / (dataset[channel_2] + dataset[channel_1]))


def clean_outliers(dataset:xr.Dataset, var="ndvi"):
    ds = dataset.where((dataset[var]<=1) & (dataset[var]>=-1))
    return ds.dropna(dim="lon", how="all")


def ndvi_colormap(colormap: Literal["diverging","sequential"]):
    from  matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap


    if colormap == "diverging":

        # List of corresponding colors in hexadecimal format (reversed order)
        cols = [
            "#c0c0c0",
            "#954535",
            "#FF0000",
            "#E97451",
            "#FFA500",
            "#FFD700",
            "#DFFF00",
            "#CCFF00",
            "#00FF00",
            "#00BB00",
            "#008800",
            "#006600",
            "#7F00FF"
        ]

    elif colormap == "sequential":
        cols = ["#ffffe5","#f7fcb9","#d9f0a3","#addd8e","#78c679","#41ab5d",
                "#238443","#006837","#004529"]

    cmap_custom = ListedColormap(cols)
    return cmap_custom