"""
MajorTOM-format sparse Zarr export for FCI geostationary data.

Output structure (mirrors modis_majortom convention):

    <label>.zarr/
      patches/
        <channel>/          e.g. vis_06
          <timestamp>/      e.g. 2025-08-01T12:00:00
            <grid_id>/      e.g. 0266_0764
              data          zarr.Array shape (patch_size, patch_size)

No resampling to a regular lat/lon grid is performed; patches are extracted
from the native FCI scene in its native geostationary projection.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Union

import numpy as np
import zarr
from pydantic import PositiveInt
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class MajorTomZarrExport:
    """
    Download FCI data and write patches to a sparse MajorTOM Zarr store.

    Parameters
    ----------
    args :
        Namespace from argparse (requires ``args.threading``, ``args.yes``,
        ``args.remove``, ``args.resampler``).
    downloader :
        An initialised and authenticated ``EUMDownloader`` instance.
    label :
        Output filename stem — the zarr will be ``{output_dir}/{label}.zarr``.
    channels :
        List of FCI channel names to extract.
    patch_size :
        Half-side of each square patch in pixels.  Full patch = 2 * patch_size.
    max_null_fraction :
        Reject patches with more than this fraction of NaN pixels.
    processes :
        Threads used for parallel file downloads.
    grid_path :
        Optional path to a local MajorTOM grid parquet/geojson file.
    bbox_filter :
        ``(lat_min, lat_max, lon_min, lon_max)`` to restrict grid cells.
    """

    def __init__(
        self,
        args,
        downloader,
        label: str,
        channels: list[str] = ("vis_06", "vis_08"),
        patch_size: int = 64,
        max_null_fraction: float = 0.5,
        processes: PositiveInt = 4,
        grid_path: str | Path | None = None,
        spacing_km: float = 100.0,
        bbox_filter: tuple[float, float, float, float] | None = None,
        preextracted_natfolders: list[str] | None = None,
        dtype: str = "float32",
        zarr_path: str | Path | None = None,
    ):
        """
        Parameters
        ----------
        grid_path :
            Path to an external MajorTOM grid file.  When None (default) a
            regular lat/lon grid is generated at ``spacing_km`` spacing.
        spacing_km :
            Cell spacing for the generated grid (default 100 km).  Ignored
            when ``grid_path`` is provided.
        zarr_path :
            Override the output zarr path.  When provided, the store is
            opened in append mode so multiple tiles can write into the same
            file sequentially.  When None (default) the path is derived from
            ``output_dir / label_majortom.zarr``.
        """
        from eumetsearch.transform import FCIMajorTomGrid

        self.output_dir = downloader.output_dir
        self.channels = list(channels)
        self.patch_size = patch_size
        self.max_null_fraction = max_null_fraction
        self.processes = processes
        self._satpy_reader = getattr(downloader, "satpy_reader", "fci_l1c_nc")
        self._calibration = getattr(downloader, "calibration", "reflectance")
        self._threading = args.threading
        self.dtype = np.dtype(dtype)

        self.zip_path = Path(self.output_dir) / "disk/Data" / "zipfolder"
        self.nat_path = Path(self.output_dir) / "disk/Data" / "natfolder"
        self.zip_path.mkdir(parents=True, exist_ok=True)
        self.nat_path.mkdir(parents=True, exist_ok=True)

        self._dl_lock = Lock()

        if zarr_path is not None:
            self.zarr_path = str(zarr_path)
        else:
            self.zarr_path = str(Path(self.output_dir) / f"{label}_majortom.zarr")

        # Build grid
        bbox_kw = {}
        if bbox_filter is not None:
            lat_min, lat_max, lon_min, lon_max = bbox_filter
            bbox_kw = dict(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)
        self.grid = FCIMajorTomGrid(grid_path=grid_path, spacing_km=spacing_km, **bbox_kw)

        if preextracted_natfolders is not None:
            # Skip download: NC files are already on disk from a prior ZarrExport run.
            logger.info(
                f"MajorTOM export (pre-extracted): processing {len(preextracted_natfolders)} folders"
            )
            self._run_from_natfolders(preextracted_natfolders)
        else:
            from definitions import DATA_PATH
            downloader.initiate_download(
                aggregated_file=Path(DATA_PATH) / "datastore_data" / "aggregated_data.json"
            )
            self.file_list = downloader.file_list
            if self.file_list:
                logger.info(f"MajorTOM export: {len(self.file_list)} files to process")
                self._run(args, label)
            else:
                logger.info("All files already processed, nothing to do.")

    # ------------------------------------------------------------------
    # Zarr helpers
    # ------------------------------------------------------------------

    def _init_or_open_zarr(self) -> zarr.Group:
        """Return the root patches group, creating the store if needed."""
        root = zarr.open_group(self.zarr_path, mode="a")
        root.require_group("patches")
        return root["patches"]

    def _write_patch(
        self,
        date_grp: zarr.Group,
        grid_id: str,
        patch: np.ndarray,
    ) -> None:
        """Write one patch into a pre-opened channel/timestamp group."""
        safe_id = grid_id.replace("/", "_").replace("\\", "_")
        if safe_id not in date_grp:
            date_grp.create_dataset(safe_id, data=patch, chunks=patch.shape)

    # ------------------------------------------------------------------
    # Scene reading
    # ------------------------------------------------------------------

    def _load_scene(self, natfolder: str):
        """Load a satpy Scene from the extracted netCDF folder."""
        import warnings
        from satpy import find_files_and_readers
        from satpy.scene import Scene

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

        files = find_files_and_readers(
            base_dir=natfolder, reader=self._satpy_reader, missing_ok=True
        )
        scn = Scene(filenames=files)
        load_kwargs = {}
        if self._calibration:
            load_kwargs["calibration"] = self._calibration
        scn.load(self.channels, **load_kwargs)
        return scn

    def _scene_timestamp(self, scn) -> str:
        """Return ISO timestamp string from the scene metadata."""
        t = scn[self.channels[0]].attrs["time_parameters"]["nominal_start_time"]
        if hasattr(t, "isoformat"):
            return t.isoformat()
        return str(t)

    # ------------------------------------------------------------------
    # Patch extraction for one scene
    # ------------------------------------------------------------------

    def _process_scene(self, scn, patches_grp: zarr.Group) -> int:
        """Extract patches for all MajorTOM cells visible in this scene."""
        import dask
        from eumetsearch.transform import extract_fci_patch, latlon_to_fci_pixel

        date_str = self._scene_timestamp(scn)
        area = scn[self.channels[0]].area
        n_written = 0

        # Compute all channels once with a synchronous scheduler.
        # netCDF4/HDF5 is not thread-safe, so the default threaded Dask
        # scheduler causes "NetCDF: HDF error" on multi-chunk FCI scenes.
        with dask.config.set(scheduler="synchronous"):
            channel_data = {ch: scn[ch].values for ch in self.channels}

        # Pre-create channel/timestamp groups once per scene so the cell loop
        # never calls require_group (which hits the filesystem each time).
        date_grps = {
            ch: patches_grp.require_group(ch).require_group(date_str)
            for ch in self.channels
        }

        for grid_id, lat, lon in self.grid.cells_in_area(area):
            row, col = latlon_to_fci_pixel(area, lat, lon)
            if row is None:
                continue

            patches_for_cell: dict[str, np.ndarray] = {}
            for ch in self.channels:
                patch = extract_fci_patch(
                    channel_data[ch], row, col, self.patch_size, self.max_null_fraction
                )
                if patch is None:
                    break
                patches_for_cell[ch] = patch.astype(self.dtype)
            else:
                # All channels yielded valid patches → write them
                for ch, patch in patches_for_cell.items():
                    self._write_patch(date_grps[ch], grid_id, patch)
                n_written += 1

        logger.info(f"  [{date_str}] wrote {n_written} patches")
        return n_written

    # ------------------------------------------------------------------
    # Main download → extract → write pipeline
    # ------------------------------------------------------------------

    def _download_and_extract(self, product, t: int) -> str | None:
        """Download zip, extract NC files, return path to nat folder."""
        import shutil
        import zipfile

        dsnm = product.metadata["properties"]["title"]
        outfile = self.zip_path / (dsnm + ".zip")
        dssz = product.metadata["properties"]["productInformation"]["size"]

        if not outfile.exists() or outfile.stat().st_size / 1000 <= dssz:
            from http.client import IncompleteRead
            from urllib3.exceptions import ProtocolError

            for attempt in range(5):
                try:
                    with product.open() as fsrc, open(outfile, "wb") as fdst:
                        shutil.copyfileobj(fsrc, fdst)
                    break
                except (IncompleteRead, OSError, ProtocolError) as e:
                    logger.warning(f"Download attempt {attempt + 1}/5 failed: {e}")
                    time.sleep(5 * (attempt + 1))
            else:
                logger.error(f"Failed to download {dsnm} after 5 attempts")
                return None

        natfolder_t = self.nat_path / str(t)
        natfolder_t.mkdir(exist_ok=True)

        if not list(natfolder_t.glob("*.nc")):
            with zipfile.ZipFile(outfile) as zf:
                for name in zf.namelist():
                    if name.endswith(".nc"):
                        zf.extract(name, natfolder_t)

        return str(natfolder_t)

    def _run_from_natfolders(self, natfolders: list[str]) -> None:
        """Process pre-extracted NC folders directly, skipping download."""
        patches_grp = self._init_or_open_zarr()
        total_patches = 0
        for natfolder in tqdm(natfolders, desc="Extracting MajorTOM patches"):
            if not os.path.isdir(natfolder):
                logger.warning(f"Natfolder not found, skipping: {natfolder}")
                continue
            try:
                scn = self._load_scene(natfolder)
                n = self._process_scene(scn, patches_grp)
                total_patches += n
            except Exception as e:
                logger.error(f"Error processing {natfolder}: {e}")
        zarr.consolidate_metadata(self.zarr_path)
        logger.info(f"MajorTOM export done: {total_patches} cell-time patches written to {self.zarr_path}")

    def _run(self, args, label: str) -> None:
        """Main pipeline: parallel download → sequential scene processing."""
        file_list = sorted(
            self.file_list,
            key=lambda p: p._browse_properties["date"].split("/")[0],
        )

        patches_grp = self._init_or_open_zarr()

        # Parallel downloads
        natfolders = [None] * len(file_list)
        with ThreadPoolExecutor(max_workers=self.processes) as ex:
            futures = {
                ex.submit(self._download_and_extract, file_list[i], i): i
                for i in range(len(file_list))
            }
            for f in tqdm(as_completed(futures), total=len(file_list), desc="Downloading"):
                i = futures[f]
                try:
                    natfolders[i] = f.result()
                except Exception as e:
                    logger.error(f"Error downloading file {i}: {e}")

        # Sequential scene processing (satpy/pyproj are not fork-safe)
        total_patches = 0
        for i, natfolder in enumerate(
            tqdm(natfolders, desc="Extracting MajorTOM patches")
        ):
            if natfolder is None:
                continue
            try:
                scn = self._load_scene(natfolder)
                n = self._process_scene(scn, patches_grp)
                total_patches += n
            except Exception as e:
                logger.error(f"Error processing file {i}: {e}")

        # Consolidate metadata for fast xr.open_zarr / zarr.open
        zarr.consolidate_metadata(self.zarr_path)
        logger.info(
            f"MajorTOM export done: {total_patches} cell-time patches written to {self.zarr_path}"
        )


def merge_majortom_zarrs(
    tile_zarr_paths: list[str | Path],
    output_path: str | Path,
) -> None:
    """
    Merge multiple per-tile MajorTOM sparse Zarr stores into a single store.

    Patches from each tile are copied without re-downloading or re-reading
    raw data.  If the same grid_id/timestamp already exists in the output
    (e.g. from a previous partial run), it is skipped.

    Parameters
    ----------
    tile_zarr_paths :
        Ordered list of per-tile `*_majortom.zarr` paths.
    output_path :
        Destination zarr path for the consolidated store.
    """
    output_path = str(output_path)
    root_out = zarr.open_group(output_path, mode="a")
    patches_out = root_out.require_group("patches")

    total_copied = 0
    total_skipped = 0

    for tile_path in tqdm(tile_zarr_paths, desc="Merging tile zarrs"):
        tile_path = str(tile_path)
        if not os.path.exists(tile_path):
            logger.warning(f"Tile zarr not found, skipping: {tile_path}")
            continue
        root_in = zarr.open_group(tile_path, mode="r")
        if "patches" not in root_in:
            logger.warning(f"No 'patches' group in {tile_path}, skipping")
            continue
        patches_in = root_in["patches"]

        for channel in patches_in:
            ch_out = patches_out.require_group(channel)
            ch_in = patches_in[channel]
            for timestamp in ch_in:
                ts_out = ch_out.require_group(timestamp)
                ts_in = ch_in[timestamp]
                for grid_id in ts_in:
                    if grid_id in ts_out:
                        total_skipped += 1
                        continue
                    arr = ts_in[grid_id][:]
                    ts_out.require_dataset(
                        grid_id,
                        shape=arr.shape,
                        dtype=arr.dtype,
                        chunks=arr.shape,
                        data=arr,
                    )
                    total_copied += 1

    zarr.consolidate_metadata(output_path)
    logger.info(
        f"Merge complete: {total_copied} patches copied, "
        f"{total_skipped} skipped (already present). "
        f"Output: {output_path}"
    )
