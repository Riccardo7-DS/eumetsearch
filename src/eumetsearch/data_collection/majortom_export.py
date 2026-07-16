"""
MajorTOM-format batched Zarr export for FCI geostationary data.

Output structure:

    <label>.zarr/
      patches/
        <channel>/          e.g. vis_06
          <timestamp>/      e.g. 2025-08-01T12:00:00
            data            zarr.Array  shape (N, patch_H, patch_W)  float16
            grid_ids        zarr.Array  shape (N,)                   unicode

All patches for one channel × timestamp are stored as a single 3-D array,
with a companion ``grid_ids`` array mapping row index → cell identifier.
Chunks are (1, H, W) so individual patches can be accessed efficiently.

When multiple spatial tiles write to the same consolidated zarr, the arrays
are appended along axis 0 (grid cells are non-overlapping between tiles so
no deduplication is needed).
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import numpy as np
import zarr
from pydantic import PositiveInt
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class MajorTomZarrExport:
    """
    Download FCI data and write patches to a batched MajorTOM Zarr store.

    Parameters
    ----------
    args :
        Namespace from argparse (requires ``args.threading``).
    downloader :
        An initialised and authenticated ``EUMDownloader`` instance.
    label :
        Output filename stem.
    channels :
        List of FCI channel names to extract.
    patch_size :
        Half-side of each square patch in pixels.  Full patch = 2 * patch_size.
    max_null_fraction :
        Reject patches with more than this fraction of NaN pixels.
    processes :
        Threads used for parallel file downloads.
    grid_path :
        Path to an external MajorTOM grid file.  When None a regular lat/lon
        grid is generated at ``spacing_km`` spacing.
    spacing_km :
        Cell spacing for the generated grid (default 100 km).
    bbox_filter :
        ``(lat_min, lat_max, lon_min, lon_max)`` to restrict grid cells.
    dtype :
        NumPy dtype for patch data (default ``float32``).
    zarr_path :
        Override output zarr path.  When provided the store is opened in
        append mode so multiple tiles write into the same file.
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
        from eumetsearch.transform import FCIMajorTomGrid

        self.output_dir = downloader.output_dir
        self.channels = list(channels)
        self.patch_size = patch_size
        self.max_null_fraction = max_null_fraction
        self.processes = processes
        self._satpy_reader = getattr(downloader, "satpy_reader", "fci_l1c_nc")
        self._reflectance_bands = getattr(downloader, "reflectance_bands", [])
        self._bt_bands = getattr(downloader, "bt_bands", [])
        self._threading = args.threading
        self.dtype = np.dtype(dtype)

        self.zip_path = Path(self.output_dir) / "disk/Data" / "zipfolder"
        self.nat_path = Path(self.output_dir) / "disk/Data" / "natfolder"
        self.zip_path.mkdir(parents=True, exist_ok=True)
        self.nat_path.mkdir(parents=True, exist_ok=True)

        self._dl_lock = Lock()

        self.zarr_path = str(zarr_path) if zarr_path is not None else str(
            Path(self.output_dir) / f"{label}_majortom.zarr"
        )

        bbox_kw = {}
        if bbox_filter is not None:
            lat_min, lat_max, lon_min, lon_max = bbox_filter
            bbox_kw = dict(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)
        self.grid = FCIMajorTomGrid(grid_path=grid_path, spacing_km=spacing_km, **bbox_kw)

        if preextracted_natfolders is not None:
            logger.info(
                f"MajorTOM export (pre-extracted): processing {len(preextracted_natfolders)} folders"
            )
            self._run_from_natfolders(preextracted_natfolders)
        else:
            from eumetsearch.definitions import DATA_PATH
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

    def _write_scene_batch(
        self,
        patches_grp: zarr.Group,
        date_str: str,
        all_patches: dict[str, list[np.ndarray]],
        all_grid_ids: list[str],
        all_patch_origins: list[tuple[int, int]],
    ) -> None:
        """
        Write all patches for one scene as (N, H, W) arrays.

        If the timestamp group already contains data from a previous tile,
        the new patches are appended along axis 0 (tiles use non-overlapping
        grid cells so no deduplication is needed).

        ``patch_origins`` is stored as an int32 array of shape (N, 2) where
        each row is [row0, col0] — the top-left corner of the patch in the
        FCI full-disk pixel grid (row 0 = north edge, col 0 = west edge).
        These values are required by ``fci_modis_align.align_patches``.
        """
        n = len(all_grid_ids)
        if n == 0:
            return

        H = W = self.patch_size * 2
        grid_ids_arr = np.array(all_grid_ids)                                  # (N,)   unicode
        origins_arr  = np.array(all_patch_origins, dtype=np.int32)             # (N, 2) int32

        # Use large chunks to minimise the number of chunk files written.
        # chunk_n=2048 → ≤14 files per channel/scene for a 100km full-disk grid
        # (~28K patches).  Smaller values allow more granular random access at
        # the cost of more files and slower sequential writes.
        chunk_n = min(n, 2048)

        for ch in self.channels:
            data_arr = np.stack(all_patches[ch]).astype(self.dtype)  # (N, H, W)
            ts_grp = patches_grp.require_group(ch).require_group(date_str)

            if "data" not in ts_grp:
                ts_grp.create_dataset("data", data=data_arr, chunks=(chunk_n, H, W))
                ts_grp.create_dataset("grid_ids",      data=grid_ids_arr, chunks=n)
                ts_grp.create_dataset("patch_origins", data=origins_arr,  chunks=(n, 2))
            else:
                # Append only grid_ids not already stored (guards against
                # double-appending when the same tile is retried after a crash).
                existing_ids = set(ts_grp["grid_ids"][:].tolist())
                new_mask = np.array([gid not in existing_ids for gid in all_grid_ids])
                if not new_mask.any():
                    return
                ts_grp["data"].append(data_arr[new_mask], axis=0)
                ts_grp["grid_ids"].append(grid_ids_arr[new_mask], axis=0)
                if "patch_origins" in ts_grp:
                    ts_grp["patch_origins"].append(origins_arr[new_mask], axis=0)
                else:
                    # Back-fill for stores created before this field was added
                    ts_grp.create_dataset("patch_origins", data=origins_arr[new_mask], chunks=(n, 2))

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

        # Calibration is channel-dependent: VIS/NIR channels must be loaded as
        # reflectance, while IR/WV channels must be loaded as brightness_temperature.
        reflectance_channels = [c for c in self.channels if c in self._reflectance_bands]
        bt_channels = [c for c in self.channels if c in self._bt_bands]
        other_channels = [
            c for c in self.channels
            if c not in reflectance_channels and c not in bt_channels
        ]

        if reflectance_channels:
            scn.load(reflectance_channels, calibration="reflectance")
        if bt_channels:
            scn.load(bt_channels, calibration="brightness_temperature")
        if other_channels:
            scn.load(other_channels)
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
        """
        Extract all valid patches for this scene and write them as a batch.

        Returns the number of cells written.
        """
        import dask
        from eumetsearch.transform import extract_fci_patch

        date_str = self._scene_timestamp(scn)
        area = scn[self.channels[0]].area

        # Vectorised coordinate transform: one pyproj call for all grid cells.
        # This replaces the old per-cell Python loop with individual transforms.
        rows, cols, grid_ids = self.grid.visible_pixel_coords(area)

        # Compute all channels once with a synchronous scheduler.
        # netCDF4/HDF5 is not thread-safe — the default threaded Dask
        # scheduler causes "NetCDF: HDF error" on multi-chunk FCI scenes.
        with dask.config.set(scheduler="synchronous"):
            channel_data = {ch: scn[ch].values for ch in self.channels}

        # Collect valid patches
        all_patches: dict[str, list[np.ndarray]] = {ch: [] for ch in self.channels}
        all_grid_ids: list[str] = []
        all_patch_origins: list[tuple[int, int]] = []

        for row, col, grid_id in zip(rows, cols, grid_ids):
            row_i, col_i = int(row), int(col)
            row0 = row_i - self.patch_size
            col0 = col_i - self.patch_size

            patches_for_cell: dict[str, np.ndarray] = {}
            for ch in self.channels:
                patch = extract_fci_patch(
                    channel_data[ch], row_i, col_i,
                    self.patch_size, self.max_null_fraction,
                )
                if patch is None:
                    break
                patches_for_cell[ch] = patch
            else:
                all_grid_ids.append(grid_id)
                all_patch_origins.append((row0, col0))
                for ch in self.channels:
                    all_patches[ch].append(patches_for_cell[ch])

        self._write_scene_batch(patches_grp, date_str, all_patches, all_grid_ids, all_patch_origins)
        n = len(all_grid_ids)
        logger.info(f"  [{date_str}] wrote {n} patches")
        return n

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
        logger.info(
            f"MajorTOM export done: {total_patches} cell-time patches written to {self.zarr_path}"
        )

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
        for i, natfolder in enumerate(tqdm(natfolders, desc="Extracting MajorTOM patches")):
            if natfolder is None:
                continue
            try:
                scn = self._load_scene(natfolder)
                n = self._process_scene(scn, patches_grp)
                total_patches += n
            except Exception as e:
                logger.error(f"Error processing file {i}: {e}")

        zarr.consolidate_metadata(self.zarr_path)
        logger.info(
            f"MajorTOM export done: {total_patches} cell-time patches written to {self.zarr_path}"
        )


def merge_majortom_zarrs(
    tile_zarr_paths: list[str | Path],
    output_path: str | Path,
) -> None:
    """
    Merge multiple per-tile batched MajorTOM Zarr stores into one store.

    For each (channel, timestamp) the data and grid_ids arrays from all
    tiles are concatenated along axis 0 and deduplicated by grid_id.
    Already-present grid_ids are skipped.
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
                ts_in = ch_in[timestamp]
                if "data" not in ts_in or "grid_ids" not in ts_in:
                    continue

                data_in    = ts_in["data"][:]             # (N, H, W)
                gids_in    = ts_in["grid_ids"][:]         # (N,)
                origins_in = ts_in["patch_origins"][:] if "patch_origins" in ts_in else None

                ts_out = ch_out.require_group(timestamp)
                if "data" not in ts_out:
                    H, W = data_in.shape[1], data_in.shape[2]
                    ts_out.create_dataset("data", data=data_in, chunks=(1, H, W))
                    ts_out.create_dataset("grid_ids", data=gids_in, chunks=len(gids_in))
                    if origins_in is not None:
                        ts_out.create_dataset("patch_origins", data=origins_in,
                                              chunks=(len(gids_in), 2))
                    total_copied += len(gids_in)
                else:
                    existing_ids = set(ts_out["grid_ids"][:].tolist())
                    mask = np.array([g not in existing_ids for g in gids_in])
                    n_new = mask.sum()
                    if n_new == 0:
                        total_skipped += len(gids_in)
                        continue
                    ts_out["data"].append(data_in[mask], axis=0)
                    ts_out["grid_ids"].append(gids_in[mask], axis=0)
                    if origins_in is not None:
                        if "patch_origins" in ts_out:
                            ts_out["patch_origins"].append(origins_in[mask], axis=0)
                        else:
                            ts_out.create_dataset("patch_origins", data=origins_in[mask],
                                                  chunks=(int(n_new), 2))
                    total_copied += int(n_new)
                    total_skipped += int((~mask).sum())

    zarr.consolidate_metadata(output_path)
    logger.info(
        f"Merge complete: {total_copied} patches copied, "
        f"{total_skipped} skipped (already present). "
        f"Output: {output_path}"
    )
