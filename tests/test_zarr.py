"""
Tests for eumetsearch.utils.zarr:
  ZarrStore, extend_time_dim, add_channels_to_zarr, compute_auto_chunks
"""

import os
import math

import numpy as np
import pytest
import zarr
import xarray as xr

from eumetsearch.utils.zarr import (
    ZarrStore,
    add_channels_to_zarr,
    compute_auto_chunks,
    extend_time_dim,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(tmp_path, channels, num_time=5, size=(50, 50), label="test",
                custom_size=None, remove_flag=True, chunks=None):
    kwargs = dict(
        folder_path=str(tmp_path),
        size=list(size),
        file_list=[None] * num_time,
        channels=channels,
        label=label,
        remove_flag=remove_flag,
    )
    if custom_size is not None:
        kwargs["custom_size"] = custom_size
    if chunks is not None:
        kwargs["chunks"] = chunks
    return ZarrStore(**kwargs)


def _zarr_path(tmp_path, label):
    return str(tmp_path / f"MTG_FCI_{label}.zarr")


# ---------------------------------------------------------------------------
# ZarrStore creation
# ---------------------------------------------------------------------------

class TestZarrStoreCreate:
    def test_creates_store_with_correct_shape(self, tmp_path):
        channels = ["vis_06", "vis_08"]
        num_time, h, w = 5, 50, 50
        store = _make_store(tmp_path, channels, num_time=num_time, size=(h, w))
        z = zarr.open(store.path, mode="r")
        assert z["vis_06"].shape == (num_time, h, w)
        assert z["vis_08"].shape == (num_time, h, w)

    def test_time_coord_all_nat(self, tmp_path):
        store = _make_store(tmp_path, ["vis_06"], num_time=3)
        ds = xr.open_zarr(store.path)
        assert np.all(np.isnat(ds["time"].values))

    def test_filled_flag_all_false(self, tmp_path):
        store = _make_store(tmp_path, ["vis_06"], num_time=4)
        z = zarr.open(store.path, mode="r")
        assert z["filled_flag"].shape == (4,)
        assert not np.any(z["filled_flag"][:])

    def test_vis_channel_dtype_float32(self, tmp_path):
        store = _make_store(tmp_path, ["vis_06", "nir_13"])
        z = zarr.open(store.path, mode="r")
        assert z["vis_06"].dtype == np.float32
        assert z["nir_13"].dtype == np.float32

    def test_ir_channel_dtype_int32(self, tmp_path):
        store = _make_store(tmp_path, ["ir_105"])
        z = zarr.open(store.path, mode="r")
        assert z["ir_105"].dtype == np.int32

    def test_custom_size_overrides_file_list_length(self, tmp_path):
        store = _make_store(tmp_path, ["vis_06"], num_time=3, custom_size={"time": 10})
        z = zarr.open(store.path, mode="r")
        assert z["vis_06"].shape[0] == 10

    def test_invalid_channel_raises_assertion(self, tmp_path):
        with pytest.raises(AssertionError):
            _make_store(tmp_path, ["not_a_real_channel"])

    def test_existing_store_no_flag_extends_time(self, tmp_path):
        # First creation
        _make_store(tmp_path, ["vis_06"], num_time=3, label="ext", remove_flag=True)
        # Second call requesting more time — should extend rather than recreate
        _make_store(tmp_path, ["vis_06"], num_time=5, label="ext", remove_flag=False)
        z = zarr.open(_zarr_path(tmp_path, "ext"), mode="r")
        assert z["vis_06"].shape[0] == 5

    def test_existing_store_no_flag_adds_new_channels(self, tmp_path):
        # Create with vis_06 only
        _make_store(tmp_path, ["vis_06"], num_time=3, label="addch", remove_flag=True)
        # Reopen requesting vis_06 + ir_105 — ir_105 should be added
        _make_store(tmp_path, ["vis_06", "ir_105"], num_time=3, label="addch",
                    remove_flag=False)
        z = zarr.open(_zarr_path(tmp_path, "addch"), mode="r")
        assert "ir_105" in z
        assert "vis_06" in z


# ---------------------------------------------------------------------------
# extend_time_dim
# ---------------------------------------------------------------------------

class TestExtendTimeDim:
    def test_time_axis_grows(self, tmp_path):
        store = _make_store(tmp_path, ["vis_06"], num_time=3)
        extend_time_dim(store.path, n_new=2)
        z = zarr.open(store.path, mode="r")
        assert z["vis_06"].shape[0] == 5

    def test_existing_data_preserved(self, tmp_path):
        store = _make_store(tmp_path, ["vis_06"], num_time=3)
        z = zarr.open(store.path, mode="a")
        z["vis_06"][0, :, :] = 42.0
        z.store.close()

        extend_time_dim(store.path, n_new=2)

        z = zarr.open(store.path, mode="r")
        assert z["vis_06"].shape[0] == 5
        assert float(z["vis_06"][0, 0, 0]) == pytest.approx(42.0)

    def test_all_arrays_extended(self, tmp_path):
        store = _make_store(tmp_path, ["vis_06", "ir_105"], num_time=3)
        extend_time_dim(store.path, n_new=4)
        z = zarr.open(store.path, mode="r")
        assert z["vis_06"].shape[0] == 7
        assert z["ir_105"].shape[0] == 7
        assert z["filled_flag"].shape[0] == 7

    def test_filled_flag_new_slots_false(self, tmp_path):
        store = _make_store(tmp_path, ["vis_06"], num_time=2)
        # Mark first slot
        z = zarr.open(store.path, mode="a")
        z["filled_flag"][0] = True
        z.store.close()

        extend_time_dim(store.path, n_new=3)
        z = zarr.open(store.path, mode="r")
        flags = z["filled_flag"][:]
        assert bool(flags[0]) is True
        assert not np.any(flags[1:])


# ---------------------------------------------------------------------------
# add_channels_to_zarr
# ---------------------------------------------------------------------------

class TestAddChannelsToZarr:
    def test_new_channel_added(self, tmp_path):
        store = _make_store(tmp_path, ["vis_06"], num_time=3, size=(40, 40))
        add_channels_to_zarr(store.path, ["ir_105"], time_size=3,
                             height=40, width=40)
        z = zarr.open(store.path, mode="r")
        assert "ir_105" in z
        assert z["ir_105"].shape == (3, 40, 40)

    def test_new_channel_dtype_float32_for_vis(self, tmp_path):
        store = _make_store(tmp_path, ["vis_06"], num_time=3, size=(40, 40))
        add_channels_to_zarr(store.path, ["vis_08"], time_size=3,
                             height=40, width=40)
        z = zarr.open(store.path, mode="r")
        assert z["vis_08"].dtype == np.float32

    def test_new_channel_dtype_int32_for_ir(self, tmp_path):
        store = _make_store(tmp_path, ["vis_06"], num_time=3, size=(40, 40))
        add_channels_to_zarr(store.path, ["ir_87"], time_size=3,
                             height=40, width=40)
        z = zarr.open(store.path, mode="r")
        assert z["ir_87"].dtype == np.int32

    def test_existing_channels_unchanged(self, tmp_path):
        store = _make_store(tmp_path, ["vis_06"], num_time=3, size=(40, 40))
        z = zarr.open(store.path, mode="a")
        z["vis_06"][1, 2, 3] = 0.77
        z.store.close()

        add_channels_to_zarr(store.path, ["ir_105"], time_size=3,
                             height=40, width=40)
        z = zarr.open(store.path, mode="r")
        assert float(z["vis_06"][1, 2, 3]) == pytest.approx(0.77)

    def test_skips_already_present_channel(self, tmp_path):
        store = _make_store(tmp_path, ["vis_06"], num_time=3, size=(40, 40))
        z = zarr.open(store.path, mode="a")
        z["vis_06"][:] = 1.0
        z.store.close()

        # Adding vis_06 again should be a no-op (not raise)
        add_channels_to_zarr(store.path, ["vis_06"], time_size=3,
                             height=40, width=40)
        z = zarr.open(store.path, mode="r")
        assert float(z["vis_06"][0, 0, 0]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_auto_chunks
# ---------------------------------------------------------------------------

class TestComputeAutoChunks:
    def test_returns_correct_keys(self):
        shape = {"time": 100, "lat": 1000, "lon": 1000}
        chunks = compute_auto_chunks(shape)
        assert set(chunks.keys()) == {"time", "lat", "lon"}

    def test_time_chunk_is_fixed(self):
        shape = {"time": 100, "lat": 1000, "lon": 1000}
        chunks = compute_auto_chunks(shape, fixed_chunks={"time": 2})
        assert chunks["time"] == 2

    def test_lat_lon_chunks_do_not_exceed_dims(self):
        shape = {"time": 10, "lat": 200, "lon": 300}
        chunks = compute_auto_chunks(shape)
        assert 1 <= chunks["lat"] <= 200
        assert 1 <= chunks["lon"] <= 300

    def test_chunk_volume_near_target(self):
        shape = {"time": 1, "lat": 1000, "lon": 1000}
        target = 64 * 2**20  # 64 MiB
        chunks = compute_auto_chunks(shape, target_chunk_bytes=target)
        vol = chunks["time"] * chunks["lat"] * chunks["lon"] * 4  # float32
        # Within 2x of target (auto-sizing is approximate)
        assert vol <= target * 2

    def test_raises_with_wrong_number_of_auto_dims(self):
        # Only one free dimension — not supported
        shape = {"time": 100, "lat": 1000}
        with pytest.raises(ValueError):
            compute_auto_chunks(shape, fixed_chunks={"time": 1})
