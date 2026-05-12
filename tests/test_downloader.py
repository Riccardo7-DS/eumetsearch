"""
Tests for eumetsearch.data_collection.downloader:
  products_list structure, channel lists,
  ZarrExport._extract_new_index, ZarrExport._safe_write_to_zarr,
  JsonDataResponse.files_exclusion
"""

import json
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import zarr
import xarray as xr

from eumetsearch.data_collection.downloader import (
    FCI_CHANNELS_ALL,
    FCI_CHANNELS_IR,
    FCI_CHANNELS_VIS_NIR,
    FCI_HR_CHANNELS,
    SEVIRI_CHANNELS_ALL,
    SEVIRI_CHANNELS_HRV,
    SEVIRI_CHANNELS_VIS_NIR,
    ZarrExport,
    products_list,
)
from eumetsearch.utils.general import JsonDataResponse


# ---------------------------------------------------------------------------
# products_list structure
# ---------------------------------------------------------------------------

REQUIRED_PRODUCT_KEYS = {"product_id", "product_name", "satpy_reader", "bands", "calibration"}
EXPECTED_PRODUCTS = {
    "MTG-FCI-L1C-FDHSI",
    "MTG-FCI-L1C-HRFI",
    "MTG-FCI-L2-CLM",
    "MTG-FCI-L2-AFM",
    "MTG-FCI-L2-OCA",
    "MSG-SEVIRI-L1-HRY",
    "MSG-SEVIRI-L1-IODC",
    "MSG-SEVIRI-RSS",
}


class TestProductsList:
    def test_all_expected_products_present(self):
        assert set(products_list.keys()) == EXPECTED_PRODUCTS

    def test_each_product_has_required_keys(self):
        for name, meta in products_list.items():
            assert REQUIRED_PRODUCT_KEYS == set(meta.keys()), f"Missing keys in {name}"

    def test_product_ids_are_nonempty_strings(self):
        for name, meta in products_list.items():
            assert isinstance(meta["product_id"], str) and meta["product_id"], \
                f"Empty product_id for {name}"

    def test_satpy_readers_are_valid(self):
        valid_readers = {"fci_l1c_nc", "fci_l2_nc", "seviri_l1b_hrit"}
        for name, meta in products_list.items():
            assert meta["satpy_reader"] in valid_readers, \
                f"Unknown reader {meta['satpy_reader']!r} for {name}"

    def test_l2_products_have_no_calibration(self):
        l2_products = {"MTG-FCI-L2-CLM", "MTG-FCI-L2-AFM", "MTG-FCI-L2-OCA"}
        for name in l2_products:
            assert products_list[name]["calibration"] is None

    def test_fdhsi_bands_include_all_fci_channels(self):
        bands = products_list["MTG-FCI-L1C-FDHSI"]["bands"]
        assert set(FCI_CHANNELS_ALL).issubset(set(bands))


# ---------------------------------------------------------------------------
# Channel lists
# ---------------------------------------------------------------------------

class TestChannelLists:
    def test_fci_channels_all_has_16(self):
        assert len(FCI_CHANNELS_ALL) == 16

    def test_fci_channels_all_is_vis_nir_plus_ir(self):
        assert FCI_CHANNELS_ALL == FCI_CHANNELS_VIS_NIR + FCI_CHANNELS_IR

    def test_fci_vis_nir_has_8(self):
        assert len(FCI_CHANNELS_VIS_NIR) == 8

    def test_fci_ir_has_8(self):
        assert len(FCI_CHANNELS_IR) == 8

    def test_seviri_channels_hrv_has_12(self):
        assert len(SEVIRI_CHANNELS_HRV) == 12

    def test_seviri_channels_all_has_11(self):
        assert len(SEVIRI_CHANNELS_ALL) == 11

    def test_seviri_hrv_includes_hrv(self):
        assert "HRV" in SEVIRI_CHANNELS_HRV

    def test_no_duplicates_in_fci_all(self):
        assert len(FCI_CHANNELS_ALL) == len(set(FCI_CHANNELS_ALL))

    def test_no_duplicates_in_seviri_hrv(self):
        assert len(SEVIRI_CHANNELS_HRV) == len(set(SEVIRI_CHANNELS_HRV))


# ---------------------------------------------------------------------------
# ZarrExport._extract_new_index
# ---------------------------------------------------------------------------

def _make_export_shell():
    """Return a ZarrExport instance bypassing __init__ for method testing."""
    obj = object.__new__(ZarrExport)
    return obj


class TestExtractNewIndex:
    def test_returns_zero_when_no_file(self, tmp_path):
        missing = tmp_path / "nofile.status.json"
        obj = _make_export_shell()
        assert obj._extract_new_index(missing) == 0

    def test_returns_zero_when_empty_file(self, tmp_path):
        f = tmp_path / "empty.status.json"
        f.write_text("")
        obj = _make_export_shell()
        assert obj._extract_new_index(f) == 0

    def test_next_after_single_done(self, tmp_path):
        f = tmp_path / "s.status.json"
        f.write_text(json.dumps({"somekey_3": "done"}))
        obj = _make_export_shell()
        assert obj._extract_new_index(f) == 4

    def test_next_after_multiple_done(self, tmp_path):
        f = tmp_path / "s.status.json"
        status = {"ts_0": "done", "ts_1": "done", "ts_2": "done"}
        f.write_text(json.dumps(status))
        obj = _make_export_shell()
        assert obj._extract_new_index(f) == 3

    def test_check_missing_finds_gap(self, tmp_path):
        f = tmp_path / "s.status.json"
        # 0 and 2 done, 1 missing
        status = {"ts_0": "done", "ts_2": "done"}
        f.write_text(json.dumps(status))
        obj = _make_export_shell()
        assert obj._extract_new_index(f, check_missing=True) == 1

    def test_ignores_non_done_entries(self, tmp_path):
        f = tmp_path / "s.status.json"
        status = {"ts_0": "done", "ts_1": "failed"}
        f.write_text(json.dumps(status))
        obj = _make_export_shell()
        assert obj._extract_new_index(f) == 1


# ---------------------------------------------------------------------------
# ZarrExport._safe_write_to_zarr
# ---------------------------------------------------------------------------

def _minimal_zarr_store(tmp_path, num_time=3, size=(20, 20)):
    """Create a minimal zarr store manually to avoid ZarrStore dependency."""
    h, w = size
    path = str(tmp_path / "test_write.zarr")
    time_coord = np.full(num_time, np.datetime64("NaT", "ns"))

    ds = xr.Dataset(
        {
            "vis_06": (("time", "lat", "lon"),
                       np.full((num_time, h, w), np.nan, dtype=np.float32)),
            "filled_flag": (("time",), np.zeros(num_time, dtype=bool)),
        },
        coords={"time": time_coord},
    )
    ds.to_zarr(path, mode="w", consolidated=True)
    return path


class TestSafeWriteToZarr:
    def test_stamps_time_coordinate(self, tmp_path):
        zarr_path = _minimal_zarr_store(tmp_path)
        obj = _make_export_shell()
        ts = np.datetime64("2025-08-01T12:00:00", "ns")
        ds_new = xr.Dataset(
            {"vis_06": (("time", "lat", "lon"), np.ones((1, 20, 20), dtype=np.float32))},
            coords={"time": [ts]},
        )
        obj._safe_write_to_zarr(ds_new, zarr_path, t=0)

        z = zarr.open(zarr_path, mode="r")
        stored = np.datetime64(int(z["time"][0]), "ns")
        assert stored == ts

    def test_sets_filled_flag_true(self, tmp_path):
        zarr_path = _minimal_zarr_store(tmp_path)
        obj = _make_export_shell()
        ts = np.datetime64("2025-08-01T12:00:00", "ns")
        ds_new = xr.Dataset(
            {"vis_06": (("time", "lat", "lon"), np.ones((1, 20, 20), dtype=np.float32))},
            coords={"time": [ts]},
        )
        obj._safe_write_to_zarr(ds_new, zarr_path, t=0)

        z = zarr.open(zarr_path, mode="r")
        assert bool(z["filled_flag"][0]) is True

    def test_writes_data_values(self, tmp_path):
        zarr_path = _minimal_zarr_store(tmp_path)
        obj = _make_export_shell()
        ts = np.datetime64("2025-08-01T12:00:00", "ns")
        data = np.full((1, 20, 20), 0.75, dtype=np.float32)
        ds_new = xr.Dataset(
            {"vis_06": (("time", "lat", "lon"), data)},
            coords={"time": [ts]},
        )
        obj._safe_write_to_zarr(ds_new, zarr_path, t=1)

        z = zarr.open(zarr_path, mode="r")
        assert float(z["vis_06"][1, 5, 5]) == pytest.approx(0.75)

    def test_other_slots_untouched(self, tmp_path):
        zarr_path = _minimal_zarr_store(tmp_path)
        obj = _make_export_shell()
        ts = np.datetime64("2025-08-01T12:00:00", "ns")
        ds_new = xr.Dataset(
            {"vis_06": (("time", "lat", "lon"), np.ones((1, 20, 20), dtype=np.float32))},
            coords={"time": [ts]},
        )
        obj._safe_write_to_zarr(ds_new, zarr_path, t=0)

        z = zarr.open(zarr_path, mode="r")
        # Slots 1 and 2 should still be NaN / flag=False
        assert np.all(np.isnan(z["vis_06"][1]))
        assert not bool(z["filled_flag"][1])

    def test_raises_if_no_time_coord(self, tmp_path):
        zarr_path = _minimal_zarr_store(tmp_path)
        obj = _make_export_shell()
        ds_bad = xr.Dataset(
            {"vis_06": (("lat", "lon"), np.ones((20, 20), dtype=np.float32))}
        )
        # Should not raise — _safe_write_to_zarr catches and returns None
        result = obj._safe_write_to_zarr(ds_bad, zarr_path, t=0)
        assert result is None


# ---------------------------------------------------------------------------
# JsonDataResponse.files_exclusion
# ---------------------------------------------------------------------------

class TestJsonDataResponse:
    def test_excludes_done_timestamps(self, tmp_path):
        status = {
            "2025-07-21T09:00:07.000000000_0": "done",
            "2025-07-21T10:00:05.000000000_1": "done",
        }
        f = tmp_path / "agg.json"
        f.write_text(json.dumps(status))

        jdr = JsonDataResponse(f)
        # Build a file list where first entry is the datetime
        dt_done = datetime(2025, 7, 21, 9, 0)   # truncated to minute
        dt_keep = datetime(2025, 7, 21, 11, 0)
        my_files = [[dt_done, "file_a"], [dt_keep, "file_b"]]

        remaining = jdr.files_exclusion(my_files)
        assert len(remaining) == 1
        assert remaining[0][1] == "file_b"

    def test_keeps_all_when_no_done(self, tmp_path):
        status = {
            "2025-07-21T09:00:00.000000000_0": "failed",
        }
        f = tmp_path / "agg.json"
        f.write_text(json.dumps(status))

        jdr = JsonDataResponse(f)
        my_files = [[datetime(2025, 7, 21, 9, 0), "file_a"]]
        remaining = jdr.files_exclusion(my_files)
        assert len(remaining) == 1

    def test_returns_empty_list_when_all_done(self, tmp_path):
        status = {
            "2025-07-21T09:00:00.000000000_0": "done",
        }
        f = tmp_path / "agg.json"
        f.write_text(json.dumps(status))

        jdr = JsonDataResponse(f)
        my_files = [[datetime(2025, 7, 21, 9, 0), "file_a"]]
        remaining = jdr.files_exclusion(my_files)
        assert remaining == []

    def test_missing_file_returns_empty_status(self, tmp_path):
        f = tmp_path / "missing.json"
        jdr = JsonDataResponse(f)
        assert jdr._status_data == {}

    def test_extract_done_timestamps_parses_correctly(self, tmp_path):
        status = {
            "2025-07-21T09:05:30.123456789_2": "done",
            "2025-07-21T10:00:00.000000000_3": "pending",
        }
        f = tmp_path / "agg.json"
        f.write_text(json.dumps(status))

        jdr = JsonDataResponse(f)
        done = jdr.extract_done_timestamps(jdr._status_data)
        assert datetime(2025, 7, 21, 9, 5) in done
        assert datetime(2025, 7, 21, 10, 0) not in done
