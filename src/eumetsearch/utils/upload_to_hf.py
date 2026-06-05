"""
Upload a local zarr store to HuggingFace as a named subdirectory.

upload_large_folder uploads folder *contents* to the repo root, so we move
the zarr into a temp wrapper dir first:

    _upload_tmp/
        MTG_FCI_mtg_2025_06.zarr/   <- zarr files here
            .zgroup
            patches/
            ...

Uploading _upload_tmp/ then places MTG_FCI_mtg_2025_06.zarr/ at the repo root.

Usage:
    python src/eumetsearch/utils/upload_to_hf.py <local_zarr_path> <hf_folder_name>

Example:
    python src/eumetsearch/utils/upload_to_hf.py data/datastore_data/mtg_june_2025_majortom.zarr MTG_FCI_mtg_2025_06.zarr
"""
import sys
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

HF_REPO      = "riccardode/geo_leo"
HF_REPO_TYPE = "dataset"
NUM_WORKERS  = 4


def upload(local_path: str, hf_name: str):
    local_path = Path(local_path).resolve()
    if not local_path.exists():
        print(f"ERROR: {local_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    zarr_dir = local_path.parent
    tmp_parent = zarr_dir / "_upload_tmp"
    tmp_zarr   = tmp_parent / hf_name

    # Clean up any leftover tmp dir from a previous interrupted run
    if tmp_parent.exists():
        shutil.rmtree(tmp_parent)
    tmp_parent.mkdir()

    # Move zarr into the wrapper (rename — instant, no data copy)
    # Handle case where zarr is already named hf_name from a prior run
    if local_path.name == hf_name:
        src = local_path
    else:
        renamed = zarr_dir / hf_name
        if not renamed.exists():
            local_path.rename(renamed)
        src = renamed
    src.rename(tmp_zarr)

    token = os.environ.get("HF_TOKEN") or os.getenv("HF_TOKEN")

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo_id=HF_REPO, repo_type=HF_REPO_TYPE, exist_ok=True)

        print(f"Uploading {tmp_zarr} → {HF_REPO}/{hf_name} ...")
        api.upload_large_folder(
            folder_path=str(tmp_parent),
            repo_id=HF_REPO,
            repo_type=HF_REPO_TYPE,
            num_workers=NUM_WORKERS,
        )
        print(f"Done: https://huggingface.co/datasets/{HF_REPO}/tree/main/{hf_name}")
    finally:
        # On success or failure, restore zarr to its original location
        # (run_pipeline.sh will delete it after this script exits successfully)
        if tmp_zarr.exists():
            tmp_zarr.rename(zarr_dir / hf_name)
        if tmp_parent.exists():
            shutil.rmtree(tmp_parent, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/eumetsearch/utils/upload_to_hf.py <local_zarr_path> <hf_folder_name>")
        sys.exit(1)
    upload(sys.argv[1], sys.argv[2])
