"""
Download the 14:00 UTC observation for June 1–4 2025 and append to the
existing mtg_june_2025_majortom.zarr (which already has 09:00–13:00).
"""
import os, shutil, argparse
from dotenv import load_dotenv
from datetime import datetime, timedelta
from eumetsearch import products_list, EUMDownloader, init_logging, get_bbox, BBOX_REGISTRY
from eumetsearch.data_collection.majortom_export import MajorTomZarrExport

load_dotenv()
logger = init_logging("./logger_mtg_fci.log", verbose=False)

# Minimal args namespace expected by MajorTomZarrExport
args = argparse.Namespace(
    resampler="nearest",
    threading=True,
    yes=False,
    remove=True,
    region="mtg",
    dask_threads=0,
)

product_id = products_list["MTG-FCI-L1C-FDHSI"]["product_id"]
W, S, E, N = get_bbox(args.region)
NSWE = [N, S, W, E]

downloader = EUMDownloader(
    product_id=product_id,
    output_dir="./data/datastore_data",
    max_parallel_conns=10,
)

def remove_folder(folder):
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

# Download one day at a time, 1 observation at 14:00
for day in range(1, 5):
    start = datetime(2025, 6, day, 14, 0, 0)
    end   = datetime(2025, 6, day, 14, 10, 0)
    logger.info(f"Downloading 14:00 for 2025-06-{day:02d}")

    downloader.download_interval(
        start_time=start.isoformat(),
        end_time=end.isoformat(),
        bounding_box=NSWE,
        observations_per_day=1,
        jump_minutes=60,
        start_hour=14,
    )

    MajorTomZarrExport(
        args,
        downloader,
        label="mtg_june_2025",
        channels=["vis_06", "vis_08"],
        dtype="float16",
    )

    try:
        remove_folder(str(downloader.output_dir / "disk" / "Data" / "zipfolder"))
        remove_folder(str(downloader.output_dir / "disk" / "Data" / "natfolder"))
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

logger.info("Done — 14:00 timestamps for June 1–4 appended.")
