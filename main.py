import os 
from dotenv import load_dotenv
import eumdac
from eumdac import DataStore
from utils import products_list

from utils import EUMDownloader, bbox_mtg, init_logging

logger = init_logging()

# Load environment variables from .env file
load_dotenv()

product_id = products_list["MTG-1"]["product_id"]


bbox = bbox_mtg()
W,S,E,N = bbox[0], bbox[1], bbox[2], bbox[3]
NSWE = [N, S, W, E]

downloader = EUMDownloader(product_id=product_id, 
    output_dir="./data/datatailor_data", 
    format="netcdf4",
)

downloader.download_interval(
    start_time="2025-05-02T00:00:00", 
    end_time="2025-05-30T00:00:00",
    bounding_box=NSWE, 
    observations_per_day=3,
    method="datatailor",
    jump_minutes=60
)


# import xarray as xr
# from utils import load_clean_xarray_dataset, convert_mtg_fci_with_offsets
# import os 
# output_dir="./data"

# filenames = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".nc")]

# ds = load_clean_xarray_dataset(os.path.join(output_dir, "*.nc"))
# ds_clean = convert_mtg_fci_with_offsets(ds, selected_channels=["vis_06", "vis_08"])
