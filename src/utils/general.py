import logging
import pandas as pd
from datetime import datetime
import xarray as xr


def bbox_mtg():
    return [-18.105469,-37.857507,60.820313,71.413177]

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


def add_time(dataset:xr.Dataset)-> xr.Dataset:
    if 'EPCT_start_sensing_time' in dataset.attrs:
        my_date_string = dataset.attrs['EPCT_start_sensing_time']#dataset.attrs['date_time']
        date_xr = datetime.strptime(my_date_string,'%Y%m%dT%H%M%SZ') #datetime.strptime(my_date_string, '%Y%m%d/%H:%M')
        date_xr = pd.to_datetime(date_xr)
    else:
        import re
        string_time = dataset.encoding["source"].split("/")[-1]
        pattern = r'(\d{4}\d{2}\d{2})(\d{2}\d{2}\d{2})'

        # Use re.search() to find the pattern in the string
        match = re.search(pattern, string_time)
        if match:
            start_date = match.group(1)
            start_time = match.group(2)
            date_xr = pd.to_datetime(start_date + start_time, format='%Y%m%d%H%M%S')
        else:
            print("Could not parse time in xarray dataset {}".format(string_time))
            return dataset
    
    dataset = dataset.assign_coords(time=date_xr)
    dataset = dataset.expand_dims(dim="time")
    return dataset

def compute_ndvi(dataset:xr.Dataset, channel_1:str, channel_2:str) -> xr.Dataset:
    return dataset.assign(ndvi=(
        dataset[channel_2] - dataset[channel_1]) / (dataset[channel_2] + dataset[channel_1]))
