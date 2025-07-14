import os
import json
from dotenv import load_dotenv
import s3fs

from tqdm import tqdm
import xarray as xr
import math

# Load environment variables
load_dotenv()

# Load S3 configuration from environment variables
s3_options = {
    "key": os.getenv('MINIO_ACCESS_KEY'),
    "secret": os.getenv('MINIO_SECRET_KEY'),
    "client_kwargs": {'endpoint_url': os.getenv('MINIO_ENDPOINT_URL', 'http://localhost:9000')},
    "config_kwargs": {'s3': {'addressing_style': 'path'}} # Important for MinIO
}
s3 = s3fs.S3FileSystem(**s3_options)

# Load bucket name from environment variables 
bucket_name = os.getenv('MINIO_BUCKET_NAME', 'fusion-lake')

# List all STAC files in the bucket
prefix = 'stac/' 
stac_files = s3.ls(f's3://{bucket_name}/{prefix}')
stac_files = [f for f in stac_files]

cloud_masks_data = {}
bin_width = 5

# Function to convert tile index to row and column
def tile_index_to_row_col(index, n_cols):
    row = index // n_cols
    col = index % n_cols
    return row, col


# Iterate through each STAC file and extract cloud cover data
for filename in tqdm(stac_files):
    with s3.open(filename, 'r') as f:
        stac = json.load(f)
        
        cloud_arr = stac['properties']['tilewise:cloud_cover']
        
        zarr_link = stac['assets']['data_zarr']['href']
        
        gsd = stac.get('properties', {}).get('gsd', 8)
        zarr_catalog_url = f's3://{bucket_name}/{zarr_link}'
        
        ds_lazy = xr.open_zarr(
            store=zarr_catalog_url,
            storage_options=s3_options,
            consolidated=True,
            chunks={}
        )
        
        n_rows = math.ceil(ds_lazy.dims['y'] / 384)
        n_cols = math.ceil(ds_lazy.dims['x'] / 384)
        
        if gsd not in cloud_masks_data:
            cloud_masks_data[gsd] = {}
        
        for i, pct in enumerate(cloud_arr):
            row, col = tile_index_to_row_col(i, n_cols)
            
            bin = int(pct // bin_width)
            
            if bin not in cloud_masks_data[gsd]:
                cloud_masks_data[gsd][bin] = []
                
            cloud_masks_data[gsd][bin].append((row, col, zarr_link, pct))

# Save the cloud masks data to a JSON file
json.dump(cloud_masks_data, open('cloud_masks_data.json', 'w'), indent=4)