import os
import json
import s3fs
from dotenv import load_dotenv

import torch
from torch.utils.data import Dataset

import xarray as xr
import numpy as np
import random

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
bucket_name = os.getenv('MINIO_BUCKET_NAME', 'fusion-lake')

# Load the JSON data
with open('cloud_masks_data.json', 'r') as f:
    data = json.load(f)

# Set the device for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Extract patches from the dataset
def extract_patch(ds, patch_row, patch_col, patch_size=384):
    y_start = patch_row * patch_size
    x_start = patch_col * patch_size

    # Handle edge cases so we don't go out of bounds
    y_end = min(y_start + patch_size, ds.dims['y'])
    x_end = min(x_start + patch_size, ds.dims['x'])

    patch = ds.isel(
        y=slice(y_start, y_end),
        x=slice(x_start, x_end)
    )
    return patch

# Define the dataset class
class CloudBalancedDatasetWithSynthesis(Dataset):
    def __init__(self, data_dict, s3_options, bucket_name, samples_per_bin=200, patch_size=384):
        """
        data_dict: dict of the form {'8': high_res_dict, '10': high_res_dict, ...}
        where each high_res_dict is a dict with keys as cloud bins like '0', '1', etc.
        """
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())  # Outer keys: '8', '10', '30', etc.
        self.samples_per_bin = samples_per_bin
        self.s3_options = s3_options
        self.bucket_name = bucket_name
        self.patch_size = patch_size

        self.cloud_bins = self._infer_common_bins()
        self.total_samples = len(self.keys) * len(self.cloud_bins) * self.samples_per_bin
        
        self.prepare_epoch()
        
    def _infer_common_bins(self):
        """Assume all high_res dicts share the same cloud bins."""
        any_key = next(iter(self.data_dict))
        return list(self.data_dict[any_key].keys())

    def prepare_epoch(self):
        self.sampled_data = []

        for group_key in self.keys:
            high_res = self.data_dict[group_key]
            clear_pool = [x for x in high_res['0'] if self.is_valid(x)]

            if not clear_pool:
                raise ValueError(f"No valid clear images in group {group_key}")

            for bin_key in self.cloud_bins:
                bin_images = [x for x in high_res[bin_key] if self.is_valid(x)]

                if len(bin_images) >= self.samples_per_bin:
                    selected = random.sample(bin_images, self.samples_per_bin)
                    self.sampled_data.extend([('real', x, group_key, bin_key) for x in selected])
                else:
                    self.sampled_data.extend([('real', x, group_key, bin_key) for x in bin_images])
                    deficit = self.samples_per_bin - len(bin_images)

                    for _ in range(deficit):
                        cloudy_image = random.choice(bin_images) if bin_images else random.choice(clear_pool)
                        clear_image = random.choice(clear_pool)
                        self.sampled_data.append(('synthetic', (clear_image, cloudy_image), group_key, bin_key))

        if not self.sampled_data:
            raise ValueError("No usable samples could be prepared.")

        random.shuffle(self.sampled_data)
        print(f"✅ Prepared {len(self.sampled_data)} samples from {len(self.keys)} groups × {len(self.cloud_bins)} bins × {self.samples_per_bin}.")

    def __len__(self):
        return len(self.sampled_data)

    def __getitem__(self, idx):
        tag, data, group_key, bin_key = self.sampled_data[idx]
    
        def to_tensor_and_normalize(patch):
            patch = patch.astype(np.float32)
            patch = np.nan_to_num(patch)
            if patch.max() > patch.min():
                patch = (patch - patch.min()) / (patch.max() - patch.min())
        
            # If more than 3 bands, truncate to first 3
            if patch.shape[-1] > 3:
                patch = patch[..., :3]
        
            # Channel-first format for torch: (C, H, W)
            return torch.from_numpy(patch).permute(2, 0, 1)
    
        if tag == 'real':
            row, col, zarr_path, cloud_pct = data
    
            # Paths
            image_path = f"s3://{self.bucket_name}/{zarr_path}"
            mask_path = f"s3://{self.bucket_name}/{zarr_path.replace('raw', 'mask')}"
    
            # Load Zarr datasets
            image_ds = xr.open_zarr(image_path, storage_options=self.s3_options, consolidated=True, chunks={})
            mask_ds = xr.open_zarr(mask_path, storage_options=self.s3_options, consolidated=True, chunks={})
    
            # Extract patches
            image_patch = extract_patch(image_ds, row, col, self.patch_size)
            mask_patch = extract_patch(mask_ds, row, col, self.patch_size)
    
            # Extract arrays
            band = list(image_patch.keys())[0]
            image = image_patch[band].load().values
            mask = mask_patch[list(mask_patch.keys())[0]].load().values
    
            # Postprocess
            image = to_tensor_and_normalize(image)
            mask = torch.from_numpy((np.nan_to_num(mask) > 0).astype(np.float32)).unsqueeze(0)  # (1, H, W)
    
        else:  # synthetic
            clear_data, cloud_data = data
            synthetic_image, mask_np = self.overlay_and_extract(clear_data, cloud_data)

            image = to_tensor_and_normalize(synthetic_image)
            mask = torch.from_numpy((mask_np > 0).astype(np.float32)).unsqueeze(0)
    
        # Swap BGR → RGB if group_key == '30'
        if str(group_key) == '30':
            image = image[[2, 1, 0], :, :]  # Swap channels

        return {
            "pixel_values": image,                   # For SegFormer input
            "mask": mask.long().squeeze(0)           # For training target (CrossEntropyLoss)
        }


    def overlay_and_extract(self, clear_image_data, cloud_image_data):
        clear_row, clear_col, clear_zarr_link, _ = clear_image_data
        cloud_row, cloud_col, cloud_zarr_link, _ = cloud_image_data

        clear_path = f"s3://{self.bucket_name}/{clear_zarr_link}"
        cloud_path = f"s3://{self.bucket_name}/{cloud_zarr_link}"
        mask_path = f"s3://{self.bucket_name}/{cloud_zarr_link.replace('raw', 'mask')}"

        clear_ds = xr.open_zarr(clear_path, storage_options=self.s3_options, consolidated=True, chunks={})
        cloud_ds = xr.open_zarr(cloud_path, storage_options=self.s3_options, consolidated=True, chunks={})
        mask_ds = xr.open_zarr(mask_path, storage_options=self.s3_options, consolidated=True, chunks={})

        clear_patch = extract_patch(clear_ds, clear_row, clear_col, self.patch_size)
        cloud_patch = extract_patch(cloud_ds, cloud_row, cloud_col, self.patch_size)
        mask_patch = extract_patch(mask_ds, cloud_row, cloud_col, self.patch_size)

        clear = clear_patch[list(clear_patch.keys())[0]].load().values.astype(np.float32)
        cloud = cloud_patch[list(cloud_patch.keys())[0]].load().values.astype(np.float32)
        mask = mask_patch[list(mask_patch.keys())[0]].load().values

        clear = np.nan_to_num(clear)
        cloud = np.nan_to_num(cloud)
        mask = (np.nan_to_num(mask) > 0).astype(bool)

        if clear.shape != cloud.shape:
            raise ValueError(f"Shape mismatch: {clear.shape} vs {cloud.shape}")

        # Normalize
        for img in [clear, cloud]:
            if img.max() > img.min():
                img -= img.min()
                img /= (img.max() + 1e-8)

        # Overlay cloud onto clear
        mask_3d = np.stack([mask] * clear.shape[-1], axis=-1)
        synthetic = np.where(mask_3d, cloud, clear).astype(np.float32)

        # Cloud cover %
        # cloud_pct = round(100.0 * np.sum(mask) / mask.size, 2)

        return synthetic,  mask
    
    def is_valid(self, sample):
        row, col, zarr_path, _ = sample
        ds = xr.open_zarr(
            store=f"s3://{self.bucket_name}/{zarr_path}",
            storage_options=self.s3_options,
            consolidated=True,
            chunks={}
        )
        patch = extract_patch(ds, row, col, patch_size=self.patch_size)
        band = list(patch.keys())[0]
        data = patch[band]
        return data.shape[0] == self.patch_size and data.shape[1] == self.patch_size


# Create the dataset instance
dataset = CloudBalancedDatasetWithSynthesis(
    data_dict=data,
    s3_options=s3_options,
    bucket_name=bucket_name,
    samples_per_bin=500,
    patch_size=384
)

# Save the dataset to a file
torch.save(dataset,'dataset.pt')