import torch
import time
import os
import numpy as np
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))
        bn = self.bottleneck(self.pool4(d4))

        u4 = self.up4(bn)
        u4 = torch.cat([u4, d4], dim=1)
        u4 = self.conv4(u4)

        u3 = self.up3(u4)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.conv3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)

        return torch.sigmoid(self.out(u1))  # Output shape: (B, 1, H, W)
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size(model, path="temp_model.pth"):
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    os.remove(path)
    return size_mb

def measure_inference_time(model, input_tensor, resize=None, is_segformer=False, runs=30):
    model.eval()
    with torch.no_grad():
        for _ in range(5):  # warmup
            resized = F.interpolate(input_tensor, size=resize, mode='bilinear', align_corners=False) if resize else input_tensor
            model(pixel_values=resized) if is_segformer else model(resized)
        times = []
        for _ in range(runs):
            start = time.time()
            resized = F.interpolate(input_tensor, size=resize, mode='bilinear', align_corners=False) if resize else input_tensor
            model(pixel_values=resized) if is_segformer else model(resized)
            times.append((time.time() - start) * 1000)
    return np.mean(times)

batch_sizes = [1, 5, 10]
runs = 30

def benchmark_model(model, name, is_segformer=False, resize=None):
    records = []
    for bs in batch_sizes:
        input_tensor = torch.randn(bs, 3, 384, 384).to(device)
        if resize:
            resized_shape = resize
        else:
            resized_shape = (384, 384)

        time_ms = measure_inference_time(model, input_tensor, resize=resize, is_segformer=is_segformer, runs=runs)
        records.append({
            "Model": name,
            "Batch Size": bs,
            "Avg Inference Time (ms)": round(time_ms, 2),
            "Time per Image (ms)": round(time_ms / bs, 2)
        })
    return records

start = time.time()
unet = UNet(in_channels=3, out_channels=1).to(device)
unet.load_state_dict(torch.load("unet_500_0.0001_16_2", map_location=device)["model_state_dict"])
unet_time = time.time() - start
print(f"UNet loaded in {unet_time}s")

# Load SegFormer-B0
start = time.time()
segformer = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=2,
    ignore_mismatched_sizes=True
).to(device)
segformer.load_state_dict(torch.load("segformer_500_0.0001_16_1_b.pt", map_location=device)["model_state_dict"])
segformer_time = time.time() - start
print(f"Segformer loaded in {segformer_time}s")


# Run benchmark
unet_results = benchmark_model(unet, "U-Net", is_segformer=False)
segformer_results = benchmark_model(segformer, "SegFormer-B0", is_segformer=True, resize=(512, 512))

# Combine results
combined = unet_results + segformer_results

# Print as table
print(f"{'Model':<15} | {'Batch Size':<11} | {'Total Time (ms)':<18} | {'Time/Image (ms)':<17}")
print("-" * 70)
for entry in combined:
    print(f"{entry['Model']:<15} | {entry['Batch Size']:<11} | {entry['Avg Inference Time (ms)']:<18} | {entry['Time per Image (ms)']:<17}")
