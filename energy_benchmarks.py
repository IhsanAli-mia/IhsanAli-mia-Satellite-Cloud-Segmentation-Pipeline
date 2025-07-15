import time
import torch
import pynvml
import numpy as np
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F
import torch.nn as nn

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    

# NVIDIA Energy Monitor Setup
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def get_power_usage():
    """Returns power usage in Watts"""
    return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # milliwatts to watts

def energy_per_inference(model, is_segformer=False, resize_to=None, runs=100):
    model.to(device)
    model.eval()
    x = torch.randn(1, 3, 384, 384).to(device)

    energies = []
    with torch.no_grad():
        for _ in range(5):  # warm-up
            input_tensor = F.interpolate(x, size=resize_to, mode='bilinear', align_corners=False) if resize_to else x
            model(pixel_values=input_tensor) if is_segformer else model(input_tensor)

        for _ in range(runs):
            input_tensor = F.interpolate(x, size=resize_to, mode='bilinear', align_corners=False) if resize_to else x
            start_energy = get_power_usage()
            start_time = time.time()

            model(pixel_values=input_tensor) if is_segformer else model(input_tensor)

            end_time = time.time()
            end_energy = get_power_usage()
            avg_power = (start_energy + end_energy) / 2  # Watts
            duration = end_time - start_time  # seconds
            energy = avg_power * duration  # Joules = Watts × Seconds
            energies.append(energy)

    per_image = np.mean(energies)
    return round(per_image, 4), round(per_image * runs, 2)

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

# Benchmark energy
u_energy_per_inf, u_energy_100 = energy_per_inference(unet)
s_energy_per_inf, s_energy_100 = energy_per_inference(segformer, is_segformer=True, resize_to=(512, 512))

# Print LaTeX-style table
print("\\begin{table}[H]")
print("\\centering")
print("\\begin{tabular}{lcc}")
print("\\toprule")
print("\\textbf{Metric} & \\textbf{U-Net} & \\textbf{SegFormer-B0} \\\\")
print("\\midrule")
print(f"Energy per Inference (Joules) & {u_energy_per_inf} & {s_energy_per_inf} \\\\")
print(f"Energy for 100 Images (J)     & {u_energy_100} & {s_energy_100} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\caption{Measured energy consumption on edge device.}")
print("\\end{table}")