from dataset import CloudBalancedDatasetWithSynthesis
from unet import UNet

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from transformers import SegformerForSemanticSegmentation, AdamW

import wandb
from tqdm import tqdm
from collections import defaultdict

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def compute_iou(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum(dim=(1,2,3))
    union = ((pred_bin + target) > 0).float().sum(dim=(1,2,3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

# --- Main Training ---
def train_model(model, train_loader, val_loader, early_stopper, model_save_path, device, samples, batch_size, epochs=10, lr=1e-4):
    wandb.init(project="cloud-mask-segformer", config={"epochs": epochs, "lr": lr}, name=f"lr={lr}-b={batch_size}-samples={samples}")
    
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)
    scaler = GradScaler()
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            images = batch["pixel_values"].to(device)
            masks = batch["mask"].to(device)  # shape: (B, H, W)
            
            # Resize images and masks to 512x512
            images = F.interpolate(images, size=(512, 512), mode='bilinear', align_corners=False)
            masks = masks.unsqueeze(1)  # shape: (B, 1, H, W) for interpolation
            masks = F.interpolate(masks.float(), size=(512, 512), mode='nearest')
            masks = masks.squeeze(1).long()

            optimizer.zero_grad()
            with autocast():
                outputs = model(pixel_values=images, labels=masks)
                logits = outputs.logits
                logits_resized = F.interpolate(logits, size=(512, 512), mode="bilinear", align_corners=False)

                loss = F.cross_entropy(logits_resized, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            preds = logits_resized.argmax(dim=1)
            train_correct += (preds == masks).sum().item()
            train_total += masks.numel()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        model.eval()
        val_loss, val_iou, val_correct, val_total = 0, 0, 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                images = batch["pixel_values"].to(device)
                masks = batch["mask"].to(device)  # shape: (B, H, W)

                # Resize images and masks to 512x512
                images = F.interpolate(images, size=(512, 512), mode='bilinear', align_corners=False)
                masks = masks.unsqueeze(1)  # shape: (B, 1, H, W) for interpolation
                masks = F.interpolate(masks.float(), size=(512, 512), mode='nearest')
                masks = masks.squeeze(1).long()
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(pixel_values=images)
                logits = outputs.logits  # shape: (B, num_classes, H_out, W_out)
                
                # Resize logits to match mask size
                logits_resized = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                
                # Compute loss manually
                loss = F.cross_entropy(logits_resized, masks)
                val_loss += loss.item()
        
                preds = logits_resized.argmax(dim=1)  # shape: (B, H, W)
                val_correct += (preds == masks).sum().item()
                val_total += masks.numel()
                val_iou += compute_iou(preds, masks)

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_acc = val_correct / val_total

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, model_save_path)
            print(f"✅ Saved best checkpoint (val_loss={val_loss:.4f}) to {model_save_path}")

        scheduler.step(val_loss)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_iou": val_iou,
            "val_accuracy": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })

        print(f"📉 Epoch {epoch+1}: "
              f"Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, "
              f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}, Val IoU = {val_iou:.4f}")

        if early_stopper(val_loss):
            print("⏹️ Early stopping triggered.")
            break

    wandb.finish()
    
def evaluate_model(checkpoint_path, test_loader, device):
    print(f"🔍 Evaluating {checkpoint_path}")
    
    # Load model
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    total_correct = 0
    total_pixels = 0
    total_iou = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch["pixel_values"].to(device)
            masks = batch["mask"].to(device)  # shape: (B, H, W)
            
            images = F.interpolate(images, size=(512, 512), mode='bilinear', align_corners=False)
            masks = masks.unsqueeze(1)  # shape: (B, 1, H, W) for interpolation
            masks = F.interpolate(masks.float(), size=(512, 512), mode='nearest')
            masks = masks.squeeze(1).long()

            outputs = model(pixel_values=images)
            logits = outputs.logits  # shape: (B, 2, H_out, W_out)

            # Resize to match ground truth mask resolution
            logits_resized = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            preds = logits_resized.argmax(dim=1)  # shape: (B, H, W)

            total_correct += (preds == masks).sum().item()
            total_pixels += masks.numel()
            total_iou += compute_iou(preds, masks)

    acc = total_correct / total_pixels
    avg_iou = total_iou / len(test_loader)
    print(f"📊 Test Accuracy: {acc:.4f}, Test IoU: {avg_iou:.4f}")

if __name__ == "__main__":
    
    dataset = torch.load('dataset.pt')

    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    batch_size = 16
    samples = 500
    lr = 1e-4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Step 1: Organize indices by (group_key, bin_key)
    bins = defaultdict(list)
    for idx, (_, _, group_key, bin_key) in enumerate(dataset.sampled_data):
        bins[(group_key, bin_key)].append(idx)

    # Step 2: Split each bin separately
    train_idx, val_idx, test_idx = [], [], []

    for bin_key, indices in bins.items():
        train_i, rest = train_test_split(indices, train_size=train_ratio, random_state=42)
        val_i, test_i = train_test_split(rest, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)
        
        train_idx.extend(train_i)
        val_idx.extend(val_i)
        test_idx.extend(test_i)

    # Step 3: Use Subset instead of random_split
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = UNet(in_channels=3, out_channels=1).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    early_stopper = EarlyStopping(patience=6, min_delta=1e-4)
        
    train_model(model,train_loader,val_loader,early_stopper,f'unet_{samples}_{lr}_{batch_size}',device,samples,batch_size,20,lr)
    evaluate_model(f'unet_{samples}_{lr}_{batch_size}',test_loader,device)