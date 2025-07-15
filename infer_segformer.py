import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

def load_image(image_path, target_size=(512, 512)):
    """Load and preprocess an image for SegFormer."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to tensor (C, H, W) and normalize [0, 1]
            img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            
            # Resize using F.interpolate (BILINEAR for images)
            img_tensor = F.interpolate(
                img_tensor,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
            
            return img_tensor
            
    except FileNotFoundError:
        print(f"Error: File not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def run_inference(model, image_tensor, original_size=None, device='cpu'):
    """Run inference on a single image with SegFormer."""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(pixel_values=image_tensor)
        logits = outputs.logits
        
        # Resize logits to match original image size (if provided)
        if original_size is not None:
            logits = F.interpolate(
                logits,
                size=original_size,
                mode='bilinear',
                align_corners=False
            )
        
        # Get predicted class (assuming binary segmentation)
        preds = logits.argmax(dim=1).squeeze().cpu().numpy()
        return preds

def save_output(output, output_path):
    """Save the output mask."""
    # Convert to 8-bit image (0-255)
    output_img = (output * 255).astype(np.uint8)
    Image.fromarray(output_img).save(output_path)


if __name__ == "__main__":
    input_dir = 'PATH_TO_IMAGE'  # Can be directory or single image
    output_dir = 'PATH_TO_SAVE_OUTPUT'
    model_weights = 'PATH_TO_MODEL_WEIGHTS.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize SegFormer model
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=2,
        ignore_mismatched_sizes=True
    ).to(device)
    
    # Load trained weights
    try:
        checkpoint = torch.load(model_weights, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model weights loaded successfully")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        exit(1)
    
    # Process input (could be a single file or directory)
    input_path = Path(input_dir)
    if input_path.is_file():
        files = [input_path]
    else:
        files = list(input_path.glob('*'))  # Get all files in directory
    
    for file in tqdm(files, desc="Processing images"):
        print(f"\nProcessing {file.name}...")
        # Load image and remember original size
        with Image.open(file) as img:
            original_size = img.size[::-1]  # (H, W)
        
        image_tensor = load_image(file, target_size=(512, 512))
        if image_tensor is None:
            continue
            
        # Run inference (resize predictions back to original size)
        output = run_inference(
            model, 
            image_tensor, 
            original_size=original_size,
            device=device
        )
        
        # Save output
        output_path = Path(output_dir) / f"{file.stem}_mask{file.suffix}"
        save_output(output, output_path)
        print(f"Saved output to {output_path}")