import numpy as np
from pathlib import Path
from PIL import Image
import torch
from unet import UNet 

def load_image(image_path):
    """Load and preprocess an image."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if image has alpha channel
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_arr = np.array(img).astype(np.uint8)
            # Normalize to [0, 1] and add batch dimension
            img_tensor = torch.from_numpy(img_arr).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            return img_tensor
    except FileNotFoundError:
        print(f"Error: File not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def run_inference(model, image_tensor, device='cpu'):
    """Run inference on a single image."""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        # Apply sigmoid if your model doesn't include it in forward pass
        output = torch.sigmoid(output)
        return output.squeeze().cpu().numpy()

def save_output(output, output_path):
    """Save the output mask."""
    # Convert to 8-bit image (0-255)
    output_img = (output * 255).astype(np.uint8)
    Image.fromarray(output_img).save(output_path)

if __name__ == "__main__":
    # Configuration
    input_dir = 'PATH_TO_IMAGE'  # Can be directory or single image
    output_dir = 'PATH_TO_SAVE_OUTPUT'
    model_weights = 'PATH_TO_MODEL_WEIGHTS.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    unet = UNet(in_channels=3, out_channels=1).to(device)
    
    # Load trained weights
    try:
        unet.load_state_dict(torch.load(model_weights)['model_state_dict'])
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
    
    for file in files:
        print(f"Processing {file.name}...")
        # Load image
        image_tensor = load_image(file)
        if image_tensor is None:
            continue
            
        # Run inference
        output = run_inference(unet, image_tensor, device)
        
        # Save output
        output_path = Path(output_dir) / f"{file.stem}_mask{file.suffix}"
        save_output(output, output_path)
        print(f"Saved output to {output_path}")