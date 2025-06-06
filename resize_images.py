import os
from PIL import Image
import numpy as np
from tqdm import tqdm

def resize_with_padding(image_path, target_size=(448, 448), padding_color=(255, 255, 255), maintain_aspect=True):
    """
    Resize image while maintaining aspect ratio and adding padding
    
    Args:
        image_path: Path to the image
        target_size: Target size (width, height)
        padding_color: RGB color for padding (default: white)
        maintain_aspect: If True, maintain aspect ratio with padding, if False, stretch to fit
    """
    img = Image.open(image_path)
    
    if maintain_aspect:
        # Calculate aspect ratio
        aspect_ratio = img.width / img.height
        
        if aspect_ratio > 1:
            # Image is wider than tall
            new_width = target_size[0]
            new_height = int(new_width / aspect_ratio)
        else:
            # Image is taller than wide
            new_height = target_size[1]
            new_width = int(new_height * aspect_ratio)
        
        # Resize image
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create new image with padding
        new_img = Image.new("RGB", target_size, padding_color)
        
        # Calculate position to paste resized image
        paste_x = (target_size[0] - new_width) // 2
        paste_y = (target_size[1] - new_height) // 2
        
        # Paste resized image
        new_img.paste(img, (paste_x, paste_y))
    else:
        # Simply resize to target size without maintaining aspect ratio
        new_img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    return new_img

def process_directory(input_dir, output_dir, target_size=(448, 448), padding_color=(255, 255, 255), maintain_aspect=True):
    """
    Process all images in directory and its subdirectories
    
    Args:
        input_dir: Input directory with class subdirectories
        output_dir: Output directory for resized images
        target_size: Target size for images
        padding_color: RGB color for padding
        maintain_aspect: Whether to maintain aspect ratio
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    for class_dir in tqdm(class_dirs, desc="Processing classes"):
        # Create class directory in output
        class_output_dir = os.path.join(output_dir, class_dir)
        os.makedirs(class_output_dir, exist_ok=True)
        
        # Get all images in class directory
        class_input_dir = os.path.join(input_dir, class_dir)
        image_files = [f for f in os.listdir(class_input_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
        
        for img_file in tqdm(image_files, desc=f"Processing {class_dir}", leave=False):
            input_path = os.path.join(class_input_dir, img_file)
            output_path = os.path.join(class_output_dir, img_file)
            
            try:
                # Resize image
                resized_img = resize_with_padding(
                    input_path, 
                    target_size=target_size,
                    padding_color=padding_color,
                    maintain_aspect=maintain_aspect
                )
                
                # Save resized image
                resized_img.save(output_path, quality=95)
            except Exception as e:
                print(f"Error processing {input_path}: {str(e)}")

if __name__ == "__main__":
    input_directory = "/home/little-garden/CodeProjects/InteriorClass/data/reference_images"
    output_directory = "/home/little-garden/CodeProjects/InteriorClass/data/reference_images_resized"
    
    # You can choose between these options:
    # 1. White padding (default)
    process_directory(input_directory, output_directory)
    
    # 2. Black padding
    # process_directory(input_directory, output_directory, padding_color=(0, 0, 0))
    
    # 3. No padding (stretch to fit)
    # process_directory(input_directory, output_directory, maintain_aspect=False)
    
    print("Image resizing completed!") 