import os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm


def resize_with_padding(
        image: Image.Image,
        target_size: tuple = (448, 448),
        padding_color: tuple = (255, 255, 255),
        maintain_aspect: bool = True,
        resample_method: Image.Resampling = Image.Resampling.LANCZOS
    ) -> Image.Image:
    """
    Resizes image with optional padding while maintaining aspect ratio
    
    Args:
        image: PIL Image object (already loaded)
        target_size: Desired (width, height) tuple
        padding_color: RGB tuple for padding (default white)
        maintain_aspect: Whether to preserve original aspect ratio
        resample_method: Resampling filter (default LANCZOS for high quality)
    
    Returns:
        Processed PIL Image
    """
    if not maintain_aspect:
        return image.resize(target_size, resample_method)
    
    # Calculate optimal dimensions
    width, height = image.size
    target_width, target_height = target_size
    
    # Determine scaling ratio (use min to fit within target)
    ratio = min(target_width / width, target_height / height)
    new_size = (int(width * ratio), int(height * ratio))
    
    # Resize with high-quality filter
    resized = image.resize(new_size, resample_method)
    
    # Create new image with padding if needed
    if new_size != target_size:
        padded = Image.new("RGB", target_size, padding_color)
        paste_pos = (
            (target_width - new_size[0]) // 2,
            (target_height - new_size[1]) // 2
        )
        padded.paste(resized, paste_pos)
        return padded
    
    return resized


def process_directory(
        input_dir: Path,
        output_dir: Path,
        target_size: tuple = (448, 448),
        padding_color: tuple = (255, 255, 255),
        maintain_aspect: bool = True
    ) -> None:
    """
    Process all images in directory and its subdirectories with Path objects
    
    Args:
        input_dir: Input directory with class subdirectories (Path)
        output_dir: Output directory for resized images (Path)
        target_size: Target size for images (default: (448, 448))
        padding_color: RGB color for padding (default: white)
        maintain_aspect: Whether to maintain aspect ratio (default: True)
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    for class_dir in tqdm(class_dirs, desc="Processing classes"):
        # Create class directory in output
        class_output_dir = output_dir / class_dir.name
        class_output_dir.mkdir(exist_ok=True)
        
        # Get all images in class directory
        image_files = [f for f in class_dir.iterdir() 
                      if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.webp')]
        
        for img_file in tqdm(image_files, desc=f"Processing {class_dir.name}", leave=False):
            output_path = class_output_dir / img_file.name
            
            try:
                # Load and process image
                with Image.open(img_file) as img:
                    resized_img = resize_with_padding(
                        image=img,
                        target_size=target_size,
                        padding_color=padding_color,
                        maintain_aspect=maintain_aspect
                    )
                    
                    # Save resized image (preserve original format)
                    resized_img.save(output_path, quality=95)
                    
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")

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