import os

from datasets import load_dataset
from PIL import Image  # Import Image from Pillow

# --- Configuration ---
dataset_name = "benjamin-paine/imagenet-1k-256x256"
output_folder = "data/imagenet_256_images_validation"  # Choose your desired output folder name
split_to_download = "validation"  # Standard for FID-50k uses the validation set
# --- End Configuration ---

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)
print(f"Created output folder: {output_folder}")

# Load the dataset (consider using streaming=True for very large datasets if memory is an issue,
# but saving requires iterating through anyway)
# You might need to authenticate if the dataset requires it, follow library instructions if prompted.
print(f"Loading dataset '{dataset_name}', split '{split_to_download}'...")
try:
    # This dataset is stored in Parquet format, so you might need pyarrow
    # pip install pyarrow
    dataset = load_dataset(dataset_name, split=split_to_download, cache_dir="./data")
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Make sure you have 'pyarrow' installed (`pip install pyarrow`)")
    exit()

# Check if the dataset has the expected 'image' column
if "image" not in dataset.column_names:
    print(f"Error: Dataset does not contain an 'image' column. Available columns: {dataset.column_names}")
    exit()

# Iterate through the dataset and save images
print(f"Starting image download to '{output_folder}'...")
count = 0
num_digits = len(str(len(dataset) - 1))  # For zero-padding filenames

for i, example in enumerate(dataset):
    image = example["image"]  # Access the image data (likely a PIL Image object)

    # Ensure it's a PIL Image object before saving
    if not isinstance(image, Image.Image):
        print(f"Warning: Item {i} is not a PIL Image object (Type: {type(image)}). Skipping.")
        continue

    # Construct filename (e.g., image_00000.png, image_00001.png, ...)
    # Using PNG is often preferred for evaluations like FID to avoid compression artifacts.
    filename = f"image_{str(i).zfill(num_digits)}.png"
    filepath = os.path.join(output_folder, filename)

    try:
        # Save the image
        image.save(filepath)
        count += 1
    except Exception as e:
        print(f"Error saving image {i} to {filepath}: {e}")

    # Optional: Print progress
    if (i + 1) % 1000 == 0:
        print(f"Saved {i + 1}/{len(dataset)} images...")

print("\nFinished downloading.")
print(f"Successfully saved {count} images to '{output_folder}'.")
