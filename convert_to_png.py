from PIL import Image
import os

# -----------------------------
# Folder Paths
# -----------------------------
input_folder = "dataset/cover"
output_folder = "dataset/cover_png"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Supported image extensions
valid_extensions = (".jpg", ".jpeg", ".png")

# Get all image files
image_files = [
    file for file in os.listdir(input_folder)
    if file.lower().endswith(valid_extensions)
]

# Sort files for consistent numbering
image_files.sort()

print(f"Found {len(image_files)} images.\n")

# Convert and rename
for index, filename in enumerate(image_files, start=1):

    input_path = os.path.join(input_folder, filename)

    # New filename
    new_filename = f"img{index:03d}.png"

    output_path = os.path.join(output_folder, new_filename)

    # Open image
    image = Image.open(input_path)

    # Convert to RGB (important for JPG and some PNGs)
    image = image.convert("RGB")

    # Save as PNG
    image.save(output_path, "PNG")

    print(f"{filename}  --->  {new_filename}")

print("\n===================================")
print("Conversion Completed Successfully!")
print(f"Total Images Converted : {len(image_files)}")
print("Saved inside : dataset/cover_png")
print("===================================")