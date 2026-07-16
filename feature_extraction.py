import os
import cv2
import numpy as np
import pandas as pd
from scipy.stats import entropy

# =====================================================
# Folder Paths
# =====================================================

COVER_FOLDER = "dataset/cover_png"
STEGO_FOLDER = "dataset/stego"

# =====================================================
# Mean
# =====================================================

def get_mean(image):
    return float(np.mean(image))

# =====================================================
# Variance
# =====================================================

def get_variance(image):
    return float(np.var(image))

# =====================================================
# Entropy
# =====================================================

def get_entropy(image):

    histogram, _ = np.histogram(image, bins=256, range=(0, 256))

    histogram = histogram / histogram.sum()

    histogram = histogram[histogram > 0]

    return float(entropy(histogram, base=2))

# =====================================================
# LSB Ratio
# =====================================================

def get_lsb_ratio(image):

    lsb = image & 1

    ones = np.sum(lsb)

    total = lsb.size

    return float(ones / total)

# =====================================================
# Histogram Features
# (Kept for compatibility with Member 2)
# =====================================================

def get_histogram_features(image):

    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

    histogram = histogram.flatten()

    histogram = histogram / np.sum(histogram)

    return histogram

# =====================================================
# Pixel Difference
# =====================================================

def get_pixel_difference(image):

    horizontal = np.abs(np.diff(image, axis=1))

    vertical = np.abs(np.diff(image, axis=0))

    return float(np.mean(horizontal)), float(np.mean(vertical))

# =====================================================
# Extract Features
# =====================================================

def extract_features(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise Exception(f"Cannot read image: {image_path}")

    features = {}

    features["Mean"] = get_mean(image)
    features["Variance"] = get_variance(image)
    features["Entropy"] = get_entropy(image)
    features["LSB Ratio"] = get_lsb_ratio(image)

    hdiff, vdiff = get_pixel_difference(image)

    features["Horizontal Difference"] = hdiff
    features["Vertical Difference"] = vdiff

    histogram = get_histogram_features(image)

    return features, histogram

# =====================================================
# Dataset Creation
# =====================================================

rows = []
original_rows = []
stego_rows = []

print("\nProcessing Cover Images...\n")

for filename in sorted(os.listdir(COVER_FOLDER)):

    if not filename.lower().endswith(".png"):
        continue

    image_path = os.path.join(COVER_FOLDER, filename)

    features, histogram = extract_features(image_path)

    row = {
        "Image": filename,
        "Mean": features["Mean"],
        "Variance": features["Variance"],
        "Entropy": features["Entropy"],
        "LSB Ratio": features["LSB Ratio"],
        "Horizontal Difference": features["Horizontal Difference"],
        "Vertical Difference": features["Vertical Difference"],
        "Label": 0
    }

    rows.append(row)
    original_rows.append(row)

    print(f"✔ Cover : {filename}")

print("\nProcessing Stego Images...\n")

for filename in sorted(os.listdir(STEGO_FOLDER)):

    if not filename.lower().endswith(".png"):
        continue

    image_path = os.path.join(STEGO_FOLDER, filename)

    features, histogram = extract_features(image_path)

    row = {
        "Image": filename,
        "Mean": features["Mean"],
        "Variance": features["Variance"],
        "Entropy": features["Entropy"],
        "LSB Ratio": features["LSB Ratio"],
        "Horizontal Difference": features["Horizontal Difference"],
        "Vertical Difference": features["Vertical Difference"],
        "Label": 1
    }

    rows.append(row)
    stego_rows.append(row)

    print(f"✔ Stego : {filename}")

# =====================================================
# Save CSV Files
# =====================================================

columns = [
    "Image",
    "Mean",
    "Variance",
    "Entropy",
    "LSB Ratio",
    "Horizontal Difference",
    "Vertical Difference",
    "Label"
]

combined_df = pd.DataFrame(rows)[columns]
original_df = pd.DataFrame(original_rows)[columns]
stego_df = pd.DataFrame(stego_rows)[columns]

combined_df.to_csv("features.csv", index=False)
original_df.to_csv("original_features.csv", index=False)
stego_df.to_csv("stego_features.csv", index=False)

print("\n======================================")
print("Datasets Generated Successfully!")
print("======================================")
print(f"Combined Dataset : features.csv ({len(combined_df)} images)")
print(f"Original Dataset : original_features.csv ({len(original_df)} images)")
print(f"Stego Dataset    : stego_features.csv ({len(stego_df)} images)")
print("======================================")