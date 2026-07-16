import cv2
import numpy as np
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Mean
# -----------------------------
def get_mean(image):
    return np.mean(image)

# -----------------------------
# Variance
# -----------------------------
def get_variance(image):
    return np.var(image)

# -----------------------------
# Entropy
# -----------------------------
def get_entropy(image):
    histogram, _ = np.histogram(image, bins=256, range=(0, 256))

    histogram = histogram / histogram.sum()

    histogram = histogram[histogram > 0]

    return entropy(histogram, base=2)

# -----------------------------
# LSB Ratio
# -----------------------------
def get_lsb_ratio(image):
    lsb = image & 1
    ones = np.sum(lsb)
    total = lsb.size
    return ones / total

# -----------------------------
# Histogram Features
# -----------------------------
def get_histogram_features(image):

    histogram = cv2.calcHist([image], [0], None, [256], [0,256])

    histogram = histogram.flatten()

    histogram = histogram / np.sum(histogram)

    return histogram

# -----------------------------
# Pixel Difference
# -----------------------------
def get_pixel_difference(image):

    horizontal = np.abs(np.diff(image, axis=1))

    vertical = np.abs(np.diff(image, axis=0))

    return np.mean(horizontal), np.mean(vertical)

# -----------------------------
# Complete Feature Extraction
# -----------------------------
def extract_features(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise Exception("Image not found.")

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


# -----------------------------
# Example
# -----------------------------
if __name__ == "__main__":

    image_path = "1.pgm"
    features, histogram = extract_features(image_path)

    print("Image Features\n")
    for key, value in features.items():
        print(f"{key:25}: {value:.4f}")

    # Prepare your single feature vector: shape (1, 6)
    feature_vector = np.array(list(features.values()), dtype=float).reshape(1, -1)

    # DUMMY DATASOURCE: To make MinMaxScaler work, we simulate a "mock" 
    # historical dataset showing the minimum and maximum possible values for your 6 features.
    # Structure: [Mean, Variance, Entropy, LSB Ratio, H_Diff, V_Diff]
    mock_min_features = [0,   0,   0, 0.0, 0,   0]
    mock_max_features = [255, 16384, 8, 1.0, 255, 255] 
    
    mock_dataset = np.array([mock_min_features, mock_max_features])

    # Initialize and fit the scaler on the expected ranges
    scaler = MinMaxScaler()
    scaler.fit(mock_dataset) 

    # Transform your actual single image feature vector
    normalized = scaler.transform(feature_vector)

    print("\nNormalized Feature Vector (0 to 1 Range):")
    print(normalized)