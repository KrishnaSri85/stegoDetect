import math
from statistics import mean, pvariance, pstdev
from PIL import Image


def extract_features(image_path):
    with Image.open(image_path).convert("L") as image:
        width, height = image.size
        pixels = list(image.getdata())

    total = len(pixels)
    if total == 0:
        raise ValueError("Image has no readable pixel data.")

    frequencies = {}
    for value in pixels:
        frequencies[value] = frequencies.get(value, 0) + 1

    entropy = -sum(
        (count / total) * math.log2(count / total)
        for count in frequencies.values()
    )

    lsb_ratio = sum(value & 1 for value in pixels) / total

    # These directional differences match the columns in data/*.csv and are
    # therefore required by the trained classifier.
    horizontal_differences = [
        abs(pixels[row * width + column] - pixels[row * width + column + 1])
        for row in range(height)
        for column in range(width - 1)
    ]
    vertical_differences = [
        abs(pixels[row * width + column] - pixels[(row + 1) * width + column])
        for row in range(height - 1)
        for column in range(width)
    ]

    return {
        "entropy": round(entropy, 4),
        "mean": round(mean(pixels), 4),
        "variance": round(pvariance(pixels), 4),
        "std_dev": round(pstdev(pixels), 4),
        "lsb_ratio": round(lsb_ratio, 4),
        "horizontal_difference": round(mean(horizontal_differences), 4) if horizontal_differences else 0.0,
        "vertical_difference": round(mean(vertical_differences), 4) if vertical_differences else 0.0,
    }
