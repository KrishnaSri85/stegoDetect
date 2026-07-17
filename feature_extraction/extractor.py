import math
from statistics import mean, pvariance, pstdev
from PIL import Image


def extract_features(image_path):
    with Image.open(image_path).convert("L") as image:
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

    return {
        "entropy": round(entropy, 4),
        "mean": round(mean(pixels), 4),
        "variance": round(pvariance(pixels), 4),
        "std_dev": round(pstdev(pixels), 4),
        "lsb_ratio": round(lsb_ratio, 4),
    }