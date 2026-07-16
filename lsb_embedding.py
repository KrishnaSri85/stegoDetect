from PIL import Image
import os

# -----------------------------
# Folder Paths
# -----------------------------
input_folder = "dataset/cover_png"
output_folder = "dataset/stego"

os.makedirs(output_folder, exist_ok=True)


# -----------------------------
# Convert message to binary
# -----------------------------
def message_to_binary(message):
    binary = ""

    for character in message:
        binary += format(ord(character), "08b")

    # End marker so we know where the message stops
    binary += "1111111111111110"

    return binary


# -----------------------------
# Embed message into image
# -----------------------------
def embed_message(image_path, output_path, secret_message):

    image = Image.open(image_path)
    image = image.convert("RGB")

    pixels = list(image.getdata())

    binary_message = message_to_binary(secret_message)

    message_index = 0

    new_pixels = []

    for pixel in pixels:

        r, g, b = pixel

        if message_index < len(binary_message):
            r = (r & 254) | int(binary_message[message_index])
            message_index += 1

        if message_index < len(binary_message):
            g = (g & 254) | int(binary_message[message_index])
            message_index += 1

        if message_index < len(binary_message):
            b = (b & 254) | int(binary_message[message_index])
            message_index += 1

        new_pixels.append((r, g, b))

    image.putdata(new_pixels)

    image.save(output_path)

    return len(binary_message)


# -----------------------------
# Process all images
# -----------------------------
image_files = sorted(os.listdir(input_folder))

count = 0

for index, filename in enumerate(image_files, start=1):

    input_path = os.path.join(input_folder, filename)

    output_name = f"img{index:03d}_stego.png"

    output_path = os.path.join(output_folder, output_name)

    secret = f"Secret image number {index}"

    bits = embed_message(input_path, output_path, secret)

    print(f"{filename} ---> {output_name}")
    print(f"Hidden Message : {secret}")
    print(f"Bits Embedded : {bits}")
    print("-" * 50)

    count += 1

print("\n===================================")
print("LSB Embedding Completed Successfully!")
print(f"Total Stego Images Created : {count}")
print("Saved inside : dataset/stego")
print("===================================")