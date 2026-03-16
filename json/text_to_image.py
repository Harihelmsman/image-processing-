import base64

# read the base64 string
with open("text.txt", "r") as f:
    base64_data = f.read()

# decode
image_bytes = base64.b64decode(base64_data)

# save image
with open("output_image.jpg", "wb") as f:
    f.write(image_bytes)

print("Image restored successfully!")