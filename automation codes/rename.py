import os

# 🔹 Path to the folder containing images
folder_path = "./data/ovary_diseases/Normal"
starting_number = 1570

# 🔹 Get a sorted list of image files (JPG, PNG, JPEG)
image_files = sorted(
    [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
)

# 🔹 Rename files sequentially
for i, filename in enumerate(image_files, start=starting_number):
    file_ext = os.path.splitext(filename)[1]  # Get file extension
    new_name = f"{i}{file_ext}"  # e.g., "1465.jpg", "1466.png"
    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, new_name)

    if old_path != new_path:  # Avoid renaming if names match
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {new_name}")

print("File renaming completed successfully!")
