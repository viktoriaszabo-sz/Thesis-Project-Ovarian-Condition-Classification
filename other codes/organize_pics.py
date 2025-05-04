import os
import shutil
import pandas as pd

# Paths
csv_path = './data/ovary_diseases/_annotations1.csv'  # Path to your CSV
image_dir = './data/ovary_diseases/images'            # Path to your images
output_dir = './organized_images'                     # Where images will be moved/copied

# Load CSV
df = pd.read_csv(csv_path)

# Create output base folder if needed
os.makedirs(output_dir, exist_ok=True)

# Iterate through rows in CSV
for index, row in df.iterrows():
    filename = row['images']
    label = row['label']
    
    # Source and destination paths
    src_path = os.path.join(image_dir, filename)
    label_dir = os.path.join(output_dir, label)
    dst_path = os.path.join(label_dir, filename)

    # Make label directory if it doesn't exist
    os.makedirs(label_dir, exist_ok=True)

    # Copy or move the image
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)  # Use shutil.move() to move instead of copy
    else:
        print(f"Image not found: {src_path}")
