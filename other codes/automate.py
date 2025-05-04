import os
import pandas as pd

# Set the folder path containing images
folder_path = "C:/Users/vikiv/Desktop/Thesis-lab/data/ovary_diseases/Normal"

# Get all image filenames
image_titles = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

# Create a DataFrame
df = pd.DataFrame(image_titles, columns=['Image Title'])

# Save to Excel
excel_path = "image_titles.xlsx"
df.to_excel(excel_path, index=False)



print(f"Excel file saved: {excel_path}")
