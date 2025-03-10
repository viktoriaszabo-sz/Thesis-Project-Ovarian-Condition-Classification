import os

def remove_mask_files(directory):
    # Loop through all files in the given directory
    for filename in os.listdir(directory):
        # Check if the filename contains 'mask'
        if '_mask' in filename:
            # Construct the full file path
            file_path = os.path.join(directory, filename)
            try:
                os.remove(file_path)  # Delete the file
                print(f"Removed file: {file_path}")
            except Exception as e:
                print(f"Error removing file {file_path}: {e}")

# Specify the directory where the files are located
directory = './data/BreastUltrasound'  # Replace with your actual directory path

# Call the function to remove files
remove_mask_files(directory)
