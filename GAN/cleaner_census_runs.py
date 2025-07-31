import os
import shutil

# Base directory to start searching (change '.' to your target directory)
base_dir = '.'

# Iterate through all directories in the base directory
for dir_name in os.listdir(base_dir):
    dir_path = os.path.join(base_dir, dir_name)

    # Check if it's a directory and starts with "census"
    if os.path.isdir(dir_path) and dir_name.startswith("census"):
        # Check if the directory contains a subdirectory named "data"
        contains_data_dir = os.path.isdir(os.path.join(dir_path, "data"))

        # If no "data" subdirectory is found, delete the directory
        if not contains_data_dir:
            print(f"Deleting directory: {dir_path}")
            shutil.rmtree(dir_path)  # Deletes the directory and its contents
        else:
            print(f"'data' subdirectory found in: {dir_path}, keeping directory.")