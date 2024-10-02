import os


def rename_files_in_directory(directory, prefix="_", extension=".png"):
    """
    Rename files in the specified directory.

    Parameters:
    - directory: str - The path to the directory containing the files to rename.
    - prefix: str - The new prefix for the renamed files.
    - extension: str - The extension of the files to rename (e.g., .jpg, .png).
    """
    try:
        # List all files in the directory
        for index, filename in enumerate(os.listdir(directory)):
            # Check if the file has the specified extension
            if filename.endswith(extension):
                old_file_path = os.path.join(directory, filename)
                new_file_path = old_file_path[:6] + prefix + old_file_path[16:]
                os.rename(old_file_path, new_file_path)
    except Exception as e:
        print(f"Error occurred: {e}")


# Usage
directory_path = "road2"  # Replace with your directory path
rename_files_in_directory(directory_path, prefix="road2_", extension=".png")
