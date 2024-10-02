import os

# Specify the directory containing the images
directory = 'rendered_frames'  # Replace with your folder path

# Loop through all files in the directory
for filename in os.listdir(directory):
    # Check if the file starts with 'road2_' and ends with '.png'
    if filename.startswith('road2_') and filename.endswith('.png'):
        # Create the new filename by replacing 'road2_' with 'road3_'
        new_filename = filename.replace('road2_', 'road3_')

        # Define full paths for renaming
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)

        # Rename the file
        os.rename(old_file, new_file)
        print(f'Renamed: {filename} to {new_filename}')
