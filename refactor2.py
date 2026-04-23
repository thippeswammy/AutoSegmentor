import os
import shutil

base_dir = r"f:\RunningProjects\AutoSegmentor\autosegmentor"
utils_dir = os.path.join(base_dir, "utils")

mapping = {
    "Core": "core",
    "FileManagement": "file_management",
    "Models": "models",
    "Tools": "tools",
    "UserUI": "ui"
}

def move_recursively(src, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        if item == "__pycache__":
            continue
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            move_recursively(s, d)
        else:
            if not os.path.exists(d):
                shutil.move(s, d)
                print(f"Moved {s} to {d}")
            else:
                print(f"File {d} already exists, not overwriting")

for old, new in mapping.items():
    old_path = os.path.join(utils_dir, old)
    new_path = os.path.join(base_dir, new)
    if os.path.exists(old_path):
        move_recursively(old_path, new_path)

print("Done moving recursively.")
