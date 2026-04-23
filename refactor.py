import os
import shutil
import glob

base_dir = r"f:\RunningProjects\AutoSegmentor\autosegmentor"
utils_dir = os.path.join(base_dir, "utils")

mapping = {
    "Core": "core",
    "FileManagement": "file_management",
    "Models": "models",
    "Tools": "tools",
    "UserUI": "ui"
}

# Move files
for old, new in mapping.items():
    old_path = os.path.join(utils_dir, old)
    new_path = os.path.join(base_dir, new)
    if os.path.exists(old_path):
        for item in os.listdir(old_path):
            s = os.path.join(old_path, item)
            d = os.path.join(new_path, item)
            if not os.path.exists(d):
                shutil.move(s, d)
            else:
                print(f"Skipping {s} as {d} already exists")

# Move pipeline.py
if os.path.exists(os.path.join(utils_dir, "pipeline.py")):
    shutil.move(os.path.join(utils_dir, "pipeline.py"), os.path.join(base_dir, "pipeline.py"))

# Refactor imports
# We will check all .py files in f:\RunningProjects\AutoSegmentor
project_root = r"f:\RunningProjects\AutoSegmentor"
py_files = []
for root, _, files in os.walk(project_root):
    if ".venv" in root or ".git" in root or "segment_anything_2" in root or "co-tracker" in root:
        continue
    for file in files:
        if file.endswith(".py"):
            py_files.append(os.path.join(root, file))

replacements = {
    "autosegmentor.core": "autosegmentor.core",
    "autosegmentor.file_management": "autosegmentor.file_management",
    "autosegmentor.models": "autosegmentor.models",
    "autosegmentor.tools": "autosegmentor.tools",
    "autosegmentor.ui": "autosegmentor.ui",
    "autosegmentor.pipeline": "autosegmentor.pipeline",
    "from .core.": "from .core.",
    "from .file_management.": "from .file_management.",
    "from .models.": "from .models.",
    "from .tools.": "from .tools.",
    "from .ui.": "from .ui.",
    "from ..core.": "from ..core.",
    "from ..file_management.": "from ..file_management.",
    "from ..models.": "from ..models.",
    "from ..tools.": "from ..tools.",
    "from ..ui.": "from ..ui.",
    "from .": "from .",
    "from ..": "from ..",
}

for py_file in py_files:
    with open(py_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    new_content = content
    for old_str, new_str in replacements.items():
        new_content = new_content.replace(old_str, new_str)
        
    if new_content != content:
        with open(py_file, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Updated imports in {py_file}")

print("Done refactoring.")
