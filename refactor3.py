import os

project_root = r"f:\RunningProjects\AutoSegmentor\autosegmentor"
test_root = r"f:\RunningProjects\AutoSegmentor\autosegmentor\tests"
py_files = []
for root, _, files in os.walk(project_root):
    if ".venv" in root or ".git" in root or "segment_anything_2" in root or "co-tracker" in root:
        continue
    for file in files:
        if file.endswith(".py"):
            py_files.append(os.path.join(root, file))

replacements = {
    "from utils.Models": "from autosegmentor.models",
    "from utils.FileManagement": "from autosegmentor.file_management",
    "from utils.UserUI": "from autosegmentor.ui",
    "import utils.UserUI": "import autosegmentor.ui",
    "from ...UserUI.": "from ...ui.",
    "from ...FileManagement.": "from ...file_management.",
    "from ...Models.": "from ...models.",
    "from ...Tools.": "from ...tools.",
    "from ...Core.": "from ...core.",
    "from utils.Core": "from autosegmentor.core",
    "from utils.Tools": "from autosegmentor.tools",
    "from autosegmentor.utils.": "from autosegmentor.",
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

print("Done fixing extra imports.")
