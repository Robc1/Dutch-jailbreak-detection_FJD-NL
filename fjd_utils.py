"""
fjd_utils.py summary:

Shared utilities for the FJD jailbreak-detection analysis scripts:
    - fjd-analysis.py
    - detection.py

Contains folder/file discovery helpers and the benign-file check that were
previously duplicated in both scripts.
"""

import os
IGNORE_FOLDERS = {'complete', 'trash', 'reporting'}


# ===========================================================================
# Benign-file detection
# ===========================================================================

#Returns "True" if the codefile basename contains "benign" (case-insensitive).
def is_benign_file(codefile: str) -> bool:
    return 'benign' in os.path.basename(codefile).lower()


# ===========================================================================
# Folder / file discovery
# ===========================================================================

# Sorts immediate subdirectories of root, ignoring the folders listed in "IGNORE_FOLDERS".
def collect_subfolders(root: str) -> list[str]:
    return sorted([
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
        and d.lower() not in IGNORE_FOLDERS
    ])


# Recursively finds all .json files under "root_path".
def find_json_files(root_path: str) -> list[str]:
    json_files = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d.lower() not in IGNORE_FOLDERS]
        for filename in filenames:
            if filename.endswith('.json'):
                json_files.append(os.path.join(dirpath, filename))
    return json_files


# Returns the path to the temperature subfolder whose name matches "temperature_name" (case-insensitive), 
# or None if not found.
def find_exact_temperature_folder(run_folder: str, temperature_name: str) -> str | None:
    target = temperature_name.lower()
    for d in os.listdir(run_folder):
        full = os.path.join(run_folder, d)
        if (os.path.isdir(full)
                and d.lower() == target
                and d.lower() not in IGNORE_FOLDERS):
            return full
    return None

 
# Numeric sort key for temperature folders, based on the trailing number in the name. 
# The function accepts either a full path ('.../Temperature 0.5') or a bare folder name ('Temperature 0.5'). 
def temperature_sort_key(folder_path_or_name: str) -> float:
    name = os.path.basename(folder_path_or_name)
    parts = name.strip().split()
    try:
        return float(parts[-1])
    except (ValueError, IndexError):
        return float('inf')
    

# Returns all immediate subdirectories of run_folder whose names start with "temperature"(case-insensitive), 
# sorted numerically by trailing number.
def find_temperature_folders(run_folder: str) -> list[str]:
    return sorted([
        os.path.join(run_folder, d)
        for d in os.listdir(run_folder)
        if os.path.isdir(os.path.join(run_folder, d))
        and d.lower().startswith('temperature')
        and d.lower() not in IGNORE_FOLDERS
    ], key=temperature_sort_key)