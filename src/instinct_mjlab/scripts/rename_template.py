"""Rename template project — update all occurrences of 'instinct_mjlab' to a new name.

Original: InstinctLab/scripts/rename_template.py
Migrated: updated paths and default names for Instinct_mjlab project structure.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def rename_file_contents(root_dir_path: str, old_name: str, new_name: str, exclude_dirs: list = []):
    """Rename all instances of the old keyword to the new keyword in all files in the root directory.

    Args:
        root_dir_path (str): The root directory path.
        old_name (str): The old keyword to replace.
        new_name (str): The new keyword to replace with.
    """
    for dirpath, _, files in os.walk(root_dir_path):
        if any(exclude_dir in dirpath for exclude_dir in exclude_dirs):
            continue
        for file_name in files:
            if file_name == "rename_template.py":
                continue
            with open(os.path.join(dirpath, file_name)) as file:
                file_contents = file.read()
            file_contents = file_contents.replace(old_name, new_name)
            with open(os.path.join(dirpath, file_name), "w") as file:
                file.write(file_contents)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rename_template.py <new_name>")
        sys.exit(1)

    root_dir_path = str(Path(__file__).resolve().parent.parent.parent.parent)
    old_name = "instinct_mjlab"
    new_name = sys.argv[1]

    print(f"Warning, this script will rename all instances of '{old_name}' to '{new_name}' in {root_dir_path}.")
    proceed = input("Proceed? (y/n): ")

    if proceed.lower() == "y":
        # rename the instinct_mjlab folder
        src_pkg_dir = os.path.join(root_dir_path, "src", "instinct_mjlab")
        tgt_pkg_dir = os.path.join(root_dir_path, "src", new_name)
        if os.path.isdir(src_pkg_dir):
            os.rename(src_pkg_dir, tgt_pkg_dir)
            print(f"Renamed directory: {src_pkg_dir} -> {tgt_pkg_dir}")
        else:
            print(f"[WARN] Source directory not found: {src_pkg_dir}")
        # rename the file contents
        rename_file_contents(root_dir_path, old_name, new_name, exclude_dirs=[".git", "logs"])
        print(f"Done! Renamed all '{old_name}' -> '{new_name}'.")
    else:
        print("Aborting.")
