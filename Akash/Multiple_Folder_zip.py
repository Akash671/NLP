# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 06:28:30 2024

@author: PCAT
"""

import os
import zipfile

def zip_subfolders(parent_folder):
    """Zips all subfolders within a parent folder, keeping the original folder names.

    Args:
        parent_folder: The path to the parent folder containing the subfolders to zip.
    """

    for folder_name in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder_name)

        # Check if it's a directory (not a file)
        if os.path.isdir(folder_path):
            zip_file_path = os.path.join(parent_folder, folder_name + ".zip")  # Zip file in the same parent folder

            try:
                with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, _, files in os.walk(folder_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, folder_path)  # Maintain relative paths within the zip
                            zipf.write(file_path, arcname=arcname)

                print(f"Successfully zipped {folder_name} to {zip_file_path}")

            except Exception as e:
                print(f"Error zipping {folder_name}: {e}")


# Example usage:
parent_folder = r"\\10.97.116.141\akash\ATT\59\5552\Scripts\Logs"  # Replace with the actual path to your parent folder
zip_subfolders(parent_folder)