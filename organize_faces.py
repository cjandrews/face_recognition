#!/usr/bin/env python3
"""
Script to organize face images into the proper folder structure for face recognition.
"""

import os
import shutil
from pathlib import Path

def organize_faces(known_faces_dir: str):
    """Organize face images into subfolders based on filename patterns."""
    
    # Define name mappings based on your filenames
    name_mappings = {
        'ben a.jpg': 'Ben',
        'Bruno Aikido.jpg': 'Bruno',
        'chris a.jpg': 'Chris',
        'craig b.jpg': 'Craig',
        'eric b.jpg': 'Eric',
        'Evie.jpg': 'Evie',
        'jeanette a.jpg': 'Jeanette',
        'jim harrison.jpg': 'Jim',
        'joseph.jpg': 'Joseph',
        'sam aikido.jpg': 'Sam',
        'stuart b.jpg': 'Stuart'
    }
    
    # Process each image file
    for filename in os.listdir(known_faces_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            if filename in name_mappings:
                person_name = name_mappings[filename]
                person_folder = os.path.join(known_faces_dir, person_name)
                
                # Create person folder if it doesn't exist
                if not os.path.exists(person_folder):
                    os.makedirs(person_folder)
                    print(f"Created folder: {person_folder}")
                
                # Move the file
                source_path = os.path.join(known_faces_dir, filename)
                dest_path = os.path.join(person_folder, filename)
                
                if not os.path.exists(dest_path):
                    shutil.move(source_path, dest_path)
                    print(f"Moved {filename} to {person_name}/")
                else:
                    print(f"File already exists in {person_name}/: {filename}")
            else:
                print(f"No mapping found for: {filename}")

if __name__ == "__main__":
    known_faces_dir = r"C:\data\known_faces"
    
    if not os.path.exists(known_faces_dir):
        print(f"Error: Directory {known_faces_dir} does not exist!")
        exit(1)
    
    print("Organizing face images...")
    organize_faces(known_faces_dir)
    print("Done!") 