import os
from PIL import Image

folder_path = 'Bradd-PGD-test-0.07'

if os.path.isdir(folder_path):
    print(f"\nImage sizes in: {folder_path}")
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                print(f"{filename}: {width}x{height}")
        except Exception as e:
            print(f"{filename}: failed to open ({e})")
else:
    print(f"Directory '{folder_path}' does not exist.")
