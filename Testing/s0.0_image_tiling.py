from PIL import Image
import os
import pickle
import shutil


with open('./grid_size.pkl', 'rb') as f:
    grid_size = pickle.load(f)
 
print("grid_size = ",grid_size)  

with open('./image_path.pkl', 'rb') as f:
    image_path = pickle.load(f)


def empty_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
                
def cut_image_into_tiles(image_path, output_folder):
    image = Image.open(image_path)
    width, height = image.size
    tile_width = width // grid_size
    tile_height = height // grid_size
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i in range(grid_size):
        for j in range(grid_size):
            left = j * tile_width
            upper = i * tile_height
            right = (j + 1) * tile_width
            lower = (i + 1) * tile_height
            tile = image.crop((left, upper, right, lower))
            tile_filename = f"tile_{i}_{j}.png"
            tile_path = os.path.join(output_folder, tile_filename)
            tile.save(tile_path)

input_image_path = image_path
output_folder_path = "./output_tiles"
if os.path.exists(output_folder_path):
    empty_folder(output_folder_path)
cut_image_into_tiles(input_image_path, output_folder_path)
