import os
import shutil

if __name__ == "__main__":
    path = "./Crop456"
    destination = "./Crop_all"
    for filename in os.listdir(path):
        new_name = filename[:-4] + "_456" + filename[-4:]
        shutil.move(os.path.join(path, filename), os.path.join(destination, new_name))
        
        print(os.path.join(path, filename) + " -> " + os.path.join(destination, new_name))
        
        
