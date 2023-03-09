# import cv2
# import os

# for f in os.listdir("./bmp_ori"):
#     if f[-4:] == ".bmp":
#         print(f)
#         img = cv2.imread("./bmp_ori/" + f)
#         # img = cv2.resize(img, (640, 427))
#         cv2.imwrite("./jpg/" + f[-4:] + ".jpg", img)
        
        
\
# coding:utf-8
import os
from PIL import Image


path = r'./bmp_ori'
for file in os.listdir(path):
    if file.endswith('.bmp'):
        filename = os.path.join(path, file)
        new_name = path +'\\' + file[:-4] + '.jpg'
        img = Image.open(filename)
        img.save(new_name)
        del img
        os.remove(filename)
        pass

