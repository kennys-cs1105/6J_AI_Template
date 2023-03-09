from click import FloatRange
import cv2
from cv2 import findContours
import numpy as np
import os

img_ = cv2.imread('/home/kangshuai/Desktop/industry_detec/yolov5-v7.0/6J_dataset/ori/second_dataset/bmp_ori/2022-11-10_1_fabai_zangwu.jpg')
img_ = cv2.GaussianBlur(img_, (5, 5), sigmaX = 1.0)
edges = cv2.Canny(img_, 50, 150)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, 2)

valid_contours = []
for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    arc = cv2.arcLength(contours[i], False)
    if arc > 50:
        valid_contours.append(contours[i])

coords = np.squeeze(np.concatenate(valid_contours, axis=0), axis=1)
ymin = np.min(coords[:, 1])
ymax = np.max(coords[:, 1])
print(ymin)
print(ymax)
    
    #cv2.rectangle(img, (0, int(y0)), (int(img.shape[1]), int(y1)), (0, 255, 0), 3)
    #cv2.imwrite(os.path.join(store_path, im_name[:-4] + ".jpg"), img)

mid = int((ymin + ymax) / 2)
y0_ = mid - 228
y1_ = mid + 228

def find_txt(store_path):
    root_path = "/home/kangshuai/Desktop/industry_detec/yolov5-v7.0/runs/detect/zyh_without_anno/labels"
    for file_name in os.listdir(root_path):
        print(file_name)
        for i in range(0, 5):
            # new_name = file_name[:-10] + "_{}_456".format(i) + ".txt"
            new_name = file_name[:-6] + "_{}".format(i) + ".txt"
            if not os.path.exists(os.path.join(root_path, new_name)):
                continue
            with open(os.path.join(root_path, new_name), "r") as ftxt:
                annos = ftxt.readlines()
                boxes = []
                for anno in annos:
                    anno = anno.strip().split(" ")
                    class_id = int(anno[0])
                    x0 = float(anno[1])
                    y0 = float(anno[2])
                    x1 = float(anno[3])
                    y1 = float(anno[4])
                    conf = float(anno[5])
                    # raw_image.w = 1368, raw_image.h = 912
                    # crop_image.w = crop_image.h = 456
                    # crop_wh - overlap = 304
                    crop_wh = 456
                    overlap = int(crop_wh / 3)

                    x0_new = (x0 * 456 + i * (crop_wh - overlap)) / 1368 
                    y0_new = (y0 * 456 + y0_) / 912
                    x1_new = (x1 * 456 + i * (crop_wh - overlap)) / 1368
                    y1_new = (y1 * 456 + y0_  ) / 912
                    # y1_new = (y0_new * 912 + 456) / 912 
                    boxes.append([class_id, x0_new, y0_new, x1_new, y1_new, conf])
         
                    # if len(boxes) != 0:
                    txt_path = os.path.join(store_path, new_name)
                    txt = open(txt_path, "w")
                    for line in boxes:
                        line = str(line).strip("[]")
                        txt.write(line + "\n")
                        # txt.writelines(boxes)
                    txt.close()
    # print(len(boxes))
            # return np.array(boxes) 

# box0 = recover0(txt_path)
# box1 = recover1(txt_path)

# print(box0)
# print(box1)
store_path = "/home/kangshuai/Desktop/industry_detec/yolov5-v7.0/6J_dataset/ori/50_save"
find_txt(store_path=store_path)