import cv2
import os
import numpy as np

def draw_boxes(img_draw, boxes):
    for class_id, x0, y0, x1, y1, conf in boxes:
        cv2.rectangle(img_draw, (x0, y0), (x1, y1), (0, 255, 0), 3)

    return

img_path = "/home/kangshuai/Desktop/industry_detec/yolov5-v7.0/6J_dataset/ori/V2/without_anno/all"
txt_path = "/home/kangshuai/Desktop/industry_detec/yolov5-v7.0/6J_dataset/ori/50_save"

for img_file in os.listdir(img_path):
    for txt_file in os.listdir(txt_path):
        if txt_file[:-6] == img_file[:-4]:
            with open(os.path.join(txt_path, txt_file), "r") as ftxt:
                annos = ftxt.readlines()
                boxes = []
                for anno in annos:
                    anno = anno.strip().split(",")
                    class_id = int(anno[0])
                    x0 = float(anno[1])
                    y0 = float(anno[2])
                    x1 = float(anno[3])
                    y1 = float(anno[4])
                    conf = float(anno[5])
                    
                    draw_x0 = round(x0 * 1368)
                    draw_y0 = round(y0 * 912)
                    draw_x1 = round(x1 * 1368)
                    draw_y1 = round(y1 * 912)
                    print(draw_x0, draw_y0, draw_x1, draw_y1)
                    boxes.append([class_id, draw_x0, draw_y0, draw_x1, draw_y1, conf])
                    img = cv2.imread(os.path.join(img_path, img_file))
                    print(img.shape)
                    for class_id, draw_x0, draw_y0, draw_x1, draw_y1, conf in boxes:
                        img_draw = cv2.rectangle(img, (draw_x0, draw_y0), (draw_x1, draw_y1), (0, 255, 0), 3)
                    cv2.imwrite(os.path.join(txt_path, img_file), img_draw)







    # with open(os.path.join(txt_path, txt_name), "r") as ftxt:
    #     annos = ftxt.readlines()
    #     boxes = []
    #     for anno in annos:
    #         anno = anno.strip().split(",")
    #         class_id = int(anno[0])
    #         x0 = float(anno[1])
    #         y0 = float(anno[2])
    #         x1 = float(anno[3])
    #         y1 = float(anno[4])
    #         conf = float(anno[5])
    #         boxes.append([class_id, x0, y0, x1, y1, conf])




