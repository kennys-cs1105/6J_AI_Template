import cv2
import numpy as np
import os
import sys


CROP_WH = 456
CROP_C = 3
OVERLAP = int(456 / 3)


def get_para(root_path, im_name):
    split = im_name.split("_")
    zangwu_fabai_path = os.path.join(root_path, "{}/{}/{}".format(split[0], split[1], split[0] + "_" + split[1] + "_fabai_zangwu.bmp"))
    # zangwu_fabai_path = os.path.join(root_path, "{}/{}/{}".format(split[0], split[1], split[0] + "_" + split[1] + "_fabai_zangwu.bmp"))
    img_ = cv2.imread(zangwu_fabai_path)
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
    y0 = np.min(coords[:, 1])
    y1 = np.max(coords[:, 1])

    mid = int((y0 + y1) / 2)
    y0_ = mid - 228
    y1_ = mid + 228
    
    w = img_.shape[1]
    horizontal_num = int(np.ceil((w - OVERLAP) / (CROP_WH - OVERLAP)))
    valid_w = horizontal_num * (CROP_WH - OVERLAP) + OVERLAP
    hor_expand = valid_w - w
    valid_x0 = int(0 - hor_expand / 2)
    
    return valid_x0, y0_, img_.shape


def merge_boxes(boxes, x0_, y0_, img_shape):
    boxes_res = []
    for i, boxes0 in enumerate(boxes):
        if not len(boxes0):
            continue
        for box in boxes0:
            class_id = box[0]
            x0 = box[1]
            y0 = box[2]
            x1 = box[3]
            y1 = box[4]
            conf = box[5]
            y0 += y0_
            y1 += y0_
            x0 += i * (CROP_WH - OVERLAP)
            x0 += x0_
            x0 = 0 if x0 < 0 else x0
            x1 += i * (CROP_WH - OVERLAP)
            x1 += x0_
            x1 = 0 if x1 < 0 else x1
            boxes_res.append([class_id, x0, y0, x1, y1, conf])
            

    return boxes_res


def decode_txts(imgs, txt_path):
    boxes = []
    for fimg in imgs:
        ftxt = fimg.replace(".jpg", ".txt")
        ftxt = os.path.join(txt_path, ftxt)
        if not os.path.exists(ftxt):
            boxes.append([])
            continue
        with open(ftxt, "r") as f:
            annos = f.readlines()
            boxes0 = []
            for anno in annos:
                anno0 = anno.strip().split(" ")
                class_id = int(anno0[0])
                x_center = float(anno0[1])
                y_center = float(anno0[2])
                width = float(anno0[3])
                height = float(anno0[4])
                conf = float(anno0[5])
                x0 = round((x_center - (width / 2.0)) * CROP_WH)
                x1 = round((x_center + (width / 2.0)) * CROP_WH)
                y0 = round((y_center - (height / 2.0)) * CROP_WH)
                y1 = round((y_center + (height / 2.0)) * CROP_WH)
                boxes0.append([class_id, x0, y0, x1, y1, conf])
        boxes.append(boxes0)

    return boxes


def plot_boxes(img_plot, boxes):
    for class_id, x0, y0, x1, y1, conf in boxes:
        cv2.rectangle(img_plot, (x0, y0), (x1, y1), (0, 255, 0), 3)

    return img_plot


def main():
    # root_path = "/mnt/sda1/6J_Anno/V2"
    root_path = "/home/kangshuai/6J_Anno/V2"
    # img_path = "/home/zyh/yolov5-v7.0/6J_dataset/TEST/V2_dev"
    img_path = "/home/kangshuai/Desktop/industry_detec/yolov5-v7.0/6J_dataset/ori/TEST/dev"
    # txt_path = "/home/zyh/yolov5-v7.0/6J_dataset/TEST/exp/labels"
    txt_path = "/home/kangshuai/Desktop/industry_detec/yolov5-v7.0/runs/detect/zyh_without_anno/labels"
    # plot_path = "/home/zyh/yolov5-v7.0/6J_dataset/TEST/plot"
    plot_path = "/home/kangshuai/Desktop/industry_detec/yolov5-v7.0/6J_dataset/ori/TEST/plot"
    img_list = os.listdir(img_path)
    img_list.sort()
    
    count = 0
    name = ""
    imgs = []
    for f in img_list:
        count += 1
        imgs.append(f)
        if count == 1:
            name = f[:-6]
        elif count == 4:
            count = 0
            x0_, y0_, img_shape = get_para(root_path, f)
            boxes = decode_txts(imgs, txt_path)
            boxes = merge_boxes(boxes, x0_, y0_, img_shape)
            imgs.clear()
            if not len(boxes):                
                continue

            print(name)
            print(boxes)
            
            split = name.split("_")
            img_path = os.path.join(root_path, "{}/{}/{}".format(split[0], split[1], name + ".bmp"))
            img_plot = cv2.imread(img_path)
            img_plot = plot_boxes(img_plot, boxes)
            cv2.imwrite(os.path.join(plot_path, name + ".jpg"), img_plot)
        else:
            if f[:-6] != name:
                print("Error")
                sys.exit(1)


if __name__ == "__main__":
    main()









