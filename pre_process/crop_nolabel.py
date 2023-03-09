import os
import cv2
import numpy as np

def crop_V1(img, im_name, store_path):
    root_path = "/mnt/sda1/6J_Anno/V1"
    split = im_name.split("_")
    zangwu_fabai_path = os.path.join(root_path, "{}/{}/{}".format(split[0], split[1], split[0] + "_" + split[1] + "_fabai_zangwu.bmp"))
    print(zangwu_fabai_path)

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

    #cv2.rectangle(img, (0, int(y0)), (int(img.shape[1]), int(y1)), (0, 255, 0), 3)
    #cv2.imwrite(os.path.join(store_path, im_name[:-4] + ".jpg"), img)

    mid = int((y0 + y1) / 2)
    y0_ = mid - 228
    y1_ = mid + 228
    
    crop_wh = 456
    overlap = int(456 / 3)
    w = img.shape[1]
    horizontal_num = int(np.ceil((w - overlap) / (crop_wh - overlap)))
    valid_w = horizontal_num * (crop_wh - overlap) + overlap
    hor_expand = valid_w - w
    valid_x0 = int(0 - hor_expand / 2)
    if valid_x0 < 0:
        img = np.concatenate((np.full((valid_h, -valid_x0, c), 255, dtype=np.uint8), img), axis=1)
    if valid_x0 + valid_w > w:
        img = np.concatenate((img, np.full((valid_h, -valid_x0, c), 255, dtype=np.uint8)), axis=1)
    
    for i1 in range(0, horizontal_num):
        crop_x0 = i1 * (crop_wh - overlap)
        crop_x1 = crop_x0 + crop_wh
        crop_y0 = y0_
        crop_y1 = y1_
        print(crop_x0, crop_y0, crop_x1, crop_y1)
        crop_img = img[crop_y0:crop_y1, crop_x0:crop_x1, :].copy()
    
        crop_img_path = os.path.join(store_path,
                os.path.basename(im_name)[:-4] + "_{}.jpg".format(str(i1)))
        cv2.imwrite(crop_img_path, crop_img)

def main():
    load_path = "/home/zyh/yolov5-v7.0/6J_dataset/V1/without_anno/all"
    crop_path = "/home/zyh/yolov5-v7.0/6J_dataset/V1/without_anno/crop"

    for im_name in os.listdir(load_path):
        # if im_name != "2022-07-22_28_zangwu.jpg":
        #     continue
        img_path = os.path.join(load_path, im_name)
        if im_name.endswith(".jpg"):
            print(im_name)
            img = cv2.imread(img_path)
            crop_V1(img, im_name, crop_path)

if __name__ == "__main__":
    main()