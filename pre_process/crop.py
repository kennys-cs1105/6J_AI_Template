import cv2
import os
import numpy as np


def draw_boxes(img_draw, boxes):
    for class_id, x0, y0, x1, y1 in boxes:
        cv2.rectangle(img_draw, (x0, y0), (x1, y1), (0, 255, 0), 3)

    return

def decode_txt(txt_path, img_shape):
    h, w, c = img_shape
    with open(txt_path, "r") as ftxt:
        annos = ftxt.readlines()
        boxes = []
        for anno in annos:
            anno0 = anno.strip().split(" ")
            class_id = int(anno0[0])
            x_center = float(anno0[1])
            y_center = float(anno0[2])
            width = float(anno0[3])
            height = float(anno0[4])
            x0 = round((x_center - (width / 2.0)) * w)
            x1 = round((x_center + (width / 2.0)) * w)
            y0 = round((y_center - (height / 2.0)) * h)
            y1 = round((y_center + (height / 2.0)) * h)
            boxes.append([class_id, x0, y0, x1, y1])

    return np.array(boxes)

def store_txt(boxes, store_path):
    with open(store_path, "w") as ftxt:
        for box in boxes:
            ftxt.write("{} {} {} {} {}\n".format(box[0], box[1], box[2], box[3], box[4]))

def encode_txt(boxes, store_path, img_shape):
    h, w, c = img_shape
    print("img's h:", h)
    print(img_shape)
    with open(store_path, "w") as ftxt:
        for box in boxes:
            class_id = box[0]
            x0 = box[1]
            y0 = box[2]
            x1 = box[3]
            y1 = box[4]
            width = (x1 - x0) / w
            height = (y1 - y0) / h
            x_center = (x1 + x0) / (2 * w)
            y_center = (y1 + y0) / (2 * h)
            ftxt.write("{} {} {} {} {}\n".format(class_id, x_center, y_center, width, height))

def have_intersection(box0, box1):
    x_inter = max(box0[0], box1[0]) < min(box0[2], box1[2])
    y_inter = max(box0[1], box1[1]) < min(box0[3], box1[3])

    return x_inter and y_inter

def clamp(min_v, max_v, v):

    return max(min(max_v, v), min_v)

# 1.找出boxes中min_y0和max_y1，计算出delta
# 2.根据delta, crop_wh和overlaprop_v0(vertical_min, crop_wh, overlap, img, boxes, box_min_side, store_path, im_name):
    # h, w, c = img.shape
    # boxes0 = boxes.copy()
    # mi计算出vertical_num，不能整除则向上取整。需要满足切图时不能有剩余和不足
# 3.以min_y0和max_y1的1/2处为中点，截取图像，如果原图尺寸不满足，则填充255
# 4.根据w, crop_wh和overlap计算horizontal_num，同样向上取整，不足处填充255
# 5.在xy方向遍历，截取图像，如果其中包含object，则保存图像和标签，标签需要进行坐标处理
#
# 缺点：不能针对标签类别控制数量，不能控制缺陷在小图中出现的位置，缺陷会被切开
def crop_v0(vertical_min, crop_wh, overlap, img, boxes, box_min_side, store_path, im_name):
    h, w, c = img.shape
    print(h, w, c)
    boxes0 = boxes.copy()
    min_y0 = np.min(boxes0[:, 2])
    max_y1 = np.max(boxes0[:, -1])
    print(min_y0, max_y1)
    delta = max_y1 - min_y0
    vertical_num = vertical_min if delta <= crop_wh else int(np.ceil((delta - overlap) / (crop_wh - overlap))) + 1
    valid_h = vertical_num * (crop_wh - overlap) + overlap
    ver_expand = valid_h - delta
    valid_y0 = int(min_y0 - ver_expand / 2)
    img0 = img[valid_y0:valid_y0 + valid_h, :, :].copy()
    if valid_y0 < 0:
        img0 = np.concatenate((np.full((-valid_y0, w, c), 255, dtype=np.uint8), img0), axis=0)
    if valid_y0 + valid_h > h:
        img0 = np.concatenate((img0, np.full((valid_y0 + valid_h - h, w, c), 255, dtype=np.uint8)), axis=0)
    boxes0[:, 2] -= valid_y0
    boxes0[:, 4] -= valid_y0
    print("After vertical processing")
    print(img0.shape)
    print(min_y0, max_y1, delta, valid_y0, valid_h, vertical_num)

    horizontal_num = int(np.ceil((w - overlap) / (crop_wh - overlap)))
    valid_w = horizontal_num * (crop_wh - overlap) + overlap
    hor_expand = valid_w - w
    valid_x0 = int(0 - hor_expand / 2)
    if valid_x0 < 0:
        img0 = np.concatenate((np.full((valid_h, -valid_x0, c), 255, dtype=np.uint8), img0), axis=1)
    if valid_x0 + valid_w > w:
        img0 = np.concatenate((img0, np.full((valid_h, -valid_x0, c), 255, dtype=np.uint8)), axis=1)
    boxes0[:, 1] -= valid_x0
    boxes0[:, 3] -= valid_x0
    print("After horizontal processing")
    print("img0.shape", img0.shape)
    print(valid_x0, valid_w, horizontal_num)

    for i0 in range(0, vertical_num):
        for i1 in range(0, horizontal_num):
            crop_x0 = i1 * (crop_wh - overlap)
            crop_x1 = crop_x0 + crop_wh
            crop_y0 = i0 * (crop_wh - overlap)
            crop_y1 = crop_y0 + crop_wh
            crop_img = img0[crop_y0:crop_y1, crop_x0:crop_x1, :].copy()
            print("crop_img.shape", img0[crop_y0:crop_y1, crop_x0:crop_x1, :].shape)
            boxes1 = []
            for box in boxes0:
                class_id, x0, y0, x1, y1 = box
                if have_intersection([x0, y0, x1, y1], [crop_x0, crop_y0, crop_x1, crop_y1]):
                    x0 -= crop_x0
                    y0 -= crop_y0
                    x1 -= crop_x0
                    y1 -= crop_y0
                    x0 = clamp(0, crop_wh, x0)
                    y0 = clamp(0, crop_wh, y0)
                    x1 = clamp(0, crop_wh, x1)
                    y1 = clamp(0, crop_wh, y1)
                    if x1 - x0 <= box_min_side or y1 - y0 <= box_min_side:
                        continue
                    boxes1.append([class_id, x0, y0, x1, y1])
            if len(boxes1) != 0:
                crop_txt_path = os.path.join(store_path,
                                             os.path.basename(im_name)[:-4] + "_{}_{}_{}.txt".format(
                                                 str(i0), str(i1), crop_wh))
                # store_txt(boxes1, crop_txt_path)
                draw_boxes(crop_img, boxes1)
                encode_txt(boxes1, crop_txt_path, crop_img.shape)
                crop_img_path = os.path.join(store_path,
                                             os.path.basename(im_name)[:-4] + "_{}_{}_{}.jpg".format(
                                                 str(i0), str(i1), crop_wh))
                print(crop_img_path)
                cv2.imwrite(crop_img_path, crop_img)

    return

def main():
    #load_path = "./JPEGImages_ORI"
    load_path = "./bmp_ori"
    #crop_wh = 912
    crop_wh = 912

    store_path = "./train2/{}".format(crop_wh)

    assert crop_wh % 912 == 0
    overlap = int(crop_wh / 3)
    vertical_min = 2 if crop_wh == 912 else 1
    box_min_side = int(5 * (crop_wh / 912))

    for im_name in os.listdir(load_path):
        # if im_name != "2022-11-10_132_zangwu.jpg":
        #     continue
        if not im_name.endswith(".jpg"):
            continue
        print("kagnshuai" + im_name)
        img_path = os.path.join(load_path, im_name)
        txt_path = os.path.join(load_path, os.path.basename(im_name)[:-4] + ".txt")
        if not os.path.exists(txt_path):
            continue

        img = cv2.imread(img_path)
        boxes = decode_txt(txt_path, img.shape)
        print(boxes)

        crop_v0(vertical_min, crop_wh, overlap, img, boxes, box_min_side, store_path, im_name)

if __name__ == "__main__":
    main()