"""
Canny边缘检测算子
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

img = cv2.imread("1.jpg", flags=0)
print(img.shape)

# # 高斯核低通滤波器，sigmaY 缺省时 sigmaY=sigmaX
# kSize = (5, 5)
# imgGauss1 = cv2.GaussianBlur(img, kSize, sigmaX=1.0)  # sigma=1.0
# imgGauss2 = cv2.GaussianBlur(img, kSize, sigmaX=10.0)  # sigma=2.0

# # 高斯差分算子 (Difference of Gaussian)
# imgDoG = imgGauss2 - imgGauss1  # sigma=1.0, 10.0

# # Canny 边缘检测， kSize 为高斯核大小，t1,t2为阈值大小
# t1, t2 = 50, 150
# imgCanny = cv2.Canny(imgGauss1, t1, t2)

# cnts, _ = cv2.findContours(imgCanny.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# cnt = cnts[0]
# print(cnts)

# for i in range(0, len(cnts)):
#     x, y, w, h = cv2.boundingRect(cnts[i])
#     contourPic = cv2.rectangle(img, (x,y), (x+w,y+h), (153,153,0), 5)
#     cv2.imwrite("2.jpg",contourPic)

#contourPic = cv2.drawContours(img, cnts, -1, (0, 0, 255), thickness=cv2.FILLED,maxLevel=1)

# plt.figure(figsize=(10, 6))
# plt.subplot(141), plt.title("Origin"), plt.imshow(img, cmap='gray'), plt.axis('off')
# plt.subplot(142), plt.title("DoG"), plt.imshow(imgDoG, cmap='gray'), plt.axis('off')
# plt.subplot(143), plt.title("Canny"), plt.imshow(imgCanny, cmap='gray'), plt.axis('off')
# plt.subplot(144), plt.title("CntsPic"), plt.imshow(cv2.cvtColor(contourPic, cv2.COLOR_BGR2RGB), cmap='gray'), plt.axis('off')
# plt.tight_layout()
# plt.show()

# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
import numpy as np
import cv2


def np_list_int(tb):
    tb_2 = tb.tolist() #将np转换为列表
    return tb_2


def shot(img, dt_boxes):#应用于predict_det.py中,通过dt_boxes中获得的四个坐标点,裁剪出图像
    dt_boxes = np_list_int(dt_boxes)
    boxes_len = len(dt_boxes)
    num = 0
    while 1:
        if (num < boxes_len):
            box = dt_boxes[num]
            tl = box[0]
            tr = box[1]
            br = box[2]
            bl = box[3]
            print("打印转换成功数据num =" + str(num))
            print("tl:" + str(tl), "tr:" + str(tr), "br:" + str(br), "bl:" + str(bl))
            print(tr[1],bl[1], tl[0],br[0])


            crop = img[int(tr[1]):int(bl[1]), int(tl[0]):int(br[0]) ]

            
            # crop = img[27:45, 67:119] #测试
            # crop = img[380:395, 368:119]

            cv2.imwrite("K:/paddleOCR/PaddleOCR/screenshot/a/" + str(num) + ".jpg", crop)

            num = num + 1
        else:
            break


def shot1(img_path,tl, tr, br, bl,i):
    tl = np_list_int(tl)
    tr = np_list_int(tr)
    br = np_list_int(br)
    bl = np_list_int(bl)

    print("打印转换成功数据")
    print("tl:"+str(tl),"tr:" + str(tr), "br:" + str(br), "bl:"+ str(bl))

    img = cv2.imread(img_path)
    crop = img[tr[1]:bl[1], tl[0]:br[0]]

    # crop = img[27:45, 67:119]

    cv2.imwrite(str(i) + ".jpg", crop)

# tl1 = np.array([67,27])
# tl2= np.array([119,27])
# tl3 = np.array([119,45])
# tl4 = np.array([67,45])
# shot("K:\paddleOCR\PaddleOCR\screenshot\zong.jpg",tl1, tl2 ,tl3 , tl4 , 0)


