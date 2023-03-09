
import os
import cv2
from PIL import Image
import numpy as np
crop_height = 512 ###***********滑窗裁剪的图片高度***************
crop_width = 512 ###***********滑窗裁剪的图片宽度***********
stride = 256 ###***********滑窗裁剪的步长***********
overlap = int(crop_width - stride) ###前一张裁剪图和当前裁剪图的重叠区域
img_predict_dir = './new/2/' ###***********要进行测试的图片所在的文件夹***********
img_result_dir = './train1/2/' ###***********测试结果保存位置***********
if not os.path.exists(img_result_dir):  # 文件夹不存在，则创建
    os.mkdir(img_result_dir)
num = 1


'''
实现滑窗切分图片
'''


def fuse(output_image, row, column, overlap, counter, temp1, temp2, res_row, res_column):# output_image拼接输入图片，row高度需要融合的数量，column宽度度需要融合的数量，overlap 重复检测的区域100 由于切割图片是500，步长为100
    # output = 'process area: ' + '%04d'%(counter)                                         counter当前融合的第*张，temp1已拼接好的**，temp2已拼接好的**，res_row剩余的高度，res_column剩余的宽度
    # print(output)

    if counter % column == 0:  #如果当前融合的图片是这张图片最右边际
        output_image = output_image[:, -(res_column + overlap):]
    if counter > (row - 1) * column:
        output_image = output_image[-(res_row + overlap):, :]

    if counter % column == 1: ###一行行进行滑窗
        temp1 = output_image
    else:
        temp1_1 = temp1[:, 0:-overlap, :]  #前一张融合图片的未重叠的部分，0：-100 第0列到倒数第100列->0:400
        temp1_2 = temp1[:, -overlap:, :]    #前一张融合图片与当前融合图片重叠检测部分
        temp1_3 = output_image[:, 0:overlap, :] #当前融合图片与前一张融合图片重叠检测部分
        temp1_4 = output_image[:, overlap:, :] #当前融合图片与前一张图片为重叠部分
        temp1_fuse = 0.5 * temp1_2 + 0.5 * temp1_3  #将当前融合图片与前一张融合图片互相重叠的部分加权取平均
        temp1 = np.concatenate((temp1_1, temp1_fuse, temp1_4), axis=1)  #融合结束
        if int(counter % column) == 0: #判断是第几行
            if counter == column: ##判断是否进入下一行 刚好整除行的column 说明一行滑窗结束，准备进入下一行
                temp2 = temp1 ##将一行滑窗后的temp1保存到temp2中，进行一行滑窗时候temp2是不变的，只在进行换行滑窗的时候变化
            else:
                temp2_1 = temp2[0:-overlap, :, :]
                temp2_2 = temp2[-overlap:, :, :]
                temp2_3 = temp1[0:overlap, :, :]
                temp2_4 = temp1[overlap:, :, :]
                temp2_fuse = 0.5 * temp2_2 + 0.5 * temp2_3
                temp2 = np.concatenate((temp2_1, temp2_fuse, temp2_4), axis=0)

    return temp1, temp2



for filename in os.listdir(img_predict_dir):
    img_name = img_predict_dir + filename  ###图片名字

    loaded_image = cv2.imread(img_name)  ##cv2形式读取图片格式为BGR 后面要进行image格式转换，否则测试叠掩的三通道合成图会出问题

    img_h = loaded_image.shape[0]  ##cv2模式读取图片的高用img.shape[0] image.open()读取图片的高度显示用Img.height,宽度用img.width
    img_w = loaded_image.shape[1]  ##cv2模式读取图片的宽用img.shape[1]
    row = int((img_h - crop_height) / stride + 1)  ##高可以切为row块
    column = int((img_w - crop_width) / stride + 1)  ##宽可以切为cloumn块
    res_row = img_h - (crop_height + stride * (row - 1))  ##高度滑窗切分完row块后还剩余的部分
    res_column = img_w - (crop_width + stride * (column - 1))  ##宽度度滑窗切分完row块后还剩余的部分
    if (img_h - crop_height) % stride > 0:  ##判断剩余高度是否还可以继续进行滑窗切分
        row = row + 1
    if (img_w - crop_width) % stride > 0:  ##判断剩余宽度是否还可以继续进行滑窗切分
        column = column + 1
    counter = 1  ##起始从第一块开始


    for i in range(row):
        for j in range(column):
            if i == row - 1:  ##判断高度是否切到最后一块
                H_start = img_h - crop_height  ##如果是，最后一块的起始高度就是图片的高度往前数小图片要切分的高度crop_height
                H_end = img_h  ##结束高度就是图片的高的数字
            else:
                H_start = i * stride  ##如果不是切到最后一块，切分小图的起始高点就是第i块（第i次切分）*步长
                H_end = H_start + crop_height  ##结束高点就是 在开始的基础上直接加上crop_height
            if j == column - 1:
                W_start = img_w - crop_width
                W_end = img_w
            else:
                W_start = j * stride
                W_end = W_start + crop_width

            img_chip = loaded_image[H_start:H_end, W_start:W_end]  ##切块的小图片 nudarray格式
            image = Image.fromarray(cv2.cvtColor(img_chip, cv2.COLOR_BGR2RGB))
            # s = '%04d' % num  # 04表示0001,0002等命名排序
            # image.save(img_result_dir+str(s) + '.png')#**********注意图片格式********************#
            image.save(img_result_dir+ filename[:-4] + '_{}_{}.jpg'.format(str(i), str(j)))#**********注意图片格式********************#
            # num = num+1


