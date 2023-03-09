import os
import numpy as np
import matplotlib.pyplot as plt
 
root = "/home/kangshuai/Desktop/industry_detec/yolov5-v7.0/runs/detect/exp9/labels"
# root = "./demo_txt"
def findtxt(path, ret):
    """Finding the *.txt file in specify path"""
    filelist = os.listdir(path)
    for filename in filelist:
        de_path = os.path.join(path, filename)
        if os.path.isfile(de_path):
            if de_path.endswith(".txt"): #Specify to find the *.txt file
                ret.append(de_path)
        else:
            findtxt(de_path, ret)
 
def readtxt(path):
    """reading *.txt file in specify path"""
    try:
        txtarr = np.loadtxt(path)
    except Exception:
        txtarr = np.loadtxt(path, delimiter=',')
    return txtarr
 
def category(val, catarr):
    if val < 0.2:
        catarr[0] += 1
    if 0.2 <= val <= 0.4:
        catarr[1] += 1
    if 0.4 <= val <= 0.6:
        catarr[2] += 1
    if 0.6 <= val <= 0.8:
        catarr[3] += 1
    if 0.8 <= val <= 1.0:
        catarr[4] += 1
    # if 251 <= val <= 300:
    #     catarr[5] += 1
    # if 301 <= val <= 350:
    #     catarr[6] += 1
    # if val > 350:
    #     catarr[7]
    return catarr
 
def autolabel(ax, rects, xpos='center'):
    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off
 
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                '{}'.format(height), ha=ha[xpos], va='bottom')
 
def drawbar(arr1, xlab):
    ind = np.arange(len(arr1))  # the x locations for the groups
    width = 0.35  # the width of the bars
 
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width / 2, arr1, width, color='SkyBlue', label='conf')
    # rects2 = ax.bar(ind + width / 2, arr2, width, color='IndianRed', label='high')
 
    ax.set_ylabel('Value')
    ax.set_title('distribution')
    ax.set_xticks(ind)
    ax.set_xticklabels(xlab)
    ax.legend()
 
    autolabel(ax, rects1)  # autolabel(rects1, "left")
    # autolabel(ax, rects2)  # autolabel(rects2, "right")
 
    plt.savefig('qp', bbox_inches='tight')
    plt.show()
 
# zw ld hd hs kd fb qp
 
txt_pathlist = []
xlab = ["<0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
widthval = np.zeros(len(xlab))
findtxt(root, txt_pathlist)
highval = np.zeros(len(xlab))
for path in txt_pathlist:
    # print(len(txt_pathlist))
    if os.path.getsize(path):
        with open(path,'r') as f:
            txtarrs =  f.readlines()
            for txtarr in txtarrs:
                txtarr0 = txtarr.strip().split(" ")
                print(txtarr0)
                if txtarr0[0] == '6':
                    widthval = category(float(txtarr0[5]), widthval)
        # txtarr = readtxt(path)
        # for lineval in txtarr:
        #     # print(lineval[5])
        #     widthval = category(lineval[5], widthval)
        #     # highval = category(lineval[3], highval)
 
drawbar(widthval,xlab)