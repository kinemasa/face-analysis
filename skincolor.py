import cv2
import dlib
import numpy as np
import glob
import sys
import csv
import re
import os
import funcSkinSeparation2 as ss
from imutils import face_utils

def getVideoROI(img):
    roi = cv2.selectROI(img)
    cv2.destroyAllWindows()
    return roi

#顔領域を抽出する
def DetectFaceFeatures(detector, img, face_predictor):
    rects = detector(img)
    for rect in rects:
        #顔のランドマーク検出
        shapes = face_predictor(img, rect)
        shapes = face_utils.shape_to_np(shapes)
    return shapes

"""
#顔領域の四角形を作成
def DetectFeaturesRect(point, mask_width, mask_height):
    roi0 = point[0]-(mask_width/3)
    roi1 = point[1]-(mask_height/2)
    roi2 = point[0]+(mask_width/3)
    roi3 = point[1]+(mask_height/2)
    roi = [int(roi0), int(roi1), int(roi2), int(roi3)]
    return roi
"""

#頬（右上）の四角形を作成
def Detectupperright(point1, point2, mask_width, mask_height):
    roi0 = point1[0]+(mask_width/7)
    roi1 = point1[1]
    roi2 = point2[0]+(mask_width/3)
    roi3 = point2[1]
    roi = [int(roi0), int(roi1), int(roi2), int(roi3)]
    return roi

#頬（左上）の四角形を作成
def Detectupperleft(point1, point2, mask_width, mask_height):
    roi0 = point1[0]-(mask_width/7)
    roi1 = point1[1]
    roi2 = point2[0]-(mask_width/3)
    roi3 = point2[1]
    roi = [int(roi0), int(roi1), int(roi2), int(roi3)]
    return roi

#頬（右下）の四角形を作成
def Detectbottomright(point1, point2, mask_width, mask_height):
    roi0 = point1[0]+(mask_width/8)
    roi1 = point1[1]
    roi2 = point2[0]+(mask_width/3.5)
    roi3 = point2[1]
    roi = [int(roi0), int(roi1), int(roi2), int(roi3)]
    return roi

#頬（左下）の四角形を作成
def Detectbottomleft(point1, point2, mask_width, mask_height):
    roi0 = point1[0]-(mask_width/8)
    roi1 = point1[1]
    roi2 = point2[0]-(mask_width/3.5)
    roi3 = point2[1]
    roi = [int(roi0), int(roi1), int(roi2), int(roi3)]
    return roi

#額の四角形を作成
def Detecthead(point1, point2, mask_width, mask_height):
    roi0 = point1[0]
    roi1 = point1[1]-(mask_height/10)
    roi2 = point2[0]
    roi3 = point2[1]-(mask_height/2)
    roi = [int(roi0), int(roi1), int(roi2), int(roi3)]
    return roi

"""
#出力
def output(R, G, B, dir_name, name):
    with open(dir_name + name +'R.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for val in R:
            writer.writerow([val])
    with open(dir_name + name +'G.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for val in G:
            writer.writerow([val])
    with open(dir_name + name +'B.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for val in B:
            writer.writerow([val])
"""

#出力
def output(R, G, B, dir_name, name):
    with open(dir_name + name +'.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow([R,G,B])
    
    
#dir_name = 'C:\\Users\\koike\\Desktop\\local_remote\\image_data\\210111\\002\\Cam 1\\'
#dir_name = 'C:\\Users\\elite\\Documents\\oflinetestdata\\'
#dir_name = 'F:\\test_img\\test\\'
dir_name = 'E:\\2023_5_1\\No-06-2023-0404\\'

#files = glob.glob(dir_name+'*')
#files = glob.glob(dir_name+'*.bmp')
files = glob.glob(dir_name+'colorChartImage_20230203_090000.png')

num = len(files)
# print(num)
img_name = files[0]
img = cv2.imread(img_name)
width = int(img.shape[1])
height = int(img.shape[0])

#roi = getVideoROI(img)


pulsewave = np.zeros(int(num))
time = np.zeros(int(num))
Rupperright = np.zeros(int(num))
Gupperright = np.zeros(int(num))
Bupperright = np.zeros(int(num))
Rbottomright = np.zeros(int(num))
Gbottomright = np.zeros(int(num))
Bbottomright = np.zeros(int(num))
Rupperleft = np.zeros(int(num))
Gupperleft = np.zeros(int(num))
Bupperleft = np.zeros(int(num))
Rbottomleft = np.zeros(int(num))
Gbottomleft = np.zeros(int(num))
Bbottomleft = np.zeros(int(num))
Rhead = np.zeros(int(num))
Ghead = np.zeros(int(num))
Bhead = np.zeros(int(num))

i = 0
for f in files:

    basename=os.path.basename(f)
    filename = os.path.splitext(basename)[0]  # 拡張子の除去
    root, ext = os.path.splitext(basename)
    f_sp = re.split('[-_ ]',root)
    
    

    #print(f_sp)

    #time[i] = float(f_sp[3])*3600000+float(f_sp[4])*60000+float(f_sp[5])*1000
    
    
    #画像の読み込みとグレースケール化
    img = cv2.imread(f)
    img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #顔検出ツールの呼び出し
    detector = dlib.get_frontal_face_detector()
    
    #顔のランドマーク検出のツール呼び出し
    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    face_predictor = dlib.shape_predictor(predictor_path)
    
    #ランドマーク検出
    shapes = DetectFaceFeatures(detector, img_gry, face_predictor)
    
    #鼻の点と縦横の長さを計算
    point20 = shapes[19,:]
    point25 = shapes[24,:]
    point30 = shapes[29,:]
    point32 = shapes[31,:]
    point36 = shapes[35,:]
    point50 = shapes[49,:]
    point54 = shapes[53,:]
    #print(point)
    mask_width = shapes[14,0]-shapes[2,0]
    mask_height = shapes[51,1]-shapes[29,1]
    #roi = DetectFeaturesRect(point, mask_width, mask_height) 
    roi_upperright = Detectupperright(point30, point36, mask_width, mask_height)
    roi_upperleft = Detectupperleft(point30, point32, mask_width, mask_height)
    roi_bottomright = Detectbottomright(point36, point54, mask_width, mask_height)
    roi_bottomleft = Detectbottomleft(point32, point50, mask_width, mask_height)
    roi_head = Detecthead(point20, point25, mask_width, mask_height)

    
    #バウンディングボックスの描画
    cv2.rectangle(img,
                  pt1=(roi_head[0],roi_head[1]),
                  pt2=(roi_head[2],roi_head[3]),
                  color=(0,255,0),
                  thickness=1,
                  lineType=cv2.LINE_4,
                  shift=0)
    cv2.rectangle(img,
                  pt1=(roi_upperleft[0],roi_upperleft[1]),
                  pt2=(roi_upperleft[2],roi_upperleft[3]),
                  color=(0,255,0),
                  thickness=1,
                  lineType=cv2.LINE_4,
                  shift=0)
    cv2.rectangle(img,
                  pt1=(roi_upperright[0],roi_upperright[1]),
                  pt2=(roi_upperright[2],roi_upperright[3]),
                  color=(0,255,0),
                  thickness=1,
                  lineType=cv2.LINE_4,
                  shift=0)
    cv2.rectangle(img,
                  pt1=(roi_bottomleft[0],roi_bottomleft[1]),
                  pt2=(roi_bottomleft[2],roi_bottomleft[3]),
                  color=(0,255,0),
                  thickness=1,
                  lineType=cv2.LINE_4,
                  shift=0)
    cv2.rectangle(img,
                  pt1=(roi_bottomright[0],roi_bottomright[1]),
                  pt2=(roi_bottomright[2],roi_bottomright[3]),
                  color=(0,255,0),
                  thickness=1,
                  lineType=cv2.LINE_4,
                  shift=0)
    #画像表示
    cv2.imshow('img', img)
    cv2.waitKey(0)
    #cv2.destroyAllWindows() 
    
    
    """
    #ランドマークの可視化
    for (j, (x,y)) in enumerate(landmark):
        cv2.circle(img, (x,y), 1, (255, 0, 0), -1)
    
    cv2.imshow('sample', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    
    
    #pulsewave[i] = np.mean(ss.skinSeparation(img[roi[1]:roi[3], roi[0]:roi[2],:]))
    
    
    Rupperright[i] = np.mean(img[roi_upperright[1]:roi_upperright[3], roi_upperright[0]:roi_upperright[2], 2])
    Gupperright[i] = np.mean(img[roi_upperright[1]:roi_upperright[3], roi_upperright[0]:roi_upperright[2], 1])
    Bupperright[i] = np.mean(img[roi_upperright[1]:roi_upperright[3], roi_upperright[0]:roi_upperright[2], 0])
    
    Rbottomright[i] = np.mean(img[roi_bottomright[1]:roi_bottomright[3], roi_bottomright[0]:roi_bottomright[2], 2])
    Gbottomright[i] = np.mean(img[roi_bottomright[1]:roi_bottomright[3], roi_bottomright[0]:roi_bottomright[2], 1])
    Bbottomright[i] = np.mean(img[roi_bottomright[1]:roi_bottomright[3], roi_bottomright[0]:roi_bottomright[2], 0])
    
    Rupperleft[i] = np.mean(img[roi_upperleft[1]:roi_upperleft[3], roi_upperleft[2]:roi_upperleft[0], 2])
    Gupperleft[i] = np.mean(img[roi_upperleft[1]:roi_upperleft[3], roi_upperleft[2]:roi_upperleft[0], 1])
    Bupperleft[i] = np.mean(img[roi_upperleft[1]:roi_upperleft[3], roi_upperleft[2]:roi_upperleft[0], 0])
    
    
    Rbottomleft[i] = np.mean(img[roi_bottomleft[1]:roi_bottomleft[3], roi_bottomleft[2]:roi_bottomleft[0], 2])
    Gbottomleft[i] = np.mean(img[roi_bottomleft[1]:roi_bottomleft[3], roi_bottomleft[2]:roi_bottomleft[0], 1])
    Bbottomleft[i] = np.mean(img[roi_bottomleft[1]:roi_bottomleft[3], roi_bottomleft[2]:roi_bottomleft[0], 0])
  
    Rhead[i] = np.mean(img[roi_head[3]:roi_head[1], roi_head[0]:roi_head[2], 2])
    Ghead[i] = np.mean(img[roi_head[3]:roi_head[1], roi_head[0]:roi_head[2], 1])
    Bhead[i] = np.mean(img[roi_head[3]:roi_head[1], roi_head[0]:roi_head[2], 0])
    
    
    
    
    i += 1

    sys.stdout.flush()
    sys.stdout.write('\rProcessing... (%d/%d)' %(i,num))

i = 0


cv2.destroyAllWindows() 


name = "upperright"
output(Rupperright[0],Gupperright[0],Bupperright[0], dir_name, name)
name = "bottomright"
output(Rbottomright[0],Gbottomright[0],Bbottomright[0], dir_name, name)
name = "upperleft"
output(Rupperleft[0],Gupperleft[0],Bupperleft[0], dir_name, name)
name = "bottomleft"
output(Rbottomleft[0],Gbottomleft[0],Bbottomleft[0], dir_name, name)
name = "head"
output(Rhead,Ghead,Bhead, dir_name, name)



