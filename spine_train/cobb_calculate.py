import scipy
import cv2 as cv
import numpy as np

image = cv.imread(r"F:\MyCode\mmsegmentation\spine_train\results\sunhl-1th-26-Jul-2016-70 B AP.png",
                  cv.IMREAD_GRAYSCALE)
contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
area_list = []  # 存放相应面积大小的轮廓

# 获取图像中相应面积大小的轮廓
for area in range(len(contours)):
    area_dst = cv.contourArea(contours[area].squeeze(1))
    # 面积大于100且小于5000将会被保存
    if 100 < area_dst < 5000:
        area_contours = contours[area].squeeze(1)
        area_list.append(area_contours)

"""最小外接矩形"""
for contour in area_list:
    rect = cv.minAreaRect(contour)
    box = cv.boxPoints(rect)
    box_point = np.int0(box)
    cv.drawContours(image, [box_point], 0, (0, 255, 255), 3)

cv.imshow("img", image)
cv.waitKey(0)
cv.destroyAllWindows()


