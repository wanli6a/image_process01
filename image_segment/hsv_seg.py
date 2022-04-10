#-*- coding = utf-8 -*-
#@Time : 2021-12-01 11:01
#@Author : Wanli
#@File : zx.py
#@Software:PyCharm

import cv2
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# 读入图像

rgb_img = cv2.imread(r"E://YOLO//corntest//valid//images//41.jpg", cv2.IMREAD_UNCHANGED)
hsv_img = cv2.cvtColor(rgb_img,cv2.COLOR_RGB2HSV)

h,s,v = cv2.split(hsv_img)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

pixel_colors = hsv_img.reshape((np.shape(hsv_img)[0]*np.shape(hsv_img)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()


axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")

plt.show()
