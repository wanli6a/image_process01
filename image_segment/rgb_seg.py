#-*- coding = utf-8 -*-
#@Time : 2021-12-15 15:12
#@Author : Wanli
#@File : rgb_seg.py
#@Software:PyCharm


import cv2
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# 读入图像

rgb_img = cv2.imread(r"E://YOLO//corntest//valid//images//445.jpg", cv2.IMREAD_UNCHANGED)

# hsv_imge = cv2.cvtColor(rgb_img,cv2.COLOR_RGB2HSV)

r,g,b = cv2.split(rgb_img)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

pixel_colors = rgb_img.reshape((np.shape(rgb_img)[0]*np.shape(rgb_img)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()


axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")


plt.show()
