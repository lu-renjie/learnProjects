import io
import cv2 as cv
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt



img = cv.imread('ke_qing.png')

# changing colorspace
img_BGR = img
img_GRAY = cv.cvtColor(img_BGR, cv.COLOR_BGR2GRAY)
img_HSV = cv.cvtColor(img_BGR, cv.COLOR_BGR2HSV)

# HSV
# H: 色相（颜色）,0 ~ 180
# S: 饱和度（鲜艳程度，就是颜色的密集程度，低饱和度就是偏灰色）
# V: 亮度（太亮了就是偏白色，太暗了就是偏黑色）
# HSV图像更容易处理颜色

# print(img_GRAY.shape)  # (250, 250)

imgs = np.concatenate([img_BGR, img_GRAY[:, :, None].repeat(3, 2), img_HSV], axis=1)
cv.imshow('color', imgs)

# histogram
# 不知道为啥第一个参数是list，即使list有多个图片，计算的也只是第一个图像
hist = cv.calcHist([img], [0], mask=None, histSize=[256], ranges=[0, 256])
print('hist size', hist.shape)  # (256, 1)
hist = hist.squeeze()
plt.bar(range(len(hist)), hist)
img_buffer = io.BytesIO()
plt.savefig(img_buffer, format='png')  # matplotlib依赖于dpi保存，与显示器有关
hist_img = Image.open(img_buffer).convert('L')
hist_img = np.array(hist_img, copy=False, dtype=np.float32) / 255
# 2D histogram
hist_2d = cv.calcHist([img_HSV], [0, 1], None, [180, 256], [0, 180, 0, 256])
hist_2d = np.log(hist_2d)
hist_2d = hist_2d / hist_2d.max()
# print('2D hist', hist_2d.shape)  # (180, 256)
hist_img = cv.resize(hist_img, (256, 180))
imgs = np.concatenate([hist_img, hist_2d], axis=1)
cv.imshow('hist', imgs)

# Gemoetric Transformations
# resize
img_resize_bilinear = cv.resize(img, None, fx=2, fy=2)
img_resize_cubic = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
# cubic插值更平滑一些，看起来不那么粗糙
imgs = np.concatenate([img_resize_bilinear, img_resize_cubic], axis=1)
cv.imshow('resize', imgs)
# affine transformation
# 变换的矩阵, 最后一列是平移的向量
M = np.array(
    [
        [1, 0, 100],
        [0, 1, 50]
    ], dtype=np.float32
)
W, H, C = img.shape
img_translate = cv.warpAffine(img, M, dsize=(W, H))
# rotation
M = cv.getRotationMatrix2D(center=(100, 100), angle=60, scale=1)
img_rotate = cv.warpAffine(img, M, dsize=(W, H))
# 根据3个点的位置变化求变换矩阵
points1 = np.float32([[50, 50], [200, 50], [50, 200]])
points2 = np.float32([[10, 100], [200, 50], [100, 250]])
M = cv.getAffineTransform(points1, points2)
img_affine = cv.warpAffine(img, M, dsize=(W, H))
imgs = np.concatenate([img_translate, img_rotate, img_affine], axis=1)
cv.imshow('affine', imgs)
# perspective transformation
# 需要4个点



# thresholding
threshold, result1 = cv.threshold(img_GRAY, thresh=127, maxval=255, type=cv.THRESH_BINARY)  # 小于thresh的变为0，大于的变为1
threshold, result2 = cv.threshold(img_GRAY, thresh=127, maxval=255, type=cv.THRESH_TOZERO)  # 小于thresh的变为0
threshold, result3 = cv.threshold(img_GRAY, thresh=127, maxval=255, type=cv.THRESH_TRUNC)  # 大于thresh的变为thresh
print('threshold', threshold)
# type可以是cv.THRESH_BINARY+cv.THRESH_TRUNC，表示先二值化，然后截断
# adaptive threshold
result4 = cv.adaptiveThreshold(img_GRAY, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize=11, C=2)
# 这个方法将高斯模糊的结果减去C作为阈值，其实就是提取边缘
threshold, result5 = cv.threshold(img_GRAY, thresh=None, maxval=255, type=cv.THRESH_OTSU)
imgs = np.concatenate([result1, result2, result3, result4, result5], axis=1)
cv.imshow('thresh', imgs)



# blur
img_blur = cv.blur(img, (11, 11))
img_guassion_blur = cv.GaussianBlur(img, (11, 11), sigmaX=1, sigmaY=1)
img_median_blur = cv.medianBlur(img, 5)
# cv.bilateralFilter(img, 9, 75, 75)  # 双边滤波，可以保留边缘不模糊，没学过以后再看吧
imgs = np.concatenate([img, img_blur, img_guassion_blur, img_median_blur], axis=1)
# cv.imshow('blur', imgs)



# gradient, cv.CV_64F表示输出图像的数据类型，如果用uint8会截断的
laplacian = cv.Laplacian(img_GRAY, cv.CV_64F, ksize=5)
sobelx = cv.Sobel(img_GRAY, cv.CV_64F, 1, 0, ksize=5)
sobely = cv.Sobel(img_GRAY, cv.CV_64F, 0, 1, ksize=5)
imgs = np.concatenate([laplacian, sobelx, sobely], axis=1)
cv.imshow('gradient', imgs)



# image pyramids
G0 = cv.resize(img, (256, 256))
G1 = cv.pyrDown(G0)
G2 = cv.pyrDown(G1)

G1_ = cv.pyrUp(G2)
G0_ = cv.pyrUp(G1_)  # G0_相比G0损失了细节，保留了“结构信息”

L2 = G2
L1 = cv.subtract(G1, G1_)
L0 = cv.subtract(G0, G0_)

imgs = np.concatenate([G0, G0_, L0], axis=1)
# imgs = np.concatenate([G1, G1_, L1], axis=1)
cv.imshow('pyramid1', imgs)



# besides above process
# morphological transformation can be used to handle binary image



cv.waitKey(0)
