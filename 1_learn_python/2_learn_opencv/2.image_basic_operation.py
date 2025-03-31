import cv2 as cv
import numpy as np

img = cv.imread('ke_qing.png')
cv.imshow('origin', img)
print(img.dtype)  # uint8

# split channel
b, g, r = cv.split(img)
img = cv.merge((b, g, r))

# padding, this is not an inplace operation
img_pad = cv.copyMakeBorder(img, 50, 50, 50, 50, cv.BORDER_CONSTANT, value=0)
cv.imshow('pad', img_pad)

# blend, note that dtype is uint8, so 250 + 10 = 260 % 256 = 4
# to avoid that, use cv.add() instead, it will clip 260 to 255
img1 = img
img2 = cv.imread('nuo_ai_er.jpeg')
img3 = cv.add(img1, img2)
cv.imshow('blend', img3)
img3 = cv.addWeighted(img1, 0.5, img2, 0.5, 0)  # alpha * img1 + beta * img2 + gamma
cv.imshow('blend', img3)

# bitwise operation
mask = np.zeros_like(img)
mask[100:200, 100:200] = 1
img_mask = img * mask
cv.imshow('mask', img_mask)

# resize
img = cv.resize(img, (512, 512))
cv.imshow('resize', img)
cv.waitKey(0)

print('optimization:', cv.useOptimized())  # true
# openCV的运算有优化，通常比较快
if not cv.useOptimized():
    cv.setUseOptimized(True)
