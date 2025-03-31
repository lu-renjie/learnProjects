import cv2 as cv

img = cv.imread('ke_qing.png')  # BGR

if img is None:
    exit()

print(type(img))  # ndarray
print(img.shape)

cv.imshow('ke qing', img)
cv.waitKey(1000)  # display 1000ms
# cv.waitKey(0)  # 0 means wait forever, press ESC to exit

# drawing functions in OpenCV
# draw a line, note that this is an inplace operation
# (0, 0) is the upper-left corner, 第一个坐标是横向，第二个是纵向
# (w, h) is the bottom-right corner
img_line = cv.line(img.copy(), (0, 0), (125, 125), color=(255, 0, 0))
cv.imshow('', img_line)
cv.waitKey(1000)  # display 1000ms
# draw a rectangle
img_rectangle = cv.rectangle(img.copy(), (50, 50), (200, 200), color=(255, 0, 0))
cv.imshow('', img_rectangle)
cv.waitKey(1000)
# draw a circle
img_circle = cv.circle(img.copy(), (125, 125), 50, color=(255, 0, 0))
cv.imshow('', img_circle)
cv.waitKey(1000)
# add text to an image
text = 'python'
font = cv.FONT_HERSHEY_SIMPLEX  # font face是字体的意思
(w, h), baseline = cv.getTextSize(text, font, 1, 2)
# (w, h)是文本框的大小
# 考虑用来写英语的四线本
# 四条线里第一条和第三条之间的距离是h
# 最下面两条线之间的距离就是baseline，也就是y, p这类字母溢出的部分
print(w, h, baseline)
img_text = cv.putText(img, text, (0, img.shape[1] - baseline), font, 1, (255, 0, 0), 2)
cv.imshow('', img_text)
cv.waitKey(0)


# set mouse as a paint brush


# use trackbar