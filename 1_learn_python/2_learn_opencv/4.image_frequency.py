import cv2 as cv
import numpy as np


def normalize(img):
    """
    normalize to [0, 1]
    """
    max_ = img.max()
    min_ = img.min()
    return (img - min_) / (max_ - min_)


def magnitude(img):
    """
    convert complex matrix to real matrix
    """
    assert img.ndim == 3
    return cv.magnitude(img[:, :, 0], img[:, :, 1])


# img = cv.imread('ke_qing.png')
# img = cv.imread('women.png')
img = cv.imread('nuo_ai_er.jpeg')
img = cv.imread('lena.tif')
assert img is not None


img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = img / 255  # to float

h, w = img.shape
mask_guassian_w = cv.getGaussianKernel(w, sigma=10)  # (w, 1)
mask_guassian_h = cv.getGaussianKernel(h, sigma=10)  # (h, 1)
mask = mask_guassian_h * mask_guassian_w.T  # (h, w)
mask = mask[:, :, None]  # (h, w, 1)
mask = mask / mask.max()
mask = mask > 0.1
# 计算频域
img_dft = cv.dft(img, flags=cv.DFT_COMPLEX_OUTPUT)  # (h, w, 2)
img_dft = np.fft.fftshift(img_dft, axes=[0, 1])
# 傅里叶谱可视化
img_dft_amplitude = magnitude(img_dft)
img_dft_amplitude_visualize = np.log(img_dft_amplitude + 1e-5)
img_dft_amplitude_visualize = normalize(img_dft_amplitude_visualize)
# 频域的相角
img_dft_phase = cv.phase(img_dft[:, :, 0], img_dft[:, :, 1])
img_dft_phase_visualize = np.abs(img_dft_phase)
img_dft_phase_visualize = normalize(img_dft_phase_visualize)

# 低通滤波
img_dft_low = np.fft.ifftshift(img_dft * mask, axes=[0, 1])
img_low = cv.idft(img_dft_low)
img_low = magnitude(img_low)
img_low = normalize(img_low)

# 高通滤波
img_dft_high = np.fft.ifftshift(img_dft * (1 - mask), axes=[0, 1])
img_high = cv.idft(img_dft_high, flags=cv.DFT_COMPLEX_INPUT)
img_high = magnitude(img_high)
img_high = normalize(img_high)

# 傅里叶谱对应的图像, 把相角全部归1
img_dft_amplitude_ = np.fft.ifftshift(img_dft_amplitude)
img_amplitude = cv.idft(img_dft_amplitude_)
img_amplitude = normalize(img_amplitude)

# 相角对应的图像, 把幅度全部归一化，相角描述了图像的形状信息
img_dft_phase_ = np.fft.ifftshift(img_dft / img_dft_amplitude[:, :, None], axes=[0, 1])
img_phase = cv.idft(img_dft_phase_)
img_phase = magnitude(img_phase)
img_phase = normalize(img_phase)

empty = np.ones_like(img, dtype=np.float32)
imgs_1 = np.concatenate([img, img_low, img_high], axis=1)
imgs_2 = np.concatenate([img_dft_amplitude_visualize, img_dft_phase_visualize, img_phase], axis=1)
imgs = np.concatenate([imgs_1, imgs_2], axis=0)
cv.imshow('', imgs)
cv.waitKey(0)

