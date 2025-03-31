import cv2 as cv
import gradio as gr


def blur(img, kernel_size, sigma):
    kernel_size = int(kernel_size)
    kernel_size = (kernel_size, kernel_size)
    return cv.GaussianBlur(img, kernel_size, sigmaX=sigma)



demo = gr.Interface(
    blur,
    inputs=['image', gr.Slider(3, 9, step=2), gr.Slider(0.1, 10)],
    outputs='image')
demo.launch()
