import time
import numpy as np
import gradio as gr


def random(steps):
    for _ in range(steps):
        time.sleep(1)
        image = np.random.rand(400, 400, 3)
        yield image

demo = gr.Interface(
    random,
    inputs=gr.Slider(1, 10, 3, step=1),
    outputs='image'
)
demo.queue()
demo.launch()
