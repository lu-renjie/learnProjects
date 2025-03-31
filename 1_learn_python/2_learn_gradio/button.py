import gradio as gr

def f1():
    return 'f1'


def f2():
    return 'f2'


with gr.Blocks() as demo:
    name = gr.Textbox(label="Name")
    
    btn = gr.Button("f1")
    output1 = gr.Textbox(label="Output Box")
    btn.click(
        fn=f1,
        inputs=name,
        outputs=output1)

demo.launch()