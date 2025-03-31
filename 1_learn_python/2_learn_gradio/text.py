import gradio as gr

def greet(name, is_morning, temperature):
    salutation = 'Good morning' if is_morning else 'Good evening'
    greeting = f'{salutation} {name}. It is {temperature} degrees today'
    celsius = (temperature - 32) * 5 / 9
    return greeting, round(celsius, 2)


demo = gr.Interface(
    fn=greet,
    inputs=['text', 'checkbox', gr.Slider(0, 100)],
    outputs=['text', 'number'])
# 'checkbox'是勾选框, 会显示参数is_morning
# 'slider'就是滑动条 
demo.launch(server_port=6006)
