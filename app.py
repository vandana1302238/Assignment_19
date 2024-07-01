import gradio as gr
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import BigramLanguageModel, ModelConfig

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# unique characters from text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from chars to ints
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# load model
model_spec = torch.load("ckpt.pt", map_location=torch.device('cpu'))
model_args = model_spec['model_args']
model_weights = model_spec['model']
modelconf = ModelConfig(**model_args)
trained_model = BigramLanguageModel(modelconf)
trained_model.load_state_dict(model_weights)

def generate_text(seed_text, max_new_tokens, temperature):
    text = seed_text if seed_text is not None else " "
    text = text if text.endswith(" ") else seed_text + " "
    context = torch.tensor(encode(text), dtype=torch.long).unsqueeze(0)
    temperature = temperature if temperature > 0 else 1e-5
    return decode(trained_model.generate(context, temperature = temperature, max_new_tokens=max_new_tokens)[0].tolist())

with gr.Blocks() as demo:
    gr.HTML("<h1 align = 'center'> Generate Text Based on simple GPT model <br> (Dataset = William Shakespeare) </h1>")
    
    content = gr.Textbox(label = "Enter initial text to generate content")
    with gr.Row(equal_height=True):
      with gr.Column(): 
        max_tokens = gr.Number(label = "Maximum tokens to generate content", value = 100)
        temp_val = gr.Slider(label = "Temparature (slide to higher value for higher creativity)", minimum = 0.0, maximum= 1.0,value = 0.7)
        generate_btn = gr.Button(value = 'Generate Text')
      with gr.Column(): 
        outputs  = [gr.TextArea(label = "Generated text (William Shakespeare)", lines = 7)]
    inputs = [
                content,
                max_tokens, 
                temp_val
              ]
    generate_btn.click(fn = generate_text, inputs= inputs, outputs = outputs)

if __name__ == '__main__':
    demo.launch() 
