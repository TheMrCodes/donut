import requests
import base64
import os
import json
import utils.storage as storage
from PIL import Image

file = './dataset/inference/IMG_20221010_152054_crop.jpg'
file_extension = os.path.splitext(file)[1][1:]
with open(file, 'rb') as f:
  image_bytes = f.read()

container = storage.get_storage_container('expency-input')

image_base64 = base64.b16encode(image_bytes).decode('utf-8')
image_base64 = f'data:image/{file_extension};base64,{image_base64}'

print("send: " + json.dumps({ "data": [ image_base64 ] }))
response = requests.post("http://localhost:8000/gradio/run/predict", json={ "data": [ image_base64 ] }).json()

data = response["data"]
print(data)