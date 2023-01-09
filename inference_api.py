from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
import uvicorn
import gradio as gr
import os
import time
import torch
import utils.storage as storage
import utils.environment as env
from PIL import Image
from donut import DonutModel
import numpy as np
import io as io


model_name = 'naver-clova-ix/donut-base-finetuned-cord-v2'

origins = [
    "http://localhost.apiglmann.at",
    "https://localhost.apiglmann.at",
    "https://expency.apiglmann.at",
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
domain = env.str('API_DOMAIN')
if domain: origins.append(domain)

app = FastAPI(
    root_path=env.str('API_ROOT_RESOURCE_PATH', '')
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_main():
    return RedirectResponse(url='/gradio')
@app.get("/health_check")
def read_main():
    return { "status": "Active" }


pretrained_model = DonutModel.from_pretrained(model_name)
if torch.cuda.is_available():
    pretrained_model.half()
    device = torch.device("cuda")
    pretrained_model.to(device)
else:
    pretrained_model.encoder.to(torch.bfloat16)
pretrained_model.eval()

container = storage.get_storage_container('expency-input')


def inference_cord(input_img, task_name: str = 'cord-v2', question: str = None):
    task_prompt = f"<s_{task_name}>"

    start_time = time.perf_counter()
    prompt = task_prompt.replace("{user_input}", question) \
        if task_name == "docvqa" else task_prompt
    output = pretrained_model.inference(image=input_img, prompt=prompt)["predictions"][0]
    end_time = time.perf_counter()
    print(f"Elapsed time: {end_time - start_time:.6f} seconds")

    return {"output": output}


def predict(input_img):
    return inference_cord(input_img)

async def preduct_API(text):
    obj = container.get_object(text)
    np_image = np.frombuffer(b''.join(obj.as_stream(storage.buffer_size)), dtype=np.uint8)
    input_img = Image.open(io.BytesIO(np_image))
    return inference_cord(input_img)

interface1 = gr.Interface(
    fn=predict,
    inputs= gr.Image(type="pil"),
    outputs="json",
    title=f"Donut 🍩 demonstration for `cord-v2` task",
    description="""This model is trained with 800 Indonesian receipt images of CORD dataset. <br>
Demonstrations for other types of documents/tasks are available at https://github.com/clovaai/donut <br>
More CORD receipt images are available at https://huggingface.co/datasets/naver-clova-ix/cord-v2
More details are available at:
- Paper: https://arxiv.org/abs/2111.15664
- GitHub: https://github.com/clovaai/donut""",
    examples=[["./dataset/inference/ds1.png"], ["./dataset/inference/IMG_20221010_152054_crop.jpg"]],
    cache_examples=False,
)

interface2 = gr.Interface(
    fn=preduct_API,
    inputs=gr.Textbox(label="filename", lines=1, value='IMG_20221010_152054_crop.jpg'),
    outputs="json",
    title=f"Donut 🍩 demonstration for `cord-v2` task",
    description="""This model is trained with 800 Indonesian receipt images of CORD dataset. <br>
Demonstrations for other types of documents/tasks are available at https://github.com/clovaai/donut <br>
More CORD receipt images are available at https://huggingface.co/datasets/naver-clova-ix/cord-v2
More details are available at:
- Paper: https://arxiv.org/abs/2111.15664
- GitHub: https://github.com/clovaai/donut""",
    examples=[[it.name] for it in container.list_objects()],
    cache_examples=False,
)


app = gr.mount_gradio_app(app, interface1, path="/gradio")
app = gr.mount_gradio_app(app, interface2, path="/objstorage")

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
