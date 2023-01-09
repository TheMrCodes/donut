
import argparse
import os
import time
import torch
from PIL import Image

from datasets import load_dataset
from donut import DonutModel, DonutDataset, load_json, save_json


"""
python inference.py --pretrained_model_name_or_path naver-clova-ix/donut-base-finetuned-cord-v2 --dataset_name_or_path "naver-clova-ix/cord-v2" --task_name cord --save_path "./result/output.json"
"""



def inference(args):
    if args.task_name == "docvqa":
        task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
    else:  # rvlcdip, cord, ...
        task_prompt = f"<s_{args.task_name}>"

    pretrained_model = DonutModel.from_pretrained(args.pretrained_model_name_or_path)
    if torch.cuda.is_available():
        pretrained_model.half()
        device = torch.device("cuda")
        pretrained_model.to(device)
    else:
        pretrained_model.encoder.to(torch.bfloat16)
    pretrained_model.eval()

    # Read Image as PIL Image
    # dataset = load_dataset(args.dataset_name_or_path, split="validation")
    # input_img = None #Image.open(args.dataset_name_or_path)
    # for sample in dataset:
    #     input_img = sample["image"]
    #     break

    # input_img.save("./dataset/inference/ds1.png")
    input_img2 = Image.open("./dataset/inference/IMG_20221010_152054_crop.jpg")

    start_time = time.perf_counter()
    prompt = task_prompt.replace("{user_input}", args.question) \
        if args.task_name == "docvqa" else task_prompt
    output = pretrained_model.inference(image=input_img2, prompt=prompt)["predictions"][0]
    end_time = time.perf_counter()
    print(f"Elapsed time: {end_time - start_time:.6f} seconds")

    ret = { "output": output }
    if args.save_path:
        save_json(args.save_path, ret)
    return ret

def inference_cord(image_file, pretrained_model_name_or_path):
    task_prompt = f"<s_cord>"

    pretrained_model = DonutModel.from_pretrained(args.pretrained_model_name_or_path)
    if torch.cuda.is_available():
        pretrained_model.half()
        device = torch.device("cuda")
        pretrained_model.to(device)
    else:
        pretrained_model.encoder.to(torch.bfloat16)
    pretrained_model.eval()

    # Read Image as PIL Image
    dataset = load_dataset(args.dataset_name_or_path, split="validation")
    input_img = None  # Image.open(args.dataset_name_or_path)
    for sample in dataset:
        input_img = sample["image"]
        break

    # input_img.save("./dataset/inference/ds1.png")
    input_img2 = Image.open("./dataset/inference/IMG_20221010_152054_crop.jpg")

    start_time = time.perf_counter()
    prompt = task_prompt.replace("{user_input}", args.question) \
        if args.task_name == "docvqa" else task_prompt
    output = pretrained_model.inference(image=input_img, prompt=prompt)["predictions"][0]
    end_time = time.perf_counter()
    print(f"Elapsed time: {end_time - start_time:.6f} seconds")

    ret = {"output": output}
    if args.save_path:
        save_json(args.save_path, ret)
    return ret



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dataset_name_or_path", type=str)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--question", type=str, default=None)
    args, left_argv = parser.parse_known_args()

    if args.task_name is None:
        args.task_name = os.path.basename(args.dataset_name_or_path)

    predictions = inference(args)
