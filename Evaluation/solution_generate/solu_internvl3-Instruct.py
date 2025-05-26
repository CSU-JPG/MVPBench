import math
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
import os
import json
import re

model_path = '../../junchao/pretrained/InternVL3-78B-Instruct'

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 7
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
    return device_map

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1]
    )
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = [
        resized_img.crop((
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )) for i in range(blocks)
    ]
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(im) for im in images])

def single_image_inference(model, tokenizer, image_path, input_text):
    pixel_values = load_image(image_path, max_num=2).to(torch.bfloat16).cuda()
    question = f'<image>\n{input_text}'
    response = model.chat(tokenizer, pixel_values, question, dict(max_new_tokens=1024, do_sample=False))
    return response

def clean_model_response(response):
    response = response.strip().replace('\ufeff', '')
    response = re.sub(r'^[^{[]*', '', response)
    response = re.sub(r'[^}\]]*$', '', response)
    return response.replace('```json', '').replace('```', '').strip()

def extract_step_conclusions(item):
    conclusions = []
    i = 1
    while True:
        step_key = f"key_step_{i}"
        if step_key not in item:
            break
        step_data = item[step_key]
        if "conclusion" in step_data:
            conclusions.append(step_data["conclusion"])
        i += 1
    return conclusions

def process_experiment_folders(model, tokenizer, data_dir, output_dir, max_folders=5):
    os.makedirs(output_dir, exist_ok=True)
    processed_count = 0
    supported_img_extensions = ['.jpg', '.png', '.jpeg', '.bmp', '.webp']
    for folder_name in os.listdir(data_dir):
        if processed_count >= max_folders:
            break
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path): continue
        result_folder = os.path.join(output_dir, folder_name)
        result_file = os.path.join(result_folder, f"{folder_name}_internvl_3_instruct_solu.json")
        if os.path.exists(result_file):
            print(f"跳过 {folder_name}：结果文件已存在")
            continue
        json_file = next((f for f in os.listdir(folder_path) if f.endswith('.json')), None)
        raw_image = next((f for ext in supported_img_extensions for f in os.listdir(folder_path) if f.endswith('_raw' + ext)), None)
        if not (json_file and raw_image):
            print(f"跳过 {folder_name}：缺少JSON或图片文件")
            continue
        try:
            json_path = os.path.join(folder_path, json_file)
            image_path = os.path.join(folder_path, raw_image)
            with open(json_path, 'r', encoding='utf-8') as f_json:
                json_content = json.load(f_json)
                if isinstance(json_content, list):
                    for item in json_content:
                        if not (isinstance(item, dict) and 'query' in item): continue
                    conclusions = extract_step_conclusions(item)
                    input_text = (
                        "You are given a question about a physical experiment and several key reasoning steps.\n"
                        "Your goal is to identify** ALL possible valid reasoning chains** that logically connect the question to the final answer.\n"
                        "Each reasoning chain should include** all key steps exactly once**, arranged in a logically valid order.\n"
                        "Steps may be combined in different logical orders as long as the overall reasoning makes sense.\n\n"
                        "Think carefully: there may be multiple valid chains based on how the steps can be logically ordered.\n"
                        "Your job is to find as many valid logical chains as possible.\n\n"
                        "INPUT FORMAT:\n"
                        "1. Question: The original question/task\n"
                        "2. Final Answer: Answer to the original question\n"
                        "3. Key Reasoning Steps: A list of essential reasoning steps, each with an ID and explanation.\n\n"
                        "Output format (strictly JSON array):\n"
                        "[\n"
                        "[\"key_step_1\", \"key_step_2\", \"key_step_3\"],\n"
                        "[\"key_step_1\", \"key_step_3\", \"key_step_2\"]\n"
                        "]\n\n"
                        "ADDITIONAL RULES:\n"
                        "1. Only output the json array with no additional information.\n\n"
                        "Here is the question, answer, and the Key Reasoning Steps:\n\n"
                        f"[Question]\n{item['query']}\n\n"
                        f"[Final Answer]\n{item['final_scene']['annotation']}\n\n"
                        "[Key Reasoning Steps]\n{\n"
                    )
                    for idx, conclusion in enumerate(conclusions, 1):
                        input_text += f'"key_step_{idx}": {conclusion}\n'
                    input_text += "}\n"
                    response = single_image_inference(model, tokenizer, image_path, input_text)
                    response = clean_model_response(response)
                    os.makedirs(result_folder, exist_ok=True)
                    with open(result_file, 'w', encoding='utf-8') as f_out:
                        f_out.write(response)
                    print("已处理 " + str(processed_count + 1) + " 个数据")
                    processed_count += 1
                    if processed_count >= max_folders:
                        break
                else:
                    print(f"{json_path} 不是有效的数组格式")
        except Exception as e:
            print(f"处理 {folder_name} 时出错: {str(e)}")

if __name__ == "__main__":
    data_directory = "../data1_noPro"
    output_directory = "../output1_noPro"
    print("yeah")
    print("start!")
    device_map = split_model('InternVL3-78B')
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    process_experiment_folders(model, tokenizer, data_directory, output_directory, max_folders=300)
    print("Done!")
