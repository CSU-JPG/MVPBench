import re
import logging
import os
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
import math
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def split_model(model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
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


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


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
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    try:
        if not os.path.exists(image_file):
            logging.error(f"Image file does not exist: {image_file}")
            return None 
        image = Image.open(image_file).convert('RGB')
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)           
        return pixel_values
    
    except Exception as e:
        logging.error(f"Error loading image {image_file}: {e}")           
        return None


def process_query(sample):
    query = sample['query'] # receive a sample dictionary containing the query key
    #print(f"formatted query: {query}")
    #print(f"Sample keys: {list(sample.keys())}")
    
    # use different regular expressions to match <image_n>
    # Note: The regular expression is changed from non-greedy matching to greedy matching to ensure the capture of the complete token
    matches = re.findall(r"<(image_\d+)>", query)
    
    # replace the image token in the query
    modified_query = re.sub(r"<image_\d+>", "<image>", query)
    images = []
    
    # check if the image token is found
    if not matches:
        logging.error(f"No image tokens found in query: {query}")
    
    # process the found image token
    for match in matches:
        if match in sample and sample[match]:
            #print(f"Found image path for {match}: {sample[match]}")
            # ensure the image path exists
            if os.path.exists(sample[match]):
                images.append(sample[match])
                #print(f"Verified image exists at path: {sample[match]}")
            else:
                logging.error(f"Image file does not exist: {sample[match]}")
        else:
            logging.error(f"The image token <{match}> is in the query, but there is no corresponding image provided in the data")
    
    # check if the processed image list is empty
    if not images:
        logging.error(f"No valid images found for query with {len(matches)} image tokens")
    
    #print(f"modified query: {modified_query}")
    #print(f"images: {images}")
    return modified_query, images


class Internvl_Model:
    def __init__(
            self,
            model_path,
            temperature=0.7,
            max_tokens=1024
    ):
            
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device_map = split_model(model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=self.device_map).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        

    def get_response(self, sample):
        
        try:
            query, images = process_query(sample)
            pixel_values_list = []
            num_patches_list = []

            for image in images:              
                pixel_value = load_image(image, max_num=2, input_size=448)
                if pixel_value is None:
                    logging.error(f"Failed to load image: {image}")
                    continue
                    
                # convert to lower precision before transferring to GPU
                pixel_value = pixel_value.to(torch.bfloat16).cuda()
                pixel_values_list.append(pixel_value)
                num_patches_list.append(pixel_value.size(0))

                # check if pixel_values_list is empty
            if not pixel_values_list:
                print("Error: No images found in the query or images could not be processed.")
                return None
                
            pixel_values = torch.cat(pixel_values_list, dim=0)

            generation_config = dict(max_new_tokens=self.max_tokens, do_sample=True, temperature=self.temperature)
            
            # single-image single-round conversation
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                query,
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=False
                )
            
            return response
        except Exception as e:
            print(e)           
            return None
