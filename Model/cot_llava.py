import re
import logging
import os
from PIL import Image

import torch
# import llava-cot library
from transformers import MllamaForConditionalGeneration, AutoProcessor

def create_message(sample):
    query = sample['query']
    all_contents = []
    matches = re.findall(r"<(image_\d+)>", query)
    split_text = re.split(r"<image_\d+>", query)
    images = []
    for i, fragment in enumerate(split_text):
        if fragment.strip():
            all_contents.extend([
                {"type": "text", "text": fragment}
            ])
        if i < len(matches):
            if sample[matches[i]]:
                all_contents.extend([
                    {"type": "image"}
                ])
                images.append(sample[matches[i]])
            else:
                logging.error(
                    f"The image token {matches[i]} is in the query, but there is no corresponding image provided by the data")
    messages = [
        {
            "role": "user",
            "content": all_contents
        }
    ]
    return messages, images


class Llava_Model:
    def __init__(
            self,
            model_path,
            temperature=0.7,
            max_tokens=1024
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="balanced",
            low_cpu_mem_usage=True
        )
        
        # min_pixels = 256 * 14 * 14
        # max_pixels = 416 * 18 * 18
        # self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True, min_pixels=min_pixels, max_pixels=max_pixels)
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)


    def get_response(self, sample):

        model = self.model
        processor = self.processor

        try:
            messages, image_paths = create_message(sample)
            print(messages)
            print(image_paths)

            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            
            # Image processing: Dynamically adjust the size based on the number of images
            processed_images = []
            for img_path in image_paths:               
                # open image file
                try:
                    raw_image = Image.open(img_path).convert("RGB")
                    # resize image to reduce memory usage
                    raw_image = raw_image.resize((576, 384), Image.LANCZOS)
                    processed_images.append(raw_image)
                except Exception as e:
                    logging.error(f"打开图像失败 {img_path}: {str(e)}")
                    continue
            
            # if no image is processed, return error
            if not processed_images and image_paths:
                return "Response Error: Failed to process any images"
            
            # prepare model input
            inputs = processor(
                images=processed_images,
                text=input_text,
                add_special_tokens=False,              
                return_tensors="pt"
            ).to(model.device, torch.float16)


            output = model.generate(**inputs, 
                                    do_sample=True,
                                    temperature=self.temperature, 
                                     max_new_tokens=self.max_tokens)
            
            response = processor.decode(output[0], skip_special_tokens=True)

            # extract assistant response
            assistant_index = response.find("assistant")
            if assistant_index != -1:
                final_answer = response[assistant_index + len("assistant"):].strip()
                return final_answer
            else:
                return response.strip()

        except Exception as e:
            print(e)
            return None

