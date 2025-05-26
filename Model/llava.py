import re
import logging
import os
from PIL import Image

import torch
# import llava-ov library
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

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
        # load llava-ov
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="balanced",
            low_cpu_mem_usage=True
        )
        
        # reduce the size of the processed image to reduce memory usage
        # min_pixels = 144 * 10 * 10  
        # max_pixels = 224 * 12 * 12  
        # min_pixels = 256 * 28 * 28
        # max_pixels = 768 * 28 * 28
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
            
            # process images, dynamically adjust the size based on the number of images
            processed_images = []
            # image_count = len(image_paths)
            
            # # dynamically adjust the image size - the more images, the smaller the size
            # if image_count <= 1:
            #     resize_dim = (576, 384)  # for 1 image, use larger size
            # elif image_count <= 2:
            #     resize_dim = (288, 200)
            # elif image_count <= 3:
            #     resize_dim = (192, 128)  # for 3 images, use medium size
            # elif image_count <= 4:
            #     resize_dim = (128, 96)  # for 4 images, use smaller size
            # else:
            #     resize_dim = (112, 80)  # For the 5 pictures, use the minimum size
            
            # logging.info(f"处理{image_count}张图片，调整尺寸为{resize_dim}")
            
            for img_path in image_paths:               
                # Open the image file
                try:
                    raw_image = Image.open(img_path).convert("RGB")
                    # Adjust the image size to reduce memory usage
                    raw_image = raw_image.resize((576, 384), Image.LANCZOS)
                    processed_images.append(raw_image)
                except Exception as e:
                    logging.error(f"打开图像失败 {img_path}: {str(e)}")
                    continue
            
            # If no image is processed successfully, an error is returned
            if not processed_images and image_paths:
                return "Response Error: Failed to process any images"
            
            # Prepare the model input
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

            # Extract the assistant response section
            assistant_index = response.find("assistant")
            if assistant_index != -1:
                final_answer = response[assistant_index + len("assistant"):].strip()
                return final_answer
            else:
                return response.strip()

        except Exception as e:
            print(e)
            return None

