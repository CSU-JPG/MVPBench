import re
import logging
import base64
from io import BytesIO
from PIL import Image

# import qwen2.5_vl library
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# import qvq library
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import torch

def encode_image_to_base64(image):
    buffered = BytesIO()
    image = Image.open(image)
    # image = image.resize((256, 192), Image.LANCZOS)
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def create_message(sample):
    query = sample['query']
    all_contents = []
    matches = re.findall(r"<(image_\d+)>", query)
    split_text = re.split(r"<image_\d+>", query)
    for i, fragment in enumerate(split_text):
        if fragment.strip():
            all_contents.extend([
                {"type": "text", "text": fragment}
            ])
        if i < len(matches):
            if sample[matches[i]]:
                img_base64 = encode_image_to_base64(sample[matches[i]])
                all_contents.extend([
                    {
                        "type": "image",
                        "image": f"data:image/png;base64,{img_base64}"
                    }
                ])
            else:
                logging.error(
                    f"The image token {matches[i]} is in the query, but there is no corresponding image provided by the data")

    messages = [
        {
            "role": "user",
            "content": all_contents
        }
    ]
    return messages

class Qwen_Model:
    def __init__(
            self,
            model_path,
            temperature=0.1,
            max_tokens=1024
    ):
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        # load qwen2.5_vl
        #self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_path, torch_dtype=torch.bfloat16, device_map="balanced_low_0")
        # load qvq
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="balanced")
        min_pixels = 256 * 14 * 14
        max_pixels = 576 * 20 * 20
        self.processor = AutoProcessor.from_pretrained(self.model_path, use_fast=True, min_pixels=min_pixels, max_pixels=max_pixels)


    def get_response(self, sample):

        model = self.model
        processor = self.processor

        try:
            messages = create_message(sample)

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=self.max_tokens, temperature=self.temperature)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            return response[0]
        except Exception as e:
            print(e)
            return None

