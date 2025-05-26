from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
import json
from folder_processor import process_folders

def single_image_inference(model, processor, image_path, input_text):
    
    # Build the message format that satisfies the model requirements
    messages = [
        {
            "role": "user",  # user role
            "content": [
                {
                    "type": "image",  # image type
                    "image": image_path,  # use the local image path
                },
                {"type": "text", "text": input_text},  # text question
            ],
        }
    ]
    
    # Use the processor to prepare the dialog template (not tokenize)
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True  # add the generation prompt
    )
    
    # Process the visual information (assume process_vision_info is available)
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Prepare the model input
    inputs = processor(
        text=[text],  # text input
        images=image_inputs,  # image input
        videos=video_inputs,  # video input (should be empty in this case)
        padding=True,  # auto padding
        return_tensors="pt",  # return PyTorch tensor
    )
    inputs = inputs.to("cuda")  # move to GPU
    
    # Generate the answer (add token limit to get detailed reasoning process)
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=2048  # set a long enough token limit, from 512 to 2048
    )
    
    # Remove the input tokens, only keep the generated tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    # Decode the generated result
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True,  # skip the special tokens
        clean_up_tokenization_spaces=False  # keep the original spaces
    )[0]  # get the first result
    
    return output_text


if __name__ == "__main__":
    # Configure the path
    data_directory = "../Data/PhyTest"
    output_directory = "../Data/PhyTest" 
    
    model_path = '../../pretrained/QVQ'

    print("start!")
    
    # Load the model, automatically select the device distribution
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",  # automatically select the precision
        device_map="balanced_low_0"  # optimize device memory allocation
    )
    
    min_pixels = 256 * 28 * 28
    max_pixels = 768 * 28 * 28
    # Load the default processor
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels, use_fast=True)
    
    # Execute the processing (default max 311)
    process_folders(
        inference_function=single_image_inference,
        model=model,
        processor=processor,
        data_dir=data_directory,
        output_dir=output_directory,
        model_name="qvq-72b",
        max_folders=311
    )
    print("Done!")

