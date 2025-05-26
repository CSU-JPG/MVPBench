from transformers import MllamaForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
import os
import json
from folder_processor import process_folders

def single_image_inference(model, processor, image_path, input_text):   
    # Build the message format that satisfies the model requirements
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": input_text}
        ]}
    ]
    
    # Use the processor to prepare the dialog template
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    # Load the image
    raw_image = Image.open(image_path)
    # Resize the image to a uniform resolution (576,384)
    raw_image = raw_image.resize((576, 384), Image.LANCZOS)
    
    # Prepare the model input
    inputs = processor(
        raw_image,
        prompt,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device, torch.bfloat16)
    
    # Generate the answer
    output = model.generate(
        **inputs, 
        max_new_tokens=2048,
        do_sample=False,
        temperature=0.6,
        top_p=0.9,
    )
    
    # Decode the generated result
    output_text = processor.decode(output[0], skip_special_tokens=True)
    
    return output_text


if __name__ == "__main__":
    # Configure the path
    data_directory = "../Data/PhyTest"
    output_directory = "../Data/PhyTest" 
    
    model_path = '../../pretrained/llama3.2V-cot-11B'

    print("start!")
    
    # Load the model, automatically select the device distribution
    model = MllamaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  
        device_map="balanced_low_0",  # Optimize device memory allocation
        low_cpu_mem_usage=True,
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
        model_name="llama3.2V-cot-11b",
        max_folders=311
    )
    print("Done!")

