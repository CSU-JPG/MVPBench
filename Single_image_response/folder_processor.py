import os
import json
from typing import Callable, Any

def process_folders(
    inference_function: Callable[[Any, Any, str, str], str],
    model: Any, 
    processor: Any, 
    data_dir: str, 
    output_dir: str, 
    model_name: str,
    max_folders: int = 5
):
    """
    The function for generating responses from json files in the folder
    
    Args:
        inference_function: inference_function, 
        model: model, 
        processor: processor, 
        data_dir: data_dir, 
        output_dir: output_dir, 
        model_name: model_name, 
        max_folders: max_folders
    """
    os.makedirs(output_dir, exist_ok=True)
    processed_count = 0
    
    # supported_img_extensions, all file names must end with _raw
    supported_img_extensions = ['.jpg', '.png', '.jpeg', '.bmp', '.webp']
    
    for folder_name in os.listdir(data_dir):
        if processed_count >= max_folders:
            break
            
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        # check if the result file exists
        result_folder = os.path.join(output_dir, folder_name)
        result_file = os.path.join(result_folder, f"{folder_name}_{model_name}.md")
        if os.path.exists(result_file):
            print(f"跳过 {folder_name}：结果文件已存在")
            continue

        # find the JSON file
        json_file = next((f for f in os.listdir(folder_path) if f.endswith('.json')), None)
        
        # find the image file that ends with _raw, support multiple formats
        raw_image = None
        for f in os.listdir(folder_path):
            if f.endswith('_raw' + supported_img_extensions[0]):  # try to find _raw.jpg first
                raw_image = f
                break
                
        # if _raw.jpg is not found, try other formats
        if not raw_image:
            for ext in supported_img_extensions[1:]:
                for f in os.listdir(folder_path):
                    if f.endswith('_raw' + ext):
                        raw_image = f
                        print(f"found image: {raw_image}")
                        break
                if raw_image:
                    break
        
        if not (json_file and raw_image):
            print(f"skip {folder_name}: missing JSON or image file")
            continue

        try:
            json_path = os.path.join(folder_path, json_file)
            image_path = os.path.join(folder_path, raw_image)

            with open(json_path, 'r', encoding='utf-8') as f_json:
                json_content = json.load(f_json)
                
                # handle the array case
                if isinstance(json_content, list):
                    for item in json_content:
                        if not (isinstance(item, dict) and 'query' in item):
                            continue
                            
                        # build the input text
                        input_text = (
                            f"question: {item['query']} "
                            "According to the question and the image, "
                            "Please generate a step-by-step answer, include all your intermediate reasoning process, "
                            "and provide the final answer at the end."
                        )
                    
                        response = inference_function(model, processor, image_path, input_text)
                        
                        # ensure the output folder exists
                        os.makedirs(result_folder, exist_ok=True)
                        
                        # save the result
                        with open(result_file, 'w', encoding='utf-8') as f_out:
                            f_out.write(response)
                        
                        print(f"processed {processed_count+1} data")

                        processed_count += 1
                        
                        if processed_count >= max_folders:
                            break
                else:
                    print(f"{json_path} is not a valid array format")
        except Exception as e:
            print(f"error processing {folder_name}: {str(e)}")
    
    return processed_count 