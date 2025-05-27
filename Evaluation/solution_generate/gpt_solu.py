from openai import OpenAI
import os
import json
import re
import base64
import httpx

os.environ["http_proxy"] = "http://localhost:7897"
os.environ["https_proxy"] = "http://localhost:7897"
# Use API key from environment variables (recommended method)
client = OpenAI(base_url='',
                http_client=httpx.Client(verify=False, timeout=1000.0),
                api_key='')   

def image_to_base64(image_path):
    """Convert image to base64 string (required for GPT-4o)"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def single_image_inference_gpt4o(image_path, input_text, model_name):
    # Load image and convert to base64
    base64_image = image_to_base64(image_path)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": input_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        temperature=0.2,
        max_tokens=1024
    )

    return response.choices[0].message.content

def extract_step_conclusions(item):
    """Extract conclusions from all key_steps"""
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

def clean_model_response(response):

    # 1. Remove leading/trailing whitespace and BOM marker
    response = response.strip().replace('\ufeff', '')

    # 2. Remove non-JSON characters at the beginning (until the first [ or {)
    response = re.sub(r'^[^{[]*', '', response)

    # 3. Remove non-JSON characters at the end (after the last ] or })
    response = re.sub(r'[^}\]]*$', '', response)

    # 4. Special handling for common code markers (like ```json and ```)
    response = response.replace('```json', '').replace('```', '')

    # 5. Remove leading/trailing whitespace again
    return response.strip()

def process_experiment_folders(data_dir, output_dir, model_name, json_name, max_folders=1):
    """
    Process experiment folders (up to the specified number)
    
    Args:
        data_dir: Root directory of experiment folders
        model_path: Model path
        output_dir: Output directory
        max_folders: Maximum number to process (default 10)
    """
    os.makedirs(output_dir, exist_ok=True)
    processed_count = 0

    for folder_name in os.listdir(data_dir):
        if processed_count >= max_folders:
            break
            
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        output_subdir = os.path.join(output_dir, folder_name)
        os.makedirs(output_subdir, exist_ok=True)  # Add this line

        # Find JSON and image files
        json_file = next((f for f in os.listdir(folder_path) if f.endswith('.json')), None)
    
        raw_image = next((f for f in os.listdir(folder_path) if f.endswith('_raw.jpg') or f.endswith('_raw.png')), None)
        
        if not (json_file and raw_image):
            print(f"Skipping {folder_name}: Missing JSON or image file")
            continue

        try:
            json_path = os.path.join(folder_path, json_file)
            image_path = os.path.join(folder_path, raw_image)

            # Check if output file already exists and is not empty
            output_file = os.path.join(
                output_dir, 
                folder_name,
                f"{folder_name}_{json_name}_solu.json"
            )
            
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                print(f"Skipping {folder_name}: Output file already exists and is not empty")
                processed_count += 1
                continue

            with open(json_path, 'r', encoding='utf-8') as f_json:
                json_content = json.load(f_json)

                # Handle array case
                if isinstance(json_content, list):
                    for item in json_content:
                        if not (isinstance(item, dict) and 'query' in item):
                            continue
                
                    conclusions = extract_step_conclusions(item)

                    # Build input text
                    input_text = (
                            "You are given a question about a physical experiment and several key reasoning steps. \n"
                            "Your goal is to identify** ALL possible valid reasoning chains** that logically connect the question to the final answer.\n"
                            "Each reasoning chain should include** all key steps exactly once**, arranged in a logically valid order. \n"
                            "Steps may be combined in different logical orders as long as the overall reasoning makes sense.\n"
                            "\n" 
                            "Think carefully: there may be multiple valid chains based on how the steps can be logically ordered. \n"
                            "Your job is to find as many valid logical chains as possible.\n"
                            "INPUT FORMAT: \n"
                            "1. Question: The original question/task\n"
                            "2. Final Answer: Answer to the original question\n"
                            "2. Key Reasoning Steps: A list of essential reasoning steps, each with an ID and explanation.\n"
                            "\n"
                            "Output format (strictly JSON array)\n"
                            "\n"
                            "JSON array of judgments:\n"
                            "[\n"
                            "[\"key_step_1\", \"key_step_2\", \"key_step_3\"],\n"
                            "[\"key_step_1\", \"key_step_3\", \"key_step_2\"] \n"
                            "]\n"
                            "\n"
                            "ADDITIONAL RULES:\n"
                            "1. Only output the json array with no additional information.\n"
                            "\n"
                            "Here is the question, answer, and the Key Reasoning Steps:\n"
                            "\n"
                            "[Question]\n"
                            "\n"
                            f"{{{item['query']} }}\n"
                            "\n"
                            "[Final Answer]\n"
                            "\n" 
                            f"{{{item['final_scene']['annotation']} }}\n"
                            "\n"
                            "[Key Reasoning Steps]\n"
                            "\n"
                            "{\n"
                    )
                    for idx, conclusion in enumerate(conclusions, 1):
                        input_text += f'"key_step_{idx}", {conclusion}\n'
                    input_text+=(
                            "}\n"
                    )
                    
                    response = single_image_inference_gpt4o(image_path, input_text, model_name)
                    response=clean_model_response(response)

                    # Save results
                    with open(output_file, 'w', encoding='utf-8') as f_out:
                            f_out.write(response)
                    
                    print("Processed " + str(processed_count+1) + " data items")

                    processed_count += 1
                    
                    if processed_count >= max_folders:
                        break
                else:
                    print(f"{json_path} is not a valid array format")
        except Exception as e:
            print(f"Error processing {folder_name}: {str(e)}")

if __name__ == "__main__":
    
    data_directory = r"D:\VSCode_Project\Project-MLLM\Mini-data\PhyTest"
    output_directory = r"D:\VSCode_Project\Project-MLLM\Result\Mini-data\single_img\PhyTest\gemini-2.5-flash"
    #"grok-3","gemini-2.5-flash-preview-04-17"

    model_name="gemini-2.5-flash-preview-04-17"
    json_name="gemini-2.5-flash"
    
    print("start!")
    
    # Execute processing (default max 1)
    process_experiment_folders(
        data_dir=data_directory,
        output_dir=output_directory,
        model_name=model_name,
        json_name=json_name,
        max_folders=25
    )
    print("Done!")
