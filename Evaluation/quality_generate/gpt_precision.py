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
                http_client=httpx.Client(verify=False, timeout=60.0),
                api_key='')  

def image_to_base64(image_path):
    """Convert image to base64 string (required for GPT-4o)"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def single_image_inference_gpt4o(image_path, input_text):
    # Load image and convert to base64
    base64_image = image_to_base64(image_path)

    response = client.chat.completions.create(
        model="gpt-4o",
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

def process_experiment_folders(data_dir, output_dir, md_name, max_folders=1):
    """
    Process experiment folders (up to the specified number)
    
    Args:
        data_dir: Root directory of experiment folders
        model_path: Model path
        output_dir: Output directory
        md_name: Model name (used to find .md files)
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
        
        # Match MD files case-insensitively
        md_file = next((f for f in os.listdir(folder_path) if f.lower().endswith(f'{md_name.lower()}.md')), None)
        raw_image = next((f for f in os.listdir(folder_path) if f.endswith('_raw.jpg') or f.endswith('_raw.png')), None)
        
        if not (json_file and raw_image and md_file):
            print(f"Skipping {folder_name}: Missing JSON, image, or MD file")
            # Output more detailed information for debugging
            if not json_file:
                print(f"  - Missing JSON file")
            if not raw_image:
                print(f"  - Missing raw image file")
            if not md_file:
                print(f"  - Missing MD file ({md_name}.md)")
                # List all md files for comparison
                md_files = [f for f in os.listdir(folder_path) if f.endswith('.md')]
                if md_files:
                    print(f"  - MD files in folder: {md_files}")
            processed_count += 1
            continue

        try:
            json_path = os.path.join(folder_path, json_file)
            md_path = os.path.join(folder_path, md_file)
            image_path = os.path.join(folder_path, raw_image)

            # Check if output file already exists and is not empty
            output_file = os.path.join(
                output_dir, 
                folder_name,
                f"{folder_name}_presicion.json"
            )
            
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                print(f"Skipping {folder_name}: Output file already exists and is not empty")
                continue

            with open(json_path, 'r', encoding='utf-8') as f_json:
                json_content = json.load(f_json)
                with open(md_path, 'r', encoding='utf-8') as f_md:
                    md_content = f_md.read().strip()

                    # Handle array case
                    if isinstance(json_content, list):
                        for item in json_content:
                            if not (isinstance(item, dict) and 'query' in item):
                                continue
                    
                        conclusions = extract_step_conclusions(item)
                        # Use regex to match both English commas `,` and Chinese commas `，`
                        sentences = re.split(r'[,，.]', item['original_scene']['description'])  # Match both English and Chinese commas
                        sentences = [s.strip() for s in sentences if s.strip()]  # Remove whitespace and filter empty strings
                        # Build input text
                        input_text = (
                                "# Task Overview\n"
                                "Given a solution with multiple reasoning steps for an image-based problem, reformat it into well-structured steps and evaluate their correctness.\n"
                                "\n"
                                "# Step 1: Reformatting the Solution\n"
                                "Convert the unstructured solution into distinct reasoning steps while:\n"
                                "- Preserving all original content and order\n"
                                "- Not adding new interpretations\n"
                                "- Not omitting any steps\n"
                                "\n"
                                "## Step Types\n"
                                "1. Logical Inference Steps\n"
                                "   - Contains exactly one logical deduction\n"
                                "   - Must produce a new derived conclusion\n"
                                "   - Cannot be just a summary or observation\n"
                                "\n"
                                "2. Image Description Steps\n"
                                "   - Pure visual observations\n"
                                "   - Only includes directly visible elements\n"
                                "   - No inferences or assumptions\n"
                                "\n"
                                "3. Background Information Steps\n"
                                "   - External knowledge or question context\n"
                                "   - No inference process involved\n"
                                "\n"
                                "## Step Requirements\n"
                                "- Each step must be atomic (one conclusion per step)\n"
                                "- No content duplication across steps\n"
                                "- Initial analysis counts as background information\n"
                                "- Final answer determination counts as logical inference\n"
                                "\n"
                                "# Step 2: Evaluating Correctness\n"
                                "Evaluate each step against:\n"
                                "\n"
                                "## Ground Truth Matching\n"
                                "For image descriptions:\n"
                                "- Key elements must match ground truth descriptions\n"
                                "\n"
                                "For logical inferences:\n"
                                "- Conclusion must EXACTLY match or be DIRECTLY entailed by ground truth\n"
                                "\n"
                                "## Reasonableness Check (if no direct match)\n"
                                "Step must:\n"
                                "- Premises must not contradict any ground truth or correct answer\n"
                                "- Logic is valid\n"
                                "- Conclusion must not contradict any ground truth \n"
                                "- Conclusion must support or be neutral to correct answer\n"
                                "\n"
                                "\n"
                                "## Judgement Categories\n"
                                "- \"Match\": Aligns with ground truth\n"
                                "- \"Reasonable\": Valid but not in ground truth\n"
                                "- \"Wrong\": Invalid or contradictory\n"
                                "- \"N/A\": For background information steps\n"
                                "\n"
                                "## Final Answer\n"
                                "- Final answer score: Match the model's final answer with the ground truth answer, scoring 1 if it matches and 0 if it doesn't.\n"
                                "- Only the last logical inference step must include the final_answer field.\n"
                                "- Do not include the final_answer field in any other step.\n"
                                "\n"
                                "# Output Requirements\n"
                                "1. The output format MUST be in valid JSON format without ANY other content.\n"
                                "2. For highly repetitive patterns, output it as a single step.\n"
                                "3. Output maximum 35 steps. Always include the final step that contains the answer.\n"
                                "\n"
                                "Here is the json output format:\n"
                                "## Output Format\n"
                                "[\n"
                                "  {{\n"
                                "    \"step_type\": \"image description|logical inference|background information\",\n"
                                "    \"premise\": \"Evidence (only for logical inference)\",\n"
                                "    \"conclusion\": \"Step result\",\n"
                                "    \"judgment\": \"Match|Reasonable|Wrong|N/A\",\n"
                                "    \"final_answer\":\"1|0\"\n"
                                "  }}\n"
                                "]\n"
                                "\n"
                                "Here is the problem, and the solution that needs to be reformatted to steps:\n"
                                "\n"
                                "[Problem]\n"
                                "\n"
                                f"{{{item['query']} }}\n"
                                "\n"
                                "[Correct Answer]\n"
                                "\n" 
                                f"{{{item['final_scene']['annotation']} }}\n"
                                "\n"
                                "[Solution]\n"
                                "\n"
                                f"{{{md_content} }}\n"
                                "\n"
                                "[Ground Truth Information]\n"
                                "\n"
                                "{\n"
                        )
                        idx = 1
                        for i, sentence in enumerate(sentences, 1):
                            input_text += f"{i}. {sentence}\n"
                            idx += 1
                        for i, conclusion in enumerate(conclusions, 1):
                            input_text += f"{idx}. {conclusion}\n"
                            idx += 1
                        input_text+=(
                                "}\n"
                        )
                        
                        response = single_image_inference_gpt4o(image_path, input_text)
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
   
    data_directory = r"D:\VSCode_Project\Project-MLLM\Data\空间关系"

    # internvl2.5 internvl2.5MPO internvl3 internvl3-instruct llava-cot llavaov qvq qwen7b qwen72b
    output_directory = r"D:\VSCode_Project\Project-MLLM\Data\空间关系"

    # internvl_2.5_78B internvl_2.5MPO_78B internvl_3_78B internvl_3-instruct_78B llama3.2V-cot_11B llava_ov_72B qvq_72B qwen_2.5_vl_7B qwen_2.5_vl_72B
    # InternVL2_5-78B  InternVL2_5-78B-MPO InternVL3-78B-Instruct InternVL3-78B llama3.2V-cot-11B QVQ-72B qwen2_5vl-7b qwen2_5vl-72b
    # eureka-7b, qwen_2_vl_2B, r1_vl_2B
    # MM-Eureka,qwen2-vl-2b, r1-vl-2b
    md_name = "gpt-4o_multi"

    print("start!")
    
    # Execute processing (default max 1)
    process_experiment_folders(
        data_dir=data_directory,
        output_dir=output_directory,
        md_name = md_name,
        max_folders=1
    )
    print("Done!")
