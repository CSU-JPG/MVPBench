from openai import OpenAI
import os
import json
import re
import base64
import httpx

os.environ["http_proxy"] = "http://localhost:7897"
os.environ["https_proxy"] = "http://localhost:7897"
# 使用环境变量中的 API 密钥（建议方式）
client = OpenAI(base_url='https://api.nuwaapi.com/v1',
                http_client=httpx.Client(verify=False, timeout=60.0),
                api_key='sk-rl9sIDYarvkfKRV5kOB3ZfVUoZCwS43IbEawR6JHKtzznBmO')  

def image_to_base64(image_path):
    """将图片转为base64字符串（GPT-4o需要）"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def single_image_inference_gpt4o(image_path, input_text):
    # 加载图片并转为 base64
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
    """提取所有key_step的conclusion"""
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

    # 1. 去除首尾空白字符和BOM标记
    response = response.strip().replace('\ufeff', '')

    # 2. 去除开头的非JSON字符（直到第一个 [ 或 {）
    response = re.sub(r'^[^{[]*', '', response)

    # 3. 去除结尾的非JSON字符（从最后一个 ] 或 } 之后的所有内容）
    response = re.sub(r'[^}\]]*$', '', response)

    # 4. 特别处理常见的代码标记（如 ```json 和 ```）
    response = response.replace('```json', '').replace('```', '')

    # 5. 再次去除首尾空白字符
    return response.strip()

def process_experiment_folders(data_dir, output_dir, md_name, max_folders=1):
    """
    处理实验文件夹（最多处理指定数量）
    
    Args:
        data_dir: 实验文件夹根目录
        model_path: 模型路径
        output_dir: 输出目录
        md_name: 模型名称（用于查找.md文件）
        max_folders: 最大处理数量（默认10）
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
        os.makedirs(output_subdir, exist_ok=True)  # 添加这一行

        # 查找JSON和图片文件
        json_file = next((f for f in os.listdir(folder_path) if f.endswith('.json')), None)
        # internvl_2.5_78B internvl_2.5MPO_78B internvl_3_78B internvl_3-instruct_78B llama3.2V-cot_11B llava_ov_72B qvq_72B qwen_2.5_vl_7B qwen_2.5_vl_72B
        # InternVL2_5-78B  InternVL2_5-78B-MPO InternVL3-78B-Instruct InternVL3-78B llama3.2V-cot-11B QVQ-72B qwen2_5vl-7b qwen2_5vl-72b
        md_file = next((f for f in os.listdir(folder_path) if f.lower().endswith(f'{md_name.lower()}.md')), None)
        raw_image = next((f for f in os.listdir(folder_path) if f.endswith('_raw.jpg') or f.endswith('_raw.png')), None)
        
        if not (json_file and raw_image and md_file):
            print(f"跳过 {folder_name}：缺少JSON、图片或MD文件")
            processed_count += 1
            continue

        try:
            json_path = os.path.join(folder_path, json_file)
            md_path = os.path.join(folder_path, md_file)
            image_path = os.path.join(folder_path, raw_image)

            # 检查输出文件是否已存在且非空
            output_file = os.path.join(
                output_dir, 
                folder_name,
                f"{folder_name}_relevance.json"
            )
            
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                print(f"跳过 {folder_name}：输出文件已存在且非空")
                processed_count += 1
                continue

            with open(json_path, 'r', encoding='utf-8') as f_json:
                json_content = json.load(f_json)
                with open(md_path, 'r', encoding='utf-8') as f_md:
                    md_content = f_md.read().strip()

                    # 处理数组情况
                    if isinstance(json_content, list):
                        for item in json_content:
                            if not (isinstance(item, dict) and 'query' in item):
                                continue
                    
                        conclusions = extract_step_conclusions(item)

                        # 构建输入文本
                        input_text = (
                                "# Task Overview\n"
                                "Given a solution with multiple reasoning steps for an image-based problem, evaluate the relevance to get a solution (ignore correct or wrong) of each step.\n"
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
                                "# Step 2: Evaluating Relevancy\n"
                                "A relevant step is considered as: 75% content of the step must be related to trying to get a solution (ignore correct or wrong) to the question. \n"
                                "\n"
                                "**IMPORTANT NOTE**: \n"
                                "Evaluate relevancy independent of correctness. As long as the step is trying to get to a solution, it is considered relevant. Logical fallacy, knowledge mistake, inconsistent with previous steps, or other mistakes do not affect relevance.\n"
                                "A logically wrong step can be relevant if the reasoning attempts to address the question.\n"
                                "\n"
                                "The following behaviour is considered as relevant:\n"
                                "i. The step is planning, summarizing, thinking, verifying, calculating, or confirming an intermediate/final conclusion helpful to get a solution.\n"
                                "ii. The step is summarizing or reflecting on previously reached conclusion relevant to get a solution.\n"
                                "iii. Repeating the information in the question or give the final answer.\n"
                                "iv. A relevant image depiction shoule be in one of following situation: 1. help to obtain a conclusion helpful to solve the question later; 2. help to identify certain patterns in the image later; 3. directly contributes to the answer\n"
                                "v. Depicting or analyzing the options of the question is also relevant.\n"
                                "vi. Repeating previous relevant steps are also considered relevant. \n"
                                "\n"
                                "The following behaviour is considered as irrelevant: \n"
                                "i. Depicting image information that does not related to what is asking in the question.  Example: The question asks how many cars are present in all the images. If the step focuses on other visual elements like the road or building, the step is considered as irrelevant.\n"
                                "ii. Self-thought not related to what the question is asking.\n"
                                "iii. Other information that is tangential for answering the question.\n"
                                "\n"
                                "\n"
                                "# Output Format\n"
                                "[\n"
                                "  {\n"
                                "    \"step_type\": \"image description|logical inference|background information\",\n"
                                "    \"conclusion\": \"A brief summary of step result\",\n"
                                "    \"relevant\": \"Yes|No\"\n"
                                "  }\n"
                                "]\n"
                                "\n"
                                "\n"
                                "# Output Rules\n"
                                "Direct JSON output without any other output\n"
                                "Output at most 40 steps\n"
                                "\n"
                                "Here is the problem, and the solution that needs to be reformatted to steps:\n"
                                "\n"
                                "[Problem]\n"
                                "\n"
                                f"{{{item['query']} }}\n"
                                "\n"
                                "[Solution]\n"
                                "\n"
                                f"{{{md_content} }}\n"
                        )
                    
                        response = single_image_inference_gpt4o(image_path, input_text)
                        response=clean_model_response(response)

                        # 保存结果
                        with open(output_file, 'w', encoding='utf-8') as f_out:
                             f_out.write(response)
                        
                        print("已处理"+ str(processed_count+1)+"个数据")

                        processed_count += 1
                        
                        if processed_count >= max_folders:
                            break
                    else:
                        print(f"{json_path} 不是有效的数组格式")
        except Exception as e:
            print(f"处理 {folder_name} 时出错: {str(e)}")

if __name__ == "__main__":

    data_directory = r"D:\VSCode_Project\Project-MLLM\Mini-data\post-training\single\DynamicPrediction"

    # internvl2.5 internvl2.5MPO internvl3 internvl3-instruct llava-cot llavaov qvq qwen7b qwen72b
    output_directory = r"D:\VSCode_Project\Project-MLLM\Result\output_single_img\DynamicPrediction\r1_vl_2B"

    # internvl_2.5_78B internvl_2.5MPO_78B internvl_3_78B internvl_3-instruct_78B llama3.2V-cot_11B llava_ov_72B qvq_72B qwen_2.5_vl_7B qwen_2.5_vl_72B
    # InternVL2_5-78B  InternVL2_5-78B-MPO InternVL3-78B-Instruct InternVL3-78B llama3.2V-cot-11B QVQ-72B qwen2_5vl-7b qwen2_5vl-72b
    # eureka-7b, qwen_2_vl_2B, r1_vl_2B
    # MM-Eureka,qwen2-vl-2b, r1-vl-2b
    md_name = "r1_vl_2B"

    print("start!")
    
    # 执行处理（默认最多1个）
    process_experiment_folders(
        data_dir=data_directory,
        output_dir=output_directory,
        md_name = md_name,
        max_folders=100
    )
    print("Done!")
