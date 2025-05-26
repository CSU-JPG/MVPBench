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
                f"{folder_name}_recall.json"
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
                        # 使用正则表达式同时匹配英文逗号 `,` 和中文逗号 `，`
                        sentences = re.split(r'[,，.]', item['original_scene']['description'])  # 同时匹配英文和中文逗号
                        sentences = [s.strip() for s in sentences if s.strip()]  # 去除空白并过滤空字符串

                        # 构建输入文本
                        input_text = (
                                "You are an expert system for verifying solutions to image-based problems. Your task is to match the ground truth middle steps with the provided solution. \n"
                                "\n" 
                                "INPUT FORMAT: \n"
                                "1. Problem: The original question/task\n"
                                "2. A Solution of a model\n"
                                "3. Ground Truth: Essential steps required for a correct answer\n"
                                "\n"
                                "MATCHING PROCESS:\n"
                                "\n"
                                "You need to match each ground truth middle step with the solution:\n"
                                "\n"
                                "Match Criteria:\n"
                                "- The middle step should exactly match in the content or is directly entailed by a certain content in the solution\n"
                                "- All the details must be matched, including the specific value and content\n"
                                "- You should judge all the middle steps for whethere there is a match in the solution\n"
                                "\n"
                                "OUTPUT FORMAT:\n"
                                "JSON array of judgments:\n"
                                "[\n"
                                "{{\n"
                                "\"step_index\": <integer>,\n"
                                "\"judgment\": \"Matched\" | \"Unmatched\",\n"
                                "}}\n"
                                "]\n"
                                "\n"
                                "ADDITIONAL RULES:\n"
                                "1. Only output the json array with no additional information.\n"
                                "2. Judge each ground truth middle step in order without omitting any step.\n"
                                "\n"
                                "Here is the problem, answer, solution, and the ground truth middle steps:\n"
                                "\n"
                                "[Problem]\n"
                                "\n"
                                f"{{{item['query']} }}\n"
                                "\n"
                                "[Answer]\n"
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

    data_directory = r"D:\VSCode_Project\Project-MLLM\Data\空间关系"

    # internvl2.5 internvl2.5MPO internvl3 internvl3-instruct llava-cot llavaov qvq qwen7b qwen72b
    output_directory = r"D:\VSCode_Project\Project-MLLM\Data\空间关系"

    # internvl_2.5_78B internvl_2.5MPO_78B internvl_3_78B internvl_3-instruct_78B llama3.2V-cot_11B llava_ov_72B qvq_72B qwen_2.5_vl_7B qwen_2.5_vl_72B
    # InternVL2_5-78B  InternVL2_5-78B-MPO InternVL3-78B-Instruct InternVL3-78B llama3.2V-cot-11B QVQ-72B qwen2_5vl-7b qwen2_5vl-72b
    # eureka-7b, qwen_2_vl_2B, r1_vl_2B
    # MM-Eureka,qwen2-vl-2b, r1-vl-2b
    md_name = "gpt-4o_multi"

    print("start!")
    
    # 执行处理（默认最多1个）
    process_experiment_folders(
        data_dir=data_directory,
        output_dir=output_directory,
        md_name = md_name,
        max_folders=1
    )
    print("Done!")