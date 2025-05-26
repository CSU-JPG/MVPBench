from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
import json
import re

def single_image_inference(model, processor, image_path, input_text):
    
    # 构造符合模型要求的消息格式
    messages = [
        {
            "role": "user",  # 用户角色
            "content": [
                {
                    "type": "image",  # 图片类型
                    "image": image_path,  # 使用本地图片路径
                },
                {"type": "text", "text": input_text},  # 文本问题
            ],
        }
    ]
    
    # 使用处理器准备对话模板(不进行tokenize)
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True  # 添加生成提示
    )
    
    # 处理视觉信息(假设process_vision_info可用)
    image_inputs, video_inputs = process_vision_info(messages)
    
    # 准备模型输入
    inputs = processor(
        text=[text],  # 文本输入
        images=image_inputs,  # 图片输入
        videos=video_inputs,  # 视频输入(本例中应为空)
        padding=True,  # 自动填充
        return_tensors="pt",  # 返回PyTorch张量
    )
    inputs = inputs.to("cuda")  # 移动到GPU
    
    # 生成回答(增加token限制以获得详细推理过程)
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=2048  # 设置足够长的token限制，从512增加到2048
    )
    
    # 去除输入部分的token，只保留生成的token
    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    # 解码生成结果
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True,  # 跳过特殊token
        clean_up_tokenization_spaces=False  # 保留原始空格
    )[0]  # 取第一个结果
    
    return output_text

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


def process_experiment_folders(model, processor, data_dir, output_dir, max_folders=5):
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
    
    # 支持的图片格式列表，所有文件名都必须以_raw结尾
    supported_img_extensions = ['.jpg', '.png', '.jpeg', '.bmp', '.webp']
    
    for folder_name in os.listdir(data_dir):
        if processed_count >= max_folders:
            break
            
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        # 检查结果文件是否已存在
        result_folder = os.path.join(output_dir, folder_name)
        result_file = os.path.join(result_folder, f"{folder_name}_qwen_2.5_vl_72B.json")
        if os.path.exists(result_file):
            print(f"跳过 {folder_name}：结果文件已存在")
            continue

        # 查找JSON文件
        json_file = next((f for f in os.listdir(folder_path) if f.endswith('.json')), None)
        
        # 查找以_raw结尾的图片文件，支持多种格式
        raw_image = None
        for f in os.listdir(folder_path):
            if f.endswith('_raw' + supported_img_extensions[0]):  # 先尝试查找_raw.jpg
                raw_image = f
                break
                
        # 如果没找到_raw.jpg，尝试其他格式
        if not raw_image:
            for ext in supported_img_extensions[1:]:
                for f in os.listdir(folder_path):
                    if f.endswith('_raw' + ext):
                        raw_image = f
                        print(f"找到图片: {raw_image}")
                        break
                if raw_image:
                    break
        
        if not (json_file and raw_image):
            print(f"跳过 {folder_name}：缺少JSON或图片文件")
            continue

        try:
            json_path = os.path.join(folder_path, json_file)
            image_path = os.path.join(folder_path, raw_image)

            with open(json_path, 'r', encoding='utf-8') as f_json:
                json_content = json.load(f_json)
                
                # 处理数组情况
                if isinstance(json_content, list):
                    for item in json_content:
                        if not (isinstance(item, dict) and 'query' in item):
                            continue

                    conclusions = extract_step_conclusions(item)

                    # 构建输入文本
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
                
                    response=single_image_inference(model, processor, image_path, input_text)
                    response=clean_model_response(response)
                    
                    # 确保输出文件夹存在
                    os.makedirs(result_folder, exist_ok=True)
                    
                    # 保存结果
                    with open(result_file, 'w', encoding='utf-8') as f_out:
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
    # 配置路径
    data_directory = "../data1_noPro"
    output_directory = "../output1_noPro"  
    
    model_path = '../qwen_2.5_vl_72B/dir'

    print("yes!")
    print("start!")
    
    # 加载模型，自动选择设备分布
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",  # 自动选择精度
        device_map="balanced_low_0"  # 优化设备内存分配
    )
    
    min_pixels = 256 * 28 * 28
    max_pixels = 768 * 28 * 28
    # 加载默认处理器
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels, use_fast=True)
    
    # 执行处理（默认最多1个）
    process_experiment_folders(
        model=model,
        processor=processor,
        data_dir=data_directory,
        output_dir=output_directory,
        max_folders=300
    )
    print("Done!")
