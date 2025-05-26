from openai import OpenAI
import base64
import os
import json

client = OpenAI()

def image_to_base64(image_path):
    """将图片转为base64字符串（GPT-4o需要）"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def single_image_inference_gpt4o(image_path, input_text):
    base64_image = image_to_base64(image_path)
    
    response = client.chat.completions.create(
        model="o4-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": input_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        temperature=0.2,
        max_tokens=1024  # 可调，注意API限制
    )

    return response.choices[0].message.content

def process_experiment_folders(data_dir, output_dir, max_folders=5):
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
        result_file = os.path.join(result_folder, f"{folder_name}_gpt4o.md")
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

                    # 构建输入文本
                    input_text = (
                        f"question: {item['query']} "
                        "According to the question and the image, "
                        "Please generate a step-by-step answer, include all your intermediate reasoning process, "
                        "and provide the final answer at the end."
                    )
                
                    response=single_image_inference_gpt4o(image_path, input_text)
                    
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
    data_directory = r"C:\NeurIPS_Ben\data_PhySpatial"
    output_directory = r"C:\NeurIPS_Ben\data_PhySpatial" 

    print("start!")
    
    # 执行处理（默认最多1个）
    process_experiment_folders(
        data_dir=data_directory,
        output_dir=output_directory,
        max_folders=400
    )
    print("Done!")
