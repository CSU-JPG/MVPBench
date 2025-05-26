import os
import yaml
import json
import glob
from datasets import load_dataset, Dataset
from PIL import Image

def load_yaml(file_path):
    with open(file_path, 'r') as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
            return yaml_dict
        except yaml.YAMLError as exc:
            print(exc)
            return None


def verify_response(response):
    if isinstance(response, str):
        response = response.strip()
    if response == "" or response is None:
        return False
    if "Response Error" in response:
        return False
    return True


def build_query(sample, config, strategy):
    """construct query for model, process image path and query text"""
    res_dict = {}
    
    # 获取查询文本
    query = sample.get('query', '')
    
    # 创建含有图像引用的查询文本
    image_refs = []
    
    # 检查样本中是否已经包含图像路径
    for i in range(1, 10):
        img_key = f'image_{i}'
        if img_key in sample and sample[img_key]:
            #print(f"Found existing image path for {img_key}: {sample[img_key]}")
            image_refs.append({'key': img_key, 'path': sample[img_key]})
    
    # 如果没有找到直接的图像路径，则尝试从original_scene和key_step中提取
    if not image_refs:
        # 处理original_scene(key,path)
        if 'original_scene' in sample  and 'full_path' in sample['original_scene']:
            image_refs.append({'key': 'image_1', 'path': sample['original_scene']['full_path']})
        
        cnt = 1
        # 处理key_step_1到key_step_9图像(key,path)
        for i in range(1, 10):
            step_key = f'key_step_{i}'
            if step_key not in sample:
                break
            if (sample[step_key] is not None and 'full_path' in sample[step_key] and sample[step_key]['full_path'] is not None):
                image_refs.append({'key': f'image_{i+1}', 'path': sample[step_key]['full_path']})
                cnt += 1            

        if 'final_scene' in sample  and 'full_path' in sample['final_scene']:
            image_refs.append({'key': f'image_{cnt+1}', 'path': sample['final_scene']['full_path']}) 
    
    #print(image_refs)
    # 构建带有图像标记的查询
    formatted_query = query
    
    # 检查查询中是否已经包含图像标记
    has_image_tokens = any(f"<image_{i}>" in query for i in range(1, 10))
    
    if not has_image_tokens:
        #print(f"No image tokens in query, adding them manually.")
        # 手动添加图像标记
        for i, img_ref in enumerate(image_refs, 1):
            # 在查询文本中添加<image_n>标记
            if i == 1:  # 第一张图片放在问题开头,表示原始图片
                formatted_query = f"<{img_ref['key']}>\n {formatted_query}"
            else: # 其他图片放在问题结尾，表示关键步骤示意图
                if i == 2:
                    formatted_query += '\nThe following image is the key step illustration:'
                formatted_query = f"{formatted_query} <{img_ref['key']}>"
    else:
        print(f"Query already contains image tokens.")
    
    # 根据策略添加指令
    if strategy == 'CoT':
        formatted_query += config['Strategy_Instruction']['CoT']
    else:
        formatted_query += config['Strategy_Instruction']['Directly']
    
    # 添加prompt指令
    formatted_query += config['Prompt_Instruction']
    # 将图像路径添加到结果字典中
    for img_ref in image_refs:
        # 直接使用路径而不是加载图像
        res_dict[img_ref['key']] = img_ref['path']
    
    # 把原始样本的其他字段合并到结果中
    res_dict.update(sample)
    res_dict['query'] = formatted_query
    
    #print(f"Final query with image tokens: {formatted_query}")
    return res_dict

def load_local_dataset(data_dir, subject=None):
    # 确保data_dir是绝对路径
    data_dir = os.path.abspath(data_dir)
    
    # 获取指定目录下所有JSON文件
    json_files = glob.glob(f"{data_dir}/**/*.json", recursive=True)
    
    # 读取所有JSON文件
    all_data = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 处理列表格式的JSON
                if isinstance(data, list):
                    for item in data:
                        # 处理单个item
                        processed_item = process_item(item, data_dir, subject, json_file)
                        if processed_item:
                            all_data.append(processed_item)
                
                # 处理字典格式的JSON（单个问题）
                elif isinstance(data, dict):
                    processed_item = process_item(data, data_dir, subject, json_file)
                    if processed_item:
                        all_data.append(processed_item)
        except Exception as e:
            print(f"reading file {json_file} occurred error: {e}")
            continue
    
    print(f"successfully loaded {len(all_data)} data")
    # 转换为Dataset对象
    return Dataset.from_list(all_data)

def process_item(item, data_dir, subject=None, json_file=None):
    """process single data item"""
    if not isinstance(item, dict):
        return None
        
    # 检查subject过滤条件
    if subject is not None and (not item.get('subject') or item.get('subject') != subject):
        return None
    
    # 添加json_path字段（如果需要）
    if 'json_path' not in item and json_file:
        item['json_path'] = json_file
    
    # 处理图像路径
    process_image_path(item, 'original_scene', data_dir)
    
    # 处理final_scene字段
    process_image_path(item, 'final_scene', data_dir)
    
    # 处理key_step字段
    for i in range(1, 10):  # 假设最多有9个key_step
        step_key = f'key_step_{i}'
        if step_key in item:
            process_image_path(item, step_key, data_dir)
        else: break
    return item

def process_image_path(item, field, data_dir):
    """process image path in data item"""
    if field in item and 'path' in item[field]:
        path = item[field]['path']
        
        # 如果路径为空，不处理
        if not path or path.strip() == "":
            return
        
        # 打印调试信息
        #print(f"Processing image path for {field}: {path}")
        
        # 统一路径格式
        data_dir = os.path.normpath(data_dir)
        path = os.path.normpath(path)
        
        # 尝试多种可能的路径组合方式
        possible_paths = [
            # 直接使用提供的路径
            os.path.join(data_dir, path),
            
            # 如果路径中包含PhyTest且data_dir包含Data/PhyTest，去除PhyTest前缀
            os.path.join(data_dir, path.replace('PhyTest/', '', 1)) if path.startswith('PhyTest/') else None,
            
            # 尝试在Data目录下查找
            os.path.join(os.path.dirname(data_dir), path) if 'Data' in data_dir else None,
            
            # 绝对路径
            path if os.path.isabs(path) else None
        ]
        
        # 过滤掉None值
        possible_paths = [p for p in possible_paths if p is not None]
        
        # 检查每个可能的路径
        found = False
        for full_path in possible_paths:
            if os.path.isfile(full_path):
                item[field]['full_path'] = full_path
                item[field]['file_exists'] = True
                #print(f"Found image file: {full_path}")
                found = True
                break
        
        if not found:
            # 如果所有尝试都失败，使用第一个路径作为默认值
            item[field]['full_path'] = possible_paths[0] if possible_paths else os.path.join(data_dir, path)
            item[field]['file_exists'] = False
            print(f"Image file not found after trying multiple paths. Using: {item[field]['full_path']}")
            # 输出所有尝试过的路径，以便调试
            print(f"Attempted paths: {possible_paths}")
