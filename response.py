import argparse
import json
import os
import logging
from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets

from data_utils import load_yaml, verify_response, build_query, load_local_dataset

def get_model_name_for_file(model_path, model_api_name=''):
    """获取用于文件命名的模型名称"""
    if model_path:
        model_name = os.path.basename(model_path).lower()
        
        # InternVL系列模型
        if 'internvl2_5-78b-mpo' in model_name:
            return 'InternVL2_5-78B-MPO'
        elif 'internvl2_5-78b' in model_name:
            return 'InternVL2_5-78B'
        elif 'internvl3-78b-instruct' in model_name:
            return 'InternVL3-78B-Instruct'
        elif 'internvl3-78b' in model_name:
            return 'InternVL3-78B'
        
        # Llama系列模型
        elif 'llama3.2v-cot-11b' in model_name:
            return 'llama3.2V-cot-11B'
        
        # LLaVA系列模型
        elif 'llava-ov-hf-72b' in model_name:
            return 'llava-ov-hf-72B'
        
        # Qwen系列模型
        elif 'qwen2.5_vl_72b' in model_name:
            return 'qwen2.5_vl_72B'
        elif 'qwen2.5_vl_7b' in model_name:
            return 'qwen2.5_vl_7B'
        elif 'mm-eureka' in model_name:
            return 'MM-Eureka'
        elif 'qvq' in model_name:
            return 'QVQ-72B'
        
        # 如果以上都不匹配，返回原始名称但保持正确的大小写
        else:
            # 尝试从路径提取可能的模型名称
            path_parts = model_path.split('/')
            for part in reversed(path_parts):
                if part and not part.startswith('.'):
                    return part
            return os.path.basename(model_path)
    else:
        return model_api_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='Data/PhyTest', help='local data directory path')
    parser.add_argument('--subject', nargs='+', type=str, default='')
    parser.add_argument('--split', type=str, default='')
    parser.add_argument('--strategy', type=str, default='CoT', choices=['CoT', 'Direct'])
    parser.add_argument('--config_path', type=str, default="Configs/default.yaml")
    parser.add_argument('--output_dir', type=str, default='Results/', help='output directory for md files')
    parser.add_argument('--show_every', type=int, default=20, help='display progress every n problems')
    parser.add_argument('--rerun', action='store_true', help='rerun the answer generation')
    # Remote model
    parser.add_argument('--model', type=str, default="chatgpt-4o-latest", help='remote llm engine',
                        choices=['chatgpt-4o-latest', 'claude-3-5-sonnet-latest', 'gemini-2.0-flash-exp','gemini-2.0-flash-thinking-exp-1219', 'gemini-2.5-pro-exp-03-25'])
    parser.add_argument('--api_key', type=str, default='')
    # Local model
    parser.add_argument('--model_path', type=str, default='', help="local model path or huggingface model name")
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=0.7)

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load retmote Dataset
    if args.dataset_name:
        sub_dataset_list = []
        for subj in args.subject:
            sub_dataset = load_dataset(args.dataset_name, subj, split=args.split)
            sub_dataset_list.append(sub_dataset)
        dataset = concatenate_datasets(sub_dataset_list)
    #load local dataset
    elif args.data_dir:
        if args.subject and len(args.subject) > 0:
            # 如果指定了subject，则分别加载每个subject的数据
            sub_dataset_list = []
            for subj in args.subject:
                sub_dataset = load_local_dataset(args.data_dir, subj)
                sub_dataset_list.append(sub_dataset)
            dataset = concatenate_datasets(sub_dataset_list)
        else:
            # 如果没有指定subject，则加载所有数据
            dataset = load_local_dataset(args.data_dir)
        
        logging.info(f"loading {len(dataset)} data")
    # Load Config
    logging.info(f"Loading config")
    config = load_yaml(args.config_path)

    # Load Model
    # If we were given a custom path, load that model, otherwise use a remote service model
    if args.model_path:
        logging.info(f"Loading local model {args.model_path}")
        if 'llama' in args.model_path.lower():
            from Model import cot_llava
            model = cot_llava.Llava_Model(args.model_path, temperature=args.temperature, max_tokens=args.max_tokens)
        elif 'llava' in args.model_path.lower():
            from Model import llava
            model = llava.Llava_Model(args.model_path, temperature=args.temperature, max_tokens=args.max_tokens)

        elif 'qvq' in args.model_path.lower():
            from Model import qvq
            model = qvq.Qwen_Model(args.model_path, temperature=args.temperature, max_tokens=args.max_tokens)

        elif 'qwen2' in args.model_path.lower():
            from Model import qwen2vl
            model = qwen2vl.Qwen_Model(args.model_path, temperature=args.temperature, max_tokens=args.max_tokens)
        elif 'r1-vl' in args.model_path.lower():
            from Model import qwen2vl
            model = qwen2vl.Qwen_Model(args.model_path, temperature=args.temperature, max_tokens=args.max_tokens)

        elif 'qwen'  in args.model_path.lower():
            from Model import qwen
            model = qwen.Qwen_Model(args.model_path, temperature=args.temperature, max_tokens=args.max_tokens)
        elif 'mm-eureka' in args.model_path.lower():
            from Model import qwen
            model = qwen.Qwen_Model(args.model_path, temperature=args.temperature, max_tokens=args.max_tokens)

        elif 'internvl3' in args.model_path.lower():
            from Model import internvl3
            model = internvl3.Internvl_Model(args.model_path, temperature=args.temperature, max_tokens=args.max_tokens)
        elif 'internvl' in args.model_path.lower():
            from Model import internvl
            model = internvl.Internvl_Model(args.model_path, temperature=args.temperature, max_tokens=args.max_tokens)

    else:
        logging.info(f'error model argument')
        logging.info(f"Loading {args.model}")

        if 'gpt' in args.model.lower():
            from openai import OpenAI
            from Model import gpt
            client = OpenAI(api_key=args.api_key)
            model = gpt.GPT_Model(client, args.model, temperature=args.temperature, max_tokens=args.max_tokens)

        elif 'claude' in args.model.lower():
            from anthropic import Anthropic
            from Model import claude
            client = Anthropic(api_key=args.api_key)
            model = claude.Claude_Model(client, args.model, temperature=args.temperature, max_tokens=args.max_tokens)

        elif 'gemini' in args.model.lower():
            from openai import OpenAI
            from Model import gpt
            client = OpenAI(
                api_key=args.api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            model = gpt.GPT_Model(client, args.model, temperature=args.temperature, max_tokens=args.max_tokens)

    logging.info(f"Model loaded!")
    logging.info(model)

    # 获取已处理过的问题ID列表
    processed_pids = []
    invalid_files = []
    if not args.rerun:
        # 确保输出目录已存在
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
            logging.info(f"Created output directory: {args.output_dir}")
        
        # 获取当前模型名称
        current_model_suffix = get_model_name_for_file(args.model_path.lower(), args.model)
            
        # 检查目录中已有的文件
        logging.info(f"Checking for existing files with suffix '{current_model_suffix}' in {args.output_dir}")
        
        # 查找所有以当前模型名称结尾的文件
        for filename in os.listdir(args.output_dir):
            if filename.endswith(f"_{current_model_suffix}.md"):
                # 从文件名中提取问题ID
                parts = filename.split('_')
                if len(parts) >= 2:
                    folder_name = parts[0]  # 如：PhysTest
                    problem_id = parts[1]   # 如：0001（从原来的处理方式中拆分出来）
                    
                    # 构建完整ID
                    file_id = f"{folder_name}_{problem_id}"
                    
                    # 寻找对应的数据集ID
                    matching_pids = []
                    for dataset_pid in [sample['id'] for sample in dataset]:
                        # 直接匹配完整ID
                        if dataset_pid == file_id:
                            matching_pids.append(dataset_pid)
                            break
                    
                    if matching_pids:
                        dataset_pid = matching_pids[0]  # 使用找到的第一个匹配ID
                        
                        # 验证文件内容是否有效
                        filepath = os.path.join(args.output_dir, filename)
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            if verify_response(content):
                                processed_pids.extend(matching_pids)
                                logging.debug(f"Found valid file {filename} matching IDs: {matching_pids}")
                            else:
                                # 如果内容无效，添加到无效文件列表
                                invalid_files.append((dataset_pid, filename))
                                logging.warning(f"Found invalid content in {filename}, will reprocess")
                        except Exception as e:
                            # 文件读取出错，也认为是无效文件
                            invalid_files.append((folder_name, filename))
                            logging.warning(f"Error reading {filename}: {str(e)}, will reprocess")
                    else:
                        logging.warning(f"Found file {filename} but couldn't match to any dataset ID")
        
        if processed_pids:
            logging.info(f"Found {len(processed_pids)} valid existing processed problems for model {current_model_suffix}. Skipping these problems...")
        
        if invalid_files:
            logging.warning(f"Found {len(invalid_files)} invalid files that will be reprocessed")

    logging.info(f"Starting to generate.....")
    for idx, sample in enumerate(tqdm(dataset)):
        pid = sample['id']  # 获取问题ID
        
        # 提取文件夹名称（如0001）用于文件命名
        folder_name = os.path.basename(pid) if '/' in pid else pid
        
        # 跳过已处理的问题
        if not args.rerun and pid in processed_pids:
            continue
            
        # 获取模型名称用于文件命名
        model_suffix = get_model_name_for_file(args.model_path.lower(), args.model)
        
        # 构建输出文件路径，格式: <ID>_<MODEL>.md
        # 例如：0001_InternVL2_5-78B.md
        output_file = os.path.join(args.output_dir, f"{folder_name}_{model_suffix}.md")
        
        # 确保输出目录存在
        os.makedirs(args.output_dir, exist_ok=True)
        
        sample = build_query(sample, config, args.strategy)  # 根据策略构建查询

        try:
            # 获取模型响应
            response = model.get_response(sample)
            print(f"Generated response for {pid}")
            
            # 验证响应是否有效
            if verify_response(response):
                # 保存响应到.md文件
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(response)
                
                logging.info(f"Saved response for {pid} to {output_file}")
            else:
                logging.error(f"Invalid response for {pid}, skipping...")
                # 记录无效响应到错误日志文件
                error_file = os.path.join(args.output_dir, "errors.log")
                with open(error_file, 'a', encoding='utf-8') as f:
                    f.write(f"{pid}: Invalid response received\n")
                
        except Exception as e:
            logging.error(f"Error in generating answer for {pid}")
            logging.error(e)
            
            # 记录错误到错误日志文件
            error_file = os.path.join(args.output_dir, "errors.log")
            with open(error_file, 'a', encoding='utf-8') as f:
                f.write(f"{pid}: {str(e)}\n")

        # 打印进度
        if (idx + 1) % args.show_every == 0 or idx == len(dataset) - 1:
            logging.info(f"Processed {idx + 1}/{len(dataset)} problems")

    logging.info("End Generation......")


if __name__ == "__main__":
    logging.basicConfig( # 配置日志
        level=os.environ.get("LOGLEVEL", "INFO").upper(), # 从环境变量中获取日志级别，如果环境变量中没有该设置，则使用默认的 "INFO"
        format="[%(name)s] %(message)s", # 设置日志消息的输出格式。在这个配置中，日志将以 [模块名] 消息内容 的形式显示
        datefmt="[%X]" # 设置日志消息的时间格式。在这个配置中，时间将以 [HH:MM:SS] 的形式显示
    )
    logger_blocklist = [ # 将这些模块的日志级别设置为 WARNING，这样这些模块生成的日志仅显示 WARNING 级别及以上的信息。
        "asyncio",
        "azure",
        "azureml",
        "datasets",
        "hGttpx",
        "httpcore",
        "filelock",
        "fsspec",
        "msal",
        "msrest",
        "openai",
        "PIL",
        "urllib3",
    ]
    for module in logger_blocklist:
        logging.getLogger(module).setLevel(logging.WARNING) # 将这些模块的日志级别设为WARNIN

    main()


