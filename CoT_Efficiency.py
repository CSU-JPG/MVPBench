import os
import json

def calculate_relevant_ratio(data):
    """计算'relevant'为'Yes'的比例"""
    total = len(data)
    if total == 0:
        return 0.0
    yes_count = sum(1 for item in data if item.get("relevant") == "Yes")
    return yes_count / total

def calculate_correct_ratio(data):
    """计算'judgment'为'Correct'的比例"""
    total = len(data)
    if total == 0:
        return 0.0
    correct_count = sum(1 for item in data if item.get("judgment") == "Correct")
    return correct_count / total

def analyze_relevance_and_reflection(base_dir):
    relevance_ratios = []
    reflection_ratios = []
    relevance_count = 0
    reflection_count = 0

    for folder in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # --- relevance.json ---
        relevance_file = next((f for f in os.listdir(folder_path) if f.endswith('relevance.json')), None)
        if relevance_file:
            try:
                with open(os.path.join(folder_path, relevance_file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                ratio = calculate_relevant_ratio(data)
                relevance_ratios.append(ratio)
                relevance_count += 1
                print(f"[{folder}] relevance.json: 'Yes'比例 = {ratio:.2%}")
            except Exception as e:
                print(f"[{folder}] 处理 relevance.json 出错: {e}")
        else:
            print(f"警告：{folder} 中未找到 relevance.json")

        # --- reflection.json ---
        reflection_file = next((f for f in os.listdir(folder_path) if f.endswith('reflection.json')), None)
        if reflection_file:
            try:
                with open(os.path.join(folder_path, reflection_file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                ratio = calculate_correct_ratio(data)
                reflection_ratios.append(ratio)
                reflection_count += 1
                print(f"[{folder}] reflection.json: 'Correct'比例 = {ratio:.2%} (共{len(data)}条)")
            except Exception as e:
                print(f"[{folder}] 处理 reflection.json 出错: {e}")
        else:
            print(f"警告：{folder} 中未找到 reflection.json")

    # --- 汇总输出 ---
    print("\n========== 统计结果 ==========")
    if relevance_ratios:
        avg_relevance = sum(relevance_ratios) / len(relevance_ratios)
        avg_relevance = (avg_relevance-0.8) * 5
        print(f"处理了 {relevance_count} 个 relevance.json 文件")
        print(f"平均 'Yes' 比例：{avg_relevance:.2%}")
    else:
        print("未找到有效的 relevance.json 文件")
    average=0.00
    if reflection_ratios:
        avg_reflection = sum(reflection_ratios) / len(reflection_ratios)
        avg_reflection = 1 - avg_reflection
        average = (avg_reflection+avg_relevance) / 2
        print(f"\n处理了 {reflection_count} 个 reflection.json 文件")
        print(f"平均 'Correct' 比例：{avg_reflection:.2%}")
        print(f"relevance 和 reflection 的平均值为:{average:.2%}")
    else:
        print("未找到有效的 reflection.json 文件")

# 使用示例
base_dir = r"C:\NeurIPS_Ben\output\output_PhysExperiment\multi_image\output_gpt4o"  # 替换为你的文件夹路径
analyze_relevance_and_reflection(base_dir)
