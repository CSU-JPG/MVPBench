import os
import json

def calculate_relevant_ratio(data):
    """Calculate the ratio of items with 'relevant' set to 'Yes'"""
    total = len(data)
    if total == 0:
        return 0.0
    yes_count = sum(1 for item in data if item.get("relevant") == "Yes")
    return yes_count / total

def calculate_correct_ratio(data):
    """Calculate the ratio of items with 'judgment' set to 'Correct'"""
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
                print(f"[{folder}] relevance.json: 'Yes' ratio = {ratio:.2%}")
            except Exception as e:
                print(f"[{folder}] Error processing relevance.json: {e}")
        else:
            print(f"Warning: relevance.json not found in {folder}")

        # --- reflection.json ---
        reflection_file = next((f for f in os.listdir(folder_path) if f.endswith('reflection.json')), None)
        if reflection_file:
            try:
                with open(os.path.join(folder_path, reflection_file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                ratio = calculate_correct_ratio(data)
                reflection_ratios.append(ratio)
                reflection_count += 1
                print(f"[{folder}] reflection.json: 'Correct' ratio = {ratio:.2%} (total: {len(data)} items)")
            except Exception as e:
                print(f"[{folder}] Error processing reflection.json: {e}")
        else:
            print(f"Warning: reflection.json not found in {folder}")

    # --- Summary output ---
    print("\n========== Statistics ==========")
    if relevance_ratios:
        avg_relevance = sum(relevance_ratios) / len(relevance_ratios)
        avg_relevance = (avg_relevance-0.8) * 5
        print(f"Processed {relevance_count} relevance.json files")
        print(f"Average 'Yes' ratio: {avg_relevance:.2%}")
    else:
        print("No valid relevance.json files found")
    average=0.00
    if reflection_ratios:
        avg_reflection = sum(reflection_ratios) / len(reflection_ratios)
        avg_reflection = 1 - avg_reflection
        average = (avg_reflection+avg_relevance) / 2
        print(f"\nProcessed {reflection_count} reflection.json files")
        print(f"Average 'Correct' ratio: {avg_reflection:.2%}")
        print(f"Average of relevance and reflection: {average:.2%}")
    else:
        print("No valid reflection.json files found")

# Usage example
base_dir = r"C:\NeurIPS_Ben\output\output_PhysExperiment\multi_image\output_gpt4o"  # Replace with your folder path
analyze_relevance_and_reflection(base_dir)
