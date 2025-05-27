import os
import json

def calculate_judgment_ratio(json_data, target_judgment="Matched"):
    """Calculate the ratio of judgments matching target_judgment in recall.json"""
    if not json_data:
        return 0.0
    matched_count = sum(1 for item in json_data if item.get("judgment") == target_judgment)
    return matched_count / len(json_data)

def calculate_match_ratio_and_final_score(data):
    """Calculate the ratio of 'Match' judgments and final_score in precision.json"""
    logical_inferences = [item for item in data if item.get("step_type") in ("logical inference", "image description")]
    # logical_inferences = [item for item in data if item.get("step_type") in ("logical inference")]
    total = len(logical_inferences)
    
    if total == 0:
        return 0.0, 0
    
    match_count = sum(1 for item in logical_inferences if item.get("judgment") == "Match")
    final_score = 1 if logical_inferences[-1].get("judgment") == "Match" else 0
    
    return match_count / total, final_score

def analyze_combined_metrics(base_dir, a=0.7):
    recall_ratios = []
    precision_ratios = []
    final_scores = []

    for folder in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # Find recall.json
        recall_file = next((f for f in os.listdir(folder_path) if f.endswith("recall.json")), None)
        if recall_file:
            try:
                with open(os.path.join(folder_path, recall_file), 'r', encoding='utf-8') as f:
                    recall_data = json.load(f)
                recall_ratio = calculate_judgment_ratio(recall_data, "Matched")
                recall_ratios.append(recall_ratio)
                print(f"[{folder}] Recall (Matched ratio) = {recall_ratio:.2%}")
            except Exception as e:
                print(f"Error processing {folder}/recall.json: {e}")
        else:
            print(f"Warning: recall.json not found in {folder}")

        # Find precision.json
        precision_file = next((f for f in os.listdir(folder_path) if f.endswith("presicion.json")), None)
        if precision_file:
            try:
                with open(os.path.join(folder_path, precision_file), 'r', encoding='utf-8') as f:
                    precision_data = json.load(f)
                precision_ratio, final_score = calculate_match_ratio_and_final_score(precision_data)
                precision_ratios.append(precision_ratio)
                final_scores.append(final_score)
                print(f"[{folder}] Precision (Match ratio) = {precision_ratio:.2%}, Final Score = {final_score}")
            except Exception as e:
                print(f"Error processing {folder}/presicion.json: {e}")
        else:
            print(f"Warning: presicion.json not found in {folder}")

    if recall_ratios and precision_ratios:
        avg_recall = sum(recall_ratios) / len(recall_ratios)
        avg_precision = sum(precision_ratios) / len(precision_ratios)
        avg_final_score = sum(final_scores) / len(final_scores) if final_scores else 0.0
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0

        # New combined score
        Combined_pre = avg_precision * a + avg_final_score * (1 - a)

        combined_f1_score = 2 * (Combined_pre * avg_recall) / (Combined_pre + avg_recall) if (Combined_pre + avg_recall) > 0 else 0.0

        print("\n========= Summary =========")
        print(f"Average Recall = {avg_recall:.2%}")
        print(f"Average Precision = {avg_precision:.2%}")
        print(f"Average Final Score = {avg_final_score:.2f}")
        print(f"F1-score = {f1_score:.4f}")
        print(f"Combined F1-Score (a = {a}) = {combined_f1_score:.4f}")
    else:
        print("Insufficient valid recall or precision data, cannot calculate combined score")

# Usage example
base_dir = r"C:\NeurIPS_Ben\output\output_PhysExperiment\multi_image\output_o3" # Replace with your path
a = 0.7  # Set weight factor, can be adjusted as needed
analyze_combined_metrics(base_dir, a)
