import os
import json
from typing import List, Tuple
import networkx as nx
from math import exp

def build_reasoning_graph(possible_chains: List[List[str]]) -> nx.DiGraph:
    G = nx.DiGraph()
    for chain in possible_chains:
        for i in range(len(chain) - 1):
            G.add_edge(chain[i], chain[i + 1])
        G.add_edge(chain[-1], "final_scene")
    return G

def longest_common_subsequence(a: List[str], b: List[str]) -> List[str]:
    dp = [[[] for _ in range(len(b) + 1)] for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + [a[i]]
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j], key=len)
    return dp[-1][-1]

def compute_path_score(model_path: List[str], graph: nx.DiGraph, possible_chains: List[List[str]]) -> Tuple[float, float]:
    model_edges = [(model_path[i], model_path[i+1]) for i in range(len(model_path) - 1)]
    model_edges.append((model_path[-1], "final_scene"))

    correct_edges = sum(1 for edge in model_edges if graph.has_edge(*edge))
    precision = correct_edges / len(model_edges)

    model_with_final = model_path + ["final_scene"]
    max_lcs = max(len(longest_common_subsequence(chain + ["final_scene"], model_with_final)) for chain in possible_chains)
    recall = max_lcs / max(len(chain) + 1 for chain in possible_chains)

    return precision, recall

def compute_overall_metrics(model_paths: List[List[str]], possible_chains: List[List[str]], graph: nx.DiGraph, alpha=1.5, beta: float = 1.0) -> Tuple[float, float, float]:
    if not model_paths:
        return 0.0, 0.0, 0.0

    precisions, recalls = [], []

    for path in model_paths:
        p, r = compute_path_score(path, graph, possible_chains)
        precisions.append(p)
        recalls.append(r)

    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)

    # Penalty strategy
    n_model = len(model_paths)
    n_gt = len(possible_chains)

    precision_penalty = min(n_model, n_gt) / n_gt
    recall_penalty = exp(-alpha * abs(n_model / n_gt - 1))

    # Only apply penalty when model generates more paths than GT
    if n_model > n_gt:
        precision_penalty = n_gt / n_model
        recall_penalty = 1 / (1 + beta * (n_model / n_gt - 1))

    final_precision = avg_precision * precision_penalty
    final_recall = avg_recall * recall_penalty
    final_f1 = 2 * final_precision * final_recall / (final_precision + final_recall) if (final_precision + final_recall) else 0.0

    return round(final_precision, 4), round(final_recall, 4), round(final_f1, 4)

def batch_evaluate(possible_dir: str, solu_dir: str, alpha=1.5):
    results = []
    total_precision = total_recall = total_f1 = 0.0
    count = 0

    for subfolder in os.listdir(possible_dir):
        possible_subfolder_path = os.path.join(possible_dir, subfolder)
        solu_subfolder_path = os.path.join(solu_dir, subfolder)
        if not os.path.isdir(possible_subfolder_path) or not os.path.isdir(solu_subfolder_path):
            continue

        # Find possible json in the possible paths folder
        possible_json_file = next((
            f for f in os.listdir(possible_subfolder_path)
            if f.endswith('.json') 
        ), None)

        # Modify suffix here
        solu_json_file = f"{subfolder}_solu_claude.json"

        possible_json_path = os.path.join(possible_subfolder_path, possible_json_file) if possible_json_file else None
        solu_json_path = os.path.join(solu_subfolder_path, solu_json_file)

        if not possible_json_path or not os.path.exists(possible_json_path) or not os.path.exists(solu_json_path):
            print(f"Skipping {subfolder}, missing json files")
            continue

        try:
            with open(possible_json_path, 'r', encoding='utf-8') as f:
                possible_content = f.read().strip()
                if not possible_content:
                    raise ValueError("File is empty")
                possible_data = json.loads(possible_content)
        except Exception as e:
            print(f"Skipping {subfolder}, error reading possible_json: {e}")
            precision = recall = f1 = 0.0
            results.append({
                "folder": subfolder,
                "num_model_paths": 0,
                "num_possible_paths": 0,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })
            count += 1
            continue

        try:
            with open(solu_json_path, 'r', encoding='utf-8') as f:
                solu_content = f.read().strip()
                if not solu_content:
                    raise ValueError("File is empty")
                solu_data = json.loads(solu_content)
        except Exception as e:
            print(f"Skipping {subfolder}, error reading solu_json: {e}")
            precision = recall = f1 = 0.0
            results.append({
                "folder": subfolder,
                "num_model_paths": 0,
                "num_possible_paths": len(possible_data[0].get("possible_chains", [])) if possible_data else 0,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })
            count += 1
            continue

        possible_chains = possible_data[0].get("possible_chains", [])
        
        # Check if solu_data is a valid key_step style path
        def is_valid_key_step_path(path_list: List[List[str]]) -> bool:
            return all(
                isinstance(step, str) and step.startswith("key_step")
                for path in path_list
                for step in path
            )

        if not isinstance(possible_chains, list) or not isinstance(solu_data, list):
            print(f"Format error, skipping {subfolder}")
            precision = recall = f1 = 0.0
        elif not is_valid_key_step_path(solu_data):
            print(f"{subfolder} contains non-key_step paths, metrics set to 0")
            precision = recall = f1 = 0.0
        else:
            G = build_reasoning_graph(possible_chains)
            precision, recall, f1 = compute_overall_metrics(solu_data, possible_chains, G, alpha=alpha)

        results.append({
            "folder": subfolder,
            "num_model_paths": len(solu_data),
            "num_possible_paths": len(possible_chains),
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

        total_precision += precision
        total_recall += recall
        total_f1 += f1
        count += 1

    avg_precision = round(total_precision / count, 4) if count else 0.0
    avg_recall = round(total_recall / count, 4) if count else 0.0
    avg_f1 = round(total_f1 / count, 4) if count else 0.0

    print("=== Overall Average Metrics ===")
    print(f"Average Precision: {avg_precision}")
    print(f"Average Recall: {avg_recall}")
    print(f"Average F1: {avg_f1}")

    return results, avg_precision, avg_recall, avg_f1


# Example usage
if __name__ == "__main__":
    possible_dir = r"C:\NeurIPS_Ben\data\data_PhySpatial"
    solu_dir = r"C:\NeurIPS_Ben\output\output_PhySpatial\solu\claude"
    results, avg_p, avg_r, avg_f1 = batch_evaluate(possible_dir, solu_dir)

    # for r in results:
    #     print(r)

