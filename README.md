# MVPBench

## Seeing is Not Reasoning: MVPBench for Graph-based Evaluation of Multi-path Visual Physical CoT

![logo](https://github.com/naver-ai/TextAtlas5M/raw/main/assets/logo.png)

📘 [Homepage](https://csu-jpg.github.io/MVPBench/) | 🏆 Leaderboard | 🧑‍🔬 [Dataset](https://huggingface.co/datasets/CSU-JPG/MVPBench) | 🧑‍🔬 [MVPBenchEval](https://huggingface.co/datasets/naver-ai/TextAtlasEval) | 📄 [MVPBench arXiv](https://arxiv.org/abs/2402.12345)

This repo contains the evaluation code for the paper:  
["TextAtlas5M: A Large-scale Dataset for Dense Text Image Generation"](https://arxiv.org/abs/2402.12345)

---

## 🔔 Updates

- **[2025-5-22]**: Our evaluation code is now available! 🌟  
- **[2025-5-22]**: Released MVPBench version 1.0 🔥

---

## 📑 Table of Contents

- [Setup](#️-setup)
- [Accessing Datasets](#-accessing-datasets)
- [Evaluation](#-evaluation)
- [Data Format](#-data-format)
- [Citation](#-citation)
- [Contact](#-contact)

---

## ⚙️ Setup

To set up the environment for evaluation:

```bash
conda create -n TextAtlasEval python=3.9
conda activate TextAtlasEval
pip install -r requirements.txt
```

---

## 📂 Accessing Datasets

MVPBench was meticulously designed to challenge and evaluate visual physical reasoning through the lens of visual chain-of-thought (CoT) 
For more detailed information and accessing our dataset, please refer to our Huggingface page:

- 🧑‍🔬 [MVPBench](https://huggingface.co/datasets/CSU-JPG/MVPBench)

---

## 🧪 Evaluation

Please refer to our evaluation folders for detailed information on evaluating with the TextAtlasEval benchmark:

- 🔍 [TextAtlas Evaluation](evaluation/README.md)

---

## 🗂 Data Format

- The dataset is provided in jsonl format and contains the following attributes:

```json
[
    {
        "id": [string] Problem ID, e.g.,"PhyTest_0001",
        "query": [string] The question text,
        "original_scene": {
            "id": [string] original scene ID,
            "path": [string] The original image path of the question,
            "description": [string] Description of the original image content,
        },
        "key_step_1": {
            "id": [string] The first reasoning step ID,
            "path": [string] The first reasoning step image path of the question,
            "conclusion": [string] The textual content of the first reasoning step,
        },
        "key_step_2": {
            "id": [string] The second reasoning step ID,
            "path": [string] The second reasoning step image path of the question,
            "conclusion": [string] The textual content of the second reasoning step,
        },
		...
		"key_step_n": {
            "id": [string] The nth reasoning step ID,
            "path": [string] The nth reasoning step image path of the question,
            "conclusion": [string] The textual content of the nth reasoning step,
        },
        "final_scene": {
            "id": [string] The final answer ID,
            "path": The final answer image path of the question,
            "annotation": The textual content of the final answer,
        },
        "subject": [string] The subject of subset of data(Physics problems,Physics experiments, Spatial relations and Dynamic prediction,
        "possible_chains": [string] all possible reasoning paths, e.g., [
            [ "key_step_1","key_step_2"]
        ],
        "json_path": [string] The path of the json file,
    }
]
```



---

## 📄 Citation

If you find our work useful, please cite us:

```bibtex
@article{dong2025mvpbench,
  title={Seeing is Not Reasoning: MVPBench for Graph-based Evaluation of Multi-path Visual Physical CoT},
  author={Dong, Zhuobai and Yi, Junchao and Zheng, Ziyuan and Han, haochen and Zheng, Xiangxi and Wang, Alex Jinpeng and Liu, Fangming and Li, Linjie and others},
  year={2025}
}
```

---

## 📬 Contact

For questions, suggestions or issues, feel free to open an [issue](https://github.com/CSU-JPG/MVPBench/issues) on GitHub.

---

