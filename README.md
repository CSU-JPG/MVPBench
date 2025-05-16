# MVPBench

## Seeing is Not Reasoning: MVPBench for Graph-based Evaluation of Multi-path Visual Physical CoT

![logo](https://github.com/naver-ai/TextAtlas5M/raw/main/assets/logo.png)

📘 [Homepage](https://github.com/naver-ai/TextAtlas5M) | 🏆 Leaderboard | 🧑‍🔬 [TextAtlas5M](https://huggingface.co/datasets/naver-ai/TextAtlas5M) | 🧑‍🔬 [TextAtlasEval](https://huggingface.co/datasets/naver-ai/TextAtlasEval) | 📄 [TextAtlas arXiv](https://arxiv.org/abs/2402.12345)

This repo contains the evaluation code for the paper:  
["TextAtlas5M: A Large-scale Dataset for Dense Text Image Generation"](https://arxiv.org/abs/2402.12345)

---

## 🔔 Updates

- **[2025-2-18]**: Our evaluation code is now available! 🌟  
- **[2025-2-13]**: Released TextAtlasEval & TextAtlas5M version 1.0 🔥

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

TextAtlas was meticulously designed to challenge and evaluate text-rich image generation.  
For more detailed information and accessing our dataset, please refer to our Huggingface page:

- 🧑‍🔬 [TextAtlas5M](https://huggingface.co/datasets/naver-ai/TextAtlas5M)
- 🧑‍🔬 [TextAtlasEval](https://huggingface.co/datasets/naver-ai/TextAtlasEval)

---

## 🧪 Evaluation

Please refer to our evaluation folders for detailed information on evaluating with the TextAtlasEval benchmark:

- 🔍 [TextAtlas Evaluation](evaluation/README.md)

---

## 🗂 Data Format

The TextAtlas annotation documentation is available on Huggingface:

- **Main version**: Contains image paths and pre-integrated prompts, making it suitable for direct training or evaluation.
- **Meta data**: Includes all the data from the main version, along with additional intermediate results such as bounding boxes (`bbox`), font size, and other related information.

### 📌 Example

```json
{
  "image_path": "0000089b-f1ce-41cf-9cd8-688856822244.png",
  "annotation": "In an opulent boutique, a sleek white digital display contrasts sharply with meticulously arranged merchandise and luxurious decor, creating a striking visual focal point. digital display with the text : 'Amidst the opulent ambiance of the upscale boutique, a sleek white digital display stands out as a striking contrast to the meticulously arranged merchandise and sumptuous luxury decor'"
}
```

### 📝 Field Description

- **`image_path`** (`str`): The image file name  
- **`annotation`** (`str`): A richly descriptive caption that includes embedded text

### 📊 Metadata Format

In addition to the fields above, the meta version includes:

- Bounding boxes (`bbox`)
- Font sizes
- Layout meta data

For more information, see the full [Detailed Annotation Guide](detialed_annotation/README.md)

---

## 📄 Citation

If you find our work useful, please cite us:

```bibtex
@article{wang2025textatlas5m,
  title={TextAtlas5M: A Large-scale Dataset for Dense Text Image Generation},
  author={Wang, Alex Jinpeng and Mao, Dongxing and Zhang, Jiawei and Han, Weiming and Dong, Zhuobai and Li, Linjie and Lin, Yiqi and Yang, Zhengyuan and Qin, Libo and Zhang, Fuwei and others},
  journal={arXiv preprint arXiv:2502.07870},
  year={2025}
}
```

---

## 📬 Contact

For questions, suggestions or issues, feel free to open an [issue](https://github.com/naver-ai/TextAtlas5M/issues) on GitHub.

---


