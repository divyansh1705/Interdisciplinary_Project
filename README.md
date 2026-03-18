<div align="center">

<h2 align="center">
Learning to Tell Apart: Weakly Supervised Video Anomaly Detection via Disentangled Semantic Alignment (AAAI 2026)
</h2>

<p align="center">
  <a href="https://arxiv.org/abs/2511.10334">
    <img src="https://img.shields.io/badge/Paper-arXiv%3A2511.10334-b31b1b.svg" alt="Paper">
  </a>
  <a href="https://drive.google.com/drive/folders/1PqvaNm_s-fOOrnJRqrG50zV2R2UqRwHK?usp=sharing">
    <img src="https://img.shields.io/badge/Weights-Google%20Drive-blue.svg" alt="Weights">
  </a>
</p>

</div>

> [!NOTE]
> Our paper has been accepted by **AAAI 2026**! This is the official PyTorch implementation for **DSANet**.

## 📖 Abstract

Recent advancements in weakly-supervised video anomaly detection have achieved remarkable performance by applying the multiple instance learning paradigm based on multimodal foundation models such as CLIP to highlight anomalous instances and classify categories.
However, their objectives may tend to detect the most salient response segments, while neglecting to mine diverse normal patterns separated from anomalies, and are prone to category confusion due to similar appearance, leading to unsatisfactory fine-grained classification results.
Therefore, we propose a novel Disentangled Semantic Alignment Network (DSANet) to explicitly separate abnormal and normal features from coarse-grained and fine-grained aspects, enhancing the distinguishability.
Specifically, at the coarse-grained level, we introduce a self-guided normality modeling branch that reconstructs input video features under the guidance of learned normal prototypes, encouraging the model to exploit normality cues inherent in the video, thereby improving the temporal separation of normal patterns and anomalous events.
At the fine-grained level, we present a decoupled contrastive semantic alignment mechanism, which first temporally decomposes each video into event-centric and background-centric components using frame-level anomaly scores and then applies visual-language contrastive learning to enhance class-discriminative representations.
Comprehensive experiments on two standard benchmarks, namely XD-Violence and UCF-Crime, demonstrate that DSANet outperforms existing state-of-the-art methods.

<div align="center">
  <img src="assets/framework.png" width="100%" alt="Framework of DSANet"/>
</div>

## 🛠️ Environment

The code is developed and tested on a single **NVIDIA RTX 4090** GPU.

To set up the environment, please run:

```bash
conda env create -f environment.yml
conda activate dsanet
```

## 📂 Data Preparation

### Features
We utilize the pre-extracted CLIP features provided by **VadCLIP**. Please download the features for UCF-Crime and XD-Violence datasets from their official repository:

- **VadCLIP Repository:** [GitHub - nwpu-zxr/VadCLIP](https://github.com/nwpu-zxr/VadCLIP)

### Directory Structure
After downloading, please organize the features and update the paths in the csv files located in the `list/` directory:

1. Open `list/ucf_CLIP_rgb.csv` (or similar .csv files) and replace the feature paths with your local directory paths.
2. Ensure the annotations match the feature files.

## 🚀 Training

To train the model on a specific dataset, ensure you have updated the feature paths in the csv files first.

**UCF-Crime:**
```bash
python ucf_train.py
```

**XD-Violence:**
```bash
python xd_train.py
```
## 🧪 Testing

We provide the pre-trained model weights to reproduce our results.

1. **Download Weights:** Download the `.pth` files from [Google Drive](https://drive.google.com/drive/folders/1PqvaNm_s-fOOrnJRqrG50zV2R2UqRwHK?usp=sharing).
2. **Placement:** Place the downloaded weights into the `model/` folder.

**Evaluate on UCF-Crime:**
```bash
python ucf_test.py
```

**Evaluate on XD-Violence:**
```bash
python xd_test.py
```

## 📊 Results

| Method | UCF-Crime (AUC/%) | XD-Violence (AP/%) |
| :--- | :---: | :---: |
| **DSANet (Ours)** | **89.44** | **86.95** |

> Note: Please refer to our paper for the detailed performance comparison.

## 🙏 Acknowledgement

We thank the authors of the following repositories for their valuable work and codebases:

* [VadCLIP](https://github.com/nwpu-zxr/VadCLIP)
* [INP-Former](https://github.com/luow23/INP-Former)
* [AA-CLIP](https://github.com/Mwxinnn/AA-CLIP)

## 📜 Citation

If you find this repository useful for your research, please consider citing our paper:

```bibtex
@misc{yin2025learningtellapartweakly,
      title={Learning to Tell Apart: Weakly Supervised Video Anomaly Detection via Disentangled Semantic Alignment}, 
      author={Wenti Yin and Huaxin Zhang and Xiang Wang and Yuqing Lu and Yicheng Zhang and Bingquan Gong and Jialong Zuo and Li Yu and Changxin Gao and Nong Sang},
      year={2025},
      eprint={2511.10334},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.10334}, 
}
