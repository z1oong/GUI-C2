<h1 align="center">
  GUI-C<sup>2</sup>: Coarse-to-Fine GUI Grounding via Difficulty-Aware Reinforcement Learning
</h1>
<p align="center">
    <a href='https://arxiv.org/pdf/'>
      <img src='https://img.shields.io/badge/arXiv-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='arXiv PDF'>
         </a>
    <a href='https://z1oong.github.io/GUI-C2/'>
      <img src='https://img.shields.io/badge/Project-page-purple?style=flat&logo=web&logoColor=gary' alt='Project page'>
         </a>

More details can be found in Project page.

## 📰 News

- [2026.5] 🤩 Our training dataset [**GUI-C2-4K**](https://huggingface.co/datasets/z1oong/GUI-C2-4K) released on HuggingFace.
- [2026.5] 🤩 Code for difficulty scoring released.

## 📚 Training Data

We open-source our 3B model training dataset first, as we believe it offers substantial value to future research in this field. If you find our dataset, difficulty design, and score calculation helpful to your work, please consider citing our paper.

Our dataset is sourced from:

- [OS-Atlas](https://github.com/OS-Copilot/OS-Atlas)

- [SeeClick](https://github.com/njucckevin/SeeClick)

- [UI-Bert](https://github.com/google-research-datasets/uibert)


## 😵‍💫 Difficulty Scoring

First, you need to conduct an 8-click test (using 8 rollouts as an example) on the data sources and base model you intend to use. Retain the raw output coordinates of the eight clicks, and pre-filter out samples where all 8 clicks are correct or incorrect.

```bash

cd /share/home/junlong_li/GUI-C2-main

python build_train_set_diff.py --input filter_rawoutput.json --output train_set_diff.json

```

