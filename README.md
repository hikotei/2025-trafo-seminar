# Mathematical Perspectives on Transformers: Replications and Explorations 🔬

## 📌 Overview

This repository contains code and visualizations for empirically replicating the mathematical insights and results from:

**"A Mathematical Perspective on Transformers"**  
*Borjan Geshkovski, Cyril Letrouit, Yury Polyanskiy, Philippe Rigollet*  
[arXiv:2312.10794](https://arxiv.org/abs/2312.10794)

**Authors' Code:** The `trafo-rotf-main/` directory contains example code from the authors, which can also be found on [GitHub](https://github.com/borjanG/2023-transformers-rotf/)

The focus is on empirical reproductions of key theoretical behaviors such as clustering, metastability, and the Wasserstein flow interpretation of self-attention dynamics.

## 🎯 Project Goals

- Replicate selected theoretical results from the paper using numerical simulations.
- Visualize and analyze transformer dynamics as interacting particle systems.
- Understand long-time clustering behavior and its dependence on β, d, and n.
- Link observed behaviors to gradient flow theory and optimal transport geometry.

## 📐 Key Theoretical Results of the Paper

Theorem 4.1: Clustering for β=0  

Theorem 6.1: Clustering for all β when d ≥ 3  

Theorem 6.3 / 6.9: Exponential convergence and metastability  

Figure 3: Phase transition visualizations (t, β) → clustering likelihood


## 🔬 Empirical Replications

Clustering - Self Attention Particle System Dynamics

Histogram - Pairwise Inner Products across Layers  

Phase Diagram - Numerical reproduction of metastable zones

## 📁 Repository Structure

```
trafo_seminar/
├── GESHKOVSKI - a mathematical perspective on transformers.pdf 
├── notebooks/
│   ├── test.ipynb              # xxx
│   └── test.ipynb              # xxx
│
├── trafo-rotf-main/            # code from original authors
├── mini_gpt/                   # andrej kaparthy
├── plots/                      # plot outputs
├── .gitignore
└── README.md
```

## 📊 WikiText-130 v1 Dataset

The WikiText language modeling dataset is a collection of over 100 million tokens extracted from verified Good and Featured articles on Wikipedia. This dataset is available under the Creative Commons Attribution-ShareAlike License.

Key features of the WikiText dataset:
- Over 2 times larger than the preprocessed Penn Treebank (PTB)
- Retains original case, punctuation, and numbers (unlike PTB)
- Composed of full articles, making it ideal for models that can leverage long-term dependencies
- Features a larger vocabulary compared to PTB

The dataset is particularly valuable for:
- Training language models
- Testing long-term dependency learning
- Evaluating model performance on natural language tasks
- Benchmarking against published results

## ⚙️ Setup & Installation

...
