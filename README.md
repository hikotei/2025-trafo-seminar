# Mathematical Perspectives on Transformers: Replications and Explorations 🔬

## 📌 Overview

This repository contains code and visualizations for empirically replicating the mathematical insights and results from:

**"A Mathematical Perspective on Transformers"**  
*Borjan Geshkovski, Cyril Letrouit, Yury Polyanskiy, Philippe Rigollet*  
[arXiv:2312.10794](https://arxiv.org/abs/2312.10794)

The focus is on empirical reproductions of key theoretical behaviors such as clustering, metastability, and the Wasserstein flow interpretation of self-attention dynamics.

## 🎯 Project Goals

- Replicate selected theoretical results from the paper using numerical simulations.
- Visualize and analyze transformer dynamics as interacting particle systems.
- Understand long-time clustering behavior and its dependence on β, d, and n.
- Link observed behaviors to gradient flow theory and optimal transport geometry.

---

🧠 Introduction to Transformers

🧩 Intuition Behind the Architecture : 
- self-attention
- positional encoding
- sequence modeling basics

🌊 Transformers as Particle Systems and Neural ODEs

Based on Part 1 of the paper

- Self-attention interpreted as a mean-field interacting particle system  
- Comparison with residual networks and neural ODEs  
- Difference from RNN-based neural ODEs (focus on dynamics vs. training)  
- Importance of spherical constraint via layer normalization

🔁 Optimal Transport and Flow Map Perspective

Based on Section 3 of the paper

- Transformer flow as a continuity equation on probability measures  
- Wasserstein gradient flow interpretation  
- Interaction energy functionals and their monotonicity  
- SA vs USA dynamics and surrogate models

## 📐 Key Theoretical Results

<!-- Only high-level summaries -->

- Theorem 4.1: Clustering for β=0  
- Theorem 6.1: Clustering for all β when d ≥ 3  
- Theorem 6.3 / 6.9: Exponential convergence and metastability  
- Figure 3: Phase transition visualizations (t, β) → clustering likelihood

---

## 🔬 Empirical Replications

- `rep_fig1.ipynb`: Histogram of pairwise inner products across layers  
- `3d_hypersphere.ipynb`: Geometry of embeddings on spheres  
- `SA_particle_system.ipynb`: Simulated attention dynamics  
- `phase_diagram.ipynb`: Numerical reproduction of metastable zones

---

## 📁 Repository Structure

This repository contains code and notebooks dedicated to exploring and replicating concepts from the research paper **"A mathematical perspective on Transformers"** by Borjan Geshkovski, Cyril Letrouit, Yury Polyanskiy, and Philippe Rigollet. It also includes related explorations into transformer architectures and their underlying mathematical principles.

## Core Goal 🎯

The primary objective of this project is to understand and replicate key findings and figures presented in the aforementioned paper. This involves implementing and experimenting with transformer models, analyzing their behavior through visualizations, and exploring related mathematical concepts such as rotations on a sphere and particle system interactions.

1. Clustering
2. Phase change from metastable clusters to ...

## Repository Structure 📁

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

## Dataset and Experiments 🔍

*   **Paper Replication:** The primary notebooks for replicating Figure 1 of the Geshkovski et al. paper are `2025_05_28 rep_fig1_v2.ipynb` and `2025_04_29 rep_fig1.ipynb`. These notebooks load data, configure Albert-like models, and generate visualizations of attention mechanisms or particle interactions.

WikiText-130 v1 Dataset

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

*   **Self-Attention as Particle Systems:** `2025_04_29 1st SA particle system.ipynb` explores the concept of self-attention mechanisms behaving like interacting particle systems.
*   **Hypersphere Visualization:** `2025_05_19 3d hypersphere.ipynb` is dedicated to visualizing and calculating properties of 3D hyperspheres, which may relate to the geometric interpretations of transformer embeddings.
*   **`trafo-rotf-main/` Directory:** This directory contains example code provided by the authors of the Geshkovski et al. paper. It includes Python scripts and notebooks for:
    *   Generating phase diagrams (`phase-diagrams_own.py`, `phase-diagrams.py`).
    *   Analyzing curves (`curve.ipynb`).
    *   Experimenting with ALBERT models (`albert_own.py`, `albert.ipynb`).
    *   Working with spherical projections (`sphere.py`).
    *   It also includes fascinating animations in the `movies/` subdirectory.
*   **`mini_gpt/` Directory:** This section focuses on building and understanding a smaller version of a GPT model from scratch, including bigram models (`bigram.py`, `bigram v2.py`) and a more complete GPT implementation (`gpt.py`).
*   **`plots/` Directory:** This directory stores various visualizations generated by the notebooks, including the `3d sphere.png` and timestamped directories containing plots from different experimental runs (e.g., varying the number of hidden layers `hl`).

## How to Use / Prerequisites 📋

...

## Paper Reference 📚

The work in this repository is largely based on and aims to replicate findings from:

Geshkovski, B., Letrouit, C., Polyanskiy, Y., & Rigollet, P. (2023). *A mathematical perspective on Transformers*. arXiv preprint arXiv:2312.10794.
*   **arXiv Link:** [https://arxiv.org/abs/2312.10794](https://arxiv.org/abs/2312.10794)
*   **Authors' Code:** The `trafo-rotf-main/` directory contains example code from the authors, which can also be found on [GitHub](https://github.com/borjanG/2023-transformers-rotf) (if this is the correct link, otherwise adjust).

## Visualizations 📊

Key visualizations, including plots from Figure 1 replications, 3D sphere plots, and animations from the `trafo-rotf-main/movies` directory, are available within the `plots/` and `trafo-rotf-main/movies/` folders.