# Multipartite-Entanglement-Classification

## Overview
This repository implements a framework for continuous variable multipartite entanglement classification through the neural network. It includes functionalities for loading datasets, applying quantum data augmentation, and building deep learning models for tripartite and quadripartite quantum states.

## File Descriptions

### `main.py`
The main script obtains the entire workflow of the project, including data loading, data augmentation, model training, and evaluation.

#### Functions/Components
- **Dataset Loading**: Imports datasets using `TripartiteData`, `QuadripartiteData`, or `CatData` functions from `Dataset.py` (The datasets can be downloaded from [this link](https://drive.google.com/file/d/1E2m9AUKdfc0eIr-F72tgpVa0Pq9yPWsZ/view?usp=sharing)).
- **Quantum Data Augmentation**: Enhances training data with `TripartiteQDA` or `QuadripartiteQDA` from `QDA.py`.
- **Model Training**: Builds and trains the model using either `TripartiteModel` or `QuadripartiteModel` from `sharedCNN.py`.
- **Evaluation**: Tests the model on a test set and computes accuracy and a normalized confusion matrix.

---

### `Dataset.py`
Defines functions for loading datasets for tripartite states with finite stellar rank, quadripartite states with finite stellar rank, and tripartite cat quantum states with infinite stellar rank.

#### Key Functions
- `TripartiteData`: Loads and preprocesses tripartite datasets, optionally including Monte Carlo (MC) data.
- `QuadripartiteData`: Loads and preprocesses quadripartite datasets.
- `CatData`: Loads datasets for tripartite cat states.

---

### `QDA.py`
Implements quantum data augmentation techniques for tripartite and quadripartite datasets.

---

### `sharedCNN.py`
Defines shared neural network architectures for processing tripartite and quadripartite quantum states.
