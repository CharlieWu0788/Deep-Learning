# COMP395 — Deep Learning

Course repository for **COMP395: Deep Learning** at Occidental College.

This repository serves as a structured record of coursework completed throughout the semester, including assignments, labs, lecture notebooks, experiment results, and supporting development tools.

---

## 📁 Project Structure

```text
COMP395/
├── .github
│   └── workflows
├── .vscode
├── Assignments
│   ├── Assignment 1-Differentiation through multiple representations
│   │   └── Assignment 1.ipynb
│   ├── Assignment 2-Partial Derivatives and Gradients
│   └── Assignment 3-Gradient Descent from Scratch
│       └── Assignment 3.ipynb
├── dist
├── environment
│   ├── wheels
│   │   └── README.md
│   ├── README.md
│   └── VALIDATION.md
├── Justice and Equity
│   └── Compas Analysis
│       ├── .gitignore
│       ├── COMPAS Analysis - Student.ipynb
│       ├── Compas Analysis.ipynb
│       ├── Cox with interaction term and independent variables.ipynb
│       └── truth_tables.py
├── Labs
│   ├── Lab 1-Introduction to PyTorch and GPU Computing
│   │   ├── Lab1_files
│   │   │   ├── Lab1_5_0.png
│   │   │   └── Lab1_16_0.png
│   │   ├── bonus_speedup_threshold.png
│   │   ├── cpu_vs_gpu_benchmark.png
│   │   ├── Lab1.pdf
│   │   └── wu_charlie_lab1.ipynb
│   ├── Lab 2-Breast Cancer
│   │   ├── .github
│   │   │   └── workflows
│   │   ├── report
│   │   │   ├── README.md
│   │   │   ├── report.pdf
│   │   │   └── training_results.png
│   │   ├── sample_report
│   │   │   ├── accuracy_comparison.png
│   │   │   ├── make_figures.py
│   │   │   ├── sample_report.pdf
│   │   │   └── training_loss.png
│   │   ├── .gitignore
│   │   ├── binary_classification.py
│   │   ├── binary_classification_pytorch.py
│   │   ├── comparison.py
│   │   ├── overleaf-tutorial.pdf
│   │   ├── README.md
│   │   ├── test_binary_classification.py
│   │   ├── test_comparison.py
│   │   ├── training_results.png
│   │   └── training_results_pytorch.png
│   └── Lab 3-MLPs on Sonar_Design, Experiment, Report
│       ├── .github
│       │   └── workflows
│       ├── report
│       │   ├── report.pdf
│       │   ├── wandb_train_loss.png
│       │   └── wandb_val_loss.png
│       ├── wandb
│       ├── .gitignore
│       ├── comparison.py
│       ├── Documentation Answers.md
│       ├── linear_baseline_wandb.py
│       ├── overleaf-tutorial.pdf
│       ├── read-if-m2l2.pdf
│       ├── README.md
│       ├── sonar_data.py
│       ├── sonar_mlp.py
│       ├── test_comparison.py
│       ├── test_sonar_mlp.py
│       └── training_results.png
├── Lecture
│   ├── Lecture 01-Matrix Operations in Pytorch
│   │   └── Matrix Operations.ipynb
│   ├── Lecture 02-Partial Derivatives and Gradients
│   │   └── Partial Derivatives and Gradients.ipynb
│   ├── Lecture 03-Gradient Descent from Scratch
│   │   └── Gradient Descent.ipynb
│   ├── Lecture 04-Convolutional Neural Network
│   │   ├── Day 1-Convolution
│   │   │   ├── Cameraman.png
│   │   │   └── Convolution.ipynb
│   │   └── Day 2-CNN Architecture
│   │       ├── Architecture.ipynb
│   │       ├── astronaut.png
│   │       └── cameraman.png
│   ├── Lecture 05-Word Embeddings_Meaning in Numbers
│   │   └── Lecture 5.ipynb
│   ├── Lecture 06-Recurrent Neural Network
│   │   ├── Day 1-Companion
│   │   │   └── Companion.ipynb
│   │   ├── Day 2-Sentiment
│   │   │   ├── data
│   │   │   └── Sentiment.ipynb
│   │   └── Sequence2Sequence Translation
│   │       └── seq2seq_translation-DL-F26
│   │           ├── data
│   │           │   └── spa-eng
│   │           ├── wandb
│   │           ├── .gitignore
│   │           ├── README.md
│   │           ├── seq2seq_translation_wandb.py
│   │           ├── translate.py
│   │           └── translation_results.png
│   ├── Lecture 07-Transformers
│   │   ├── attentionisallyouneed.pdf
│   │   ├── Day 1.ipynb
│   │   └── Day 2.ipynb
│   ├── Lecture 08-Bidirectional Encoder Representations from Transformers
│   │   └── BERT.ipynb
│   ├── Lecture 09-Vision Language Models
│   │   ├── aespa-kiss-n-tell.jpg
│   │   ├── dog-frisbee.jpeg
│   │   └── VLM.ipynb
│   └── Lecture 10-Large Language Models
│       └── LLM.ipynb
├── Tools
│   ├── check_environment.py
│   └── Project_tree.py
├── wandb
├── .gitignore
├── LICENSE
└── README.md
```

> The project tree can be generated automatically using `Tools/Project_tree.py`.

Runtime-generated directories such as `.pytest_cache/`, local Python environments, and Weights & Biases run files are intentionally excluded from the repository overview.

---

## 📚 Contents

### Assignments

Course assignments covering the mathematical foundations used in deep learning.

- **Assignment 1 — Differentiation through Multiple Representations**
  - Derivatives represented analytically, numerically, and computationally
  - PyTorch autograd fundamentals

- **Assignment 2 — Partial Derivatives and Gradients**
  - Multivariable differentiation
  - Partial derivatives
  - Gradient vectors

- **Assignment 3 — Gradient Descent from Scratch**
  - Gradient-based optimization
  - Iterative parameter updates
  - Convergence behavior

### Labs

Hands-on implementations and experiments using Python and PyTorch.

#### Lab 1 — Introduction to PyTorch and GPU Computing

- PyTorch tensor fundamentals
- CPU vs. GPU computation
- CUDA acceleration
- Matrix multiplication benchmarks
- GPU speedup analysis
- Performance visualization

#### Lab 2 — Breast Cancer Binary Classification

- Binary classification from scratch
- Manual parameter and gradient management
- Gradient descent implementation
- PyTorch `nn.Module` implementation
- `nn.Linear` parameter management
- Loss and accuracy comparison
- Automated tests with `pytest`
- Training-result visualization
- Report generation and analysis

#### Lab 3 — MLPs on Sonar: Design, Experiment, Report

- Binary classification of sonar returns from mines and rocks
- Multilayer perceptrons built with `nn.Sequential`
- Controlled experiments comparing hidden-layer width, depth, activation functions, and optimizers
- ReLU and Tanh activations with Adam and SGD optimizers
- Comparison with a linear baseline and a Random Forest on the same data split
- Weights & Biases tracking of configurations, losses, accuracy, and model artifacts
- Automated tests for model structure, nonlinearity, training, and performance
- Experiment report with learning curves and analysis of generalization

### Lectures

Lecture notebooks and examples organized by topic.

- **Lecture 1 — Matrix Operations in PyTorch**
  - Creating vectors and matrices as PyTorch tensors and inspecting their shapes
  - Random tensor generation with `torch.rand` and `torch.randn`
  - Element-wise addition, subtraction, and multiplication
  - Matrix multiplication with `@` and `matmul`, including input and output dimensions

- **Lecture 2 — Partial Derivatives and Gradients**
  - Building two-variable function grids with NumPy `meshgrid`
  - Visualizing the surface of `f(x, y) = x² + y²` with Matplotlib
  - Relating partial derivatives to slopes in each coordinate direction at a selected point

- **Lecture 3 — Gradient Descent**
  - Defining a quadratic objective and computing its gradient explicitly with NumPy
  - Implementing gradient descent updates from scratch
  - Learning-rate settings, iteration limits, and gradient-norm stopping criteria
  - Visualizing the optimization path on a contour plot and checking the final minimum

- **Lecture 4 — Convolutional Neural Networks**
  - Image tensors, color channels, batch dimensions, and feature-map shapes
  - Convolution and max pooling implemented from scratch
  - Edge detection and filter experiments using Sobel kernels and `nn.Conv2d`
  - CNN construction with `nn.Module`, MNIST classification, and analysis of learned filters and prediction errors

- **Lecture 5 — Word Embeddings: Meaning in Numbers**
  - Pretrained GloVe vectors, cosine similarity, and nearest-neighbor exploration
  - Word analogies through vector arithmetic
  - PCA visualization and reflection on bias in word embeddings
  - Loading pretrained vectors into PyTorch `nn.Embedding`

- **Lecture 6 — Recurrent Neural Networks**
  - Hidden-state dynamics and visualization of vanishing and exploding gradients
  - Text preprocessing, GloVe embeddings, and RNN/LSTM sentiment classification on IMDB
  - Gradient measurements by word position and analysis of custom review predictions
  - Extensions comparing text truncation, gradient clipping, and GRU models
  - English-to-Spanish encoder-decoder translation with teacher forcing, BLEU evaluation, W&B tracking, and interactive inference

- **Lecture 7 — Transformers**
  - Scaled dot-product attention, query/key/value projections, and multi-head tensor shapes
  - Causal masking and sinusoidal positional encoding
  - Self-attention implemented as an `nn.Module`
  - Position-wise feed-forward networks and layer normalization
  - KV caching for autoregressive inference, including timing and memory-use comparisons

- **Lecture 8 — BERT: Pretraining and Fine-Tuning**
  - WordPiece tokenization, special tokens, attention masks, and review truncation
  - Fine-tuning pretrained BERT-Tiny for IMDB sentiment classification
  - Comparison with RNN and LSTM results, including accuracy and recorded training times
  - Custom review tests involving negation, sarcasm, and changes in sentiment

- **Lecture 9 — Vision Language Models**
  - Image patch extraction, linear patch embeddings, and patch-similarity visualization
  - Pretrained DINOv2 vision transformers with register tokens and visualization of image self-attention
  - Comparisons of attention heads and layers using the dog-frisbee image and a custom group photo
  - BLIP image captioning and word-specific cross-attention over image patches
  - Raw attention versus Grad-CAM, including caption-generation failures and the limits of attention as an explanation

- **Lecture 10 — Large Language Models**
  - Loading pretrained GPT-2 and exploring byte-level BPE tokenization
  - Greedy text generation, repetitive outputs, and few-shot question-and-answer prompting
  - Next-token logits, softmax probabilities, and temperature sampling experiments
  - Top-k and nucleus sampling, with visualization of probability distributions and nucleus sizes
  - Reflections on generation tradeoffs, model knowledge versus behavior, and the roles of SFT and RLHF

### Tools

Utilities used to inspect and maintain the repository.

- `Tools/Project_tree.py` — generates formatted directory trees at multiple levels of detail
- `Tools/setup_environment.ps1` — rebuilds the isolated Windows Python 3.14 environment from locked dependencies
- `Tools/check_environment.py` — checks package versions, coursework libraries, and computation on each visible GPU

---

## 🛠️ Development Environment

Course work is developed primarily with:

- Python
- PyTorch
- CUDA
- Jupyter Notebook
- VS Code
- pytest
- Weights & Biases (W&B)

GPU-enabled experiments are run with PyTorch CUDA support when appropriate.

The Windows deployment package uses **Python 3.14**, **PyTorch 2.10.0**, and **CUDA 13.0**. It includes a dependency lockfile and the existing Python 3.14 Gensim wheel with its source and provenance. See [environment setup and workstation instructions](environment/README.md).

From the repository root, after installing 64-bit Python 3.14 and a compatible NVIDIA driver:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Tools\setup_environment.ps1 -ExpectedGpuCount 2
```

Use `-ExpectedGpuCount 1` on the laptop. Select `.venv\Scripts\python.exe` as the notebook interpreter after setup. Installation runs environment checks; notebook training is started separately. The two original COMPAS reference notebooks require additional R/rpy2 dependencies, as documented in the environment guide.

Local virtual environments, caches, experiment logs, and other machine-specific runtime files are excluded from version control.

---

## 📊 Experiment Tracking

Weights & Biases is configured for experiment tracking during model development and training.

Typical tracked information may include:

- training and validation metrics
- loss curves
- hyperparameters
- model experiment runs
- comparison between training configurations

Local `wandb/` runtime data is treated as generated output and is not part of the main repository structure.

---

## 🌳 Project Tree Tool

`Tools/Project_tree.py` provides three levels of project-tree output.

### Project Tree

Shows the primary course deliverables and useful project files.

```powershell
python Tools\Project_tree.py --mode project
```

### Detailed Tree

Includes additional source, data, and configuration files.

```powershell
python Tools\Project_tree.py --mode detailed
```

### Full Tree

Shows all project files while still excluding development environments, Git internals, and selected runtime/cache directories.

```powershell
python Tools\Project_tree.py --mode full
```

These modes can also be launched through:

**VS Code → Run and Debug → F5**

---

## 📌 Repository Notes

- Course materials are organized by category rather than submission date.
- Assignments, labs, and lecture notebooks are kept in separate directories.
- Generated outputs are included only when they are useful course deliverables.
- Virtual environments, caches, temporary files, experiment logs, and other unnecessary artifacts are excluded from version control.
- This repository is intended primarily as a personal course archive and development record.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

Bundled Gensim artifacts retain their upstream LGPL license; see [the bundled build and source notes](environment/wheels/README.md).
