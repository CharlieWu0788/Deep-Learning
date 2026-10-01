# COMP395 — Deep Learning

Course repository for **COMP395: Deep Learning** at Occidental College.

This repository serves as a structured record of coursework completed throughout the semester, including assignments, labs, lecture notebooks, experiment results, and supporting development tools.

---

## 📁 Project Structure

```text
COMP395/
├── .vscode/
├── Assignments/
│   ├── Assignment 1-Differentiation through multiple representations/
│   │   └── Assignment 1.ipynb
│   ├── Assignment 2-Partial Derivatives and Gradients/
│   └── Assignment 3-Gradient Descent from Scratch/
│       └── Assignment 3.ipynb
├── Labs/
│   ├── Lab 1-Introduction to PyTorch and GPU Computing/
│   │   ├── Lab1_files/
│   │   ├── bonus_speedup_threshold.png
│   │   ├── cpu_vs_gpu_benchmark.png
│   │   ├── Lab1.pdf
│   │   └── wu_charlie_lab1.ipynb
│   └── Lab 2-Breast Cancer/
│       ├── report/
│       │   ├── README.md
│       │   ├── report.pdf
│       │   └── training_results.png
│       ├── sample_report/
│       │   ├── accuracy_comparison.png
│       │   ├── make_figures.py
│       │   ├── sample_report.pdf
│       │   └── training_loss.png
│       ├── binary_classification.py
│       ├── binary_classification_pytorch.py
│       ├── comparison.py
│       ├── test_binary_classification.py
│       ├── test_comparison.py
│       ├── training_results.png
│       └── training_results_pytorch.png
├── Lecture/
│   ├── Lecture 1-Matrix_operations_in_pytorch/
│   │   └── Lecture 1.ipynb
│   ├── Lecture 2-Partial_Derivatives_and_Gradients/
│   │   └── Lecture 2.ipynb
│   └── Lecture 3-Gradient Descent/
│       └── Lecture 3.ipynb
├── Tools/
│   └── Project_tree.py
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

### Lectures

Lecture notebooks and examples organized by topic.

- **Lecture 1 — Matrix Operations in PyTorch**
- **Lecture 2 — Partial Derivatives and Gradients**
- **Lecture 3 — Gradient Descent**

### Tools

Utilities used to inspect and maintain the repository.

- `Tools/Project_tree.py` — generates formatted directory trees at multiple levels of detail

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