# Lab 2: Binary Classification from Scratch

**COMP 395 – Deep Learning**

In this lab, you will implement binary classification using gradient descent — the same algorithm that powers deep learning. You'll classify breast cancer tumors as malignant or benign using real medical data.

## Learning Objectives

By completing this lab, you will:

1. Implement the sigmoid activation function
2. Implement a forward pass (prediction)
3. Implement mean squared error loss
4. Derive and implement gradients using the chain rule
5. Train a model using gradient descent
6. Compare your model against an sklearn classifier

## Setup

1. Clone this repository
2. Install dependencies. Make sure that you have activated your virtual environment and run this in the command line:
   ```
   pip install -r requirements.txt
   ```

## Your Task

Open `binary_classification.py` and replace the `TODO` placeholders with your implementations. There are **9 TODOs** to complete across 4 functions:

| Function | Blanks | Description |
|----------|--------|-------------|
| `sigmoid(z)` | 1 | Sigmoid activation function |
| `forward(x, w, b)` | 2 | Compute prediction for one sample |
| `compute_loss(y, y_hat)` | 1 | Mean squared error loss |
| `compute_gradients(x, y, y_hat)` | 5 | Gradients for weights and bias |

The training loop has **4 additional TODOs** that call your functions.

## Testing Your Code

Run the tests to check your implementation:

```
python -m pytest test_binary_classification.py -v
```

Or run one group of tests at a time:

```
python -m pytest test_binary_classification.py::TestSigmoid -v
```

## Running the Full Training

Once all tests pass, run the complete training:

```
python binary_classification.py
```

You should see:
- Loss decreasing over epochs
- Training accuracy > 95%
- Test accuracy > 93%

## Files

| File | Description |
|------|-------------|
| `binary_classification.py` | **Your code goes here** — fill in the blanks |
| `comparison.py` | **Your code goes here** — model comparison (Part 2) |
| `test_binary_classification.py` | Automated tests for Part 1 (do not modify) |
| `test_comparison.py` | Automated tests for Part 2 (do not modify) |
| `requirements.txt` | Python dependencies |
| `overleaf-tutorial.pdf` | How to write your report in Overleaf (images, tables, exporting) |
| `sample_report/` | Example report (`.tex` + `.pdf`) and the script that made its figures |
| `report/` | **Your report goes here** — `report.tex`, `report.pdf`, and images (Part 3) |

## The Math

**Forward pass** (one sample):
```
z = w · x + b
ŷ = σ(z) = 1 / (1 + e^(-z))
```

**Loss** (one sample):
```
L = ½(ŷ - y)²
```

**Gradients** (one sample):
```
error = ŷ - y
sigmoid_deriv = ŷ(1 - ŷ)
δ = error × sigmoid_deriv

∂L/∂w = δ × x
∂L/∂b = δ
```

**Update rule**:
```
w = w - α × ∂L/∂w
b = b - α × ∂L/∂b
```

## Hints

- `torch.dot(a, b)` computes the dot product of two vectors
- `torch.exp(x)` computes e^x
- The sigmoid derivative can be computed from the output: `y_hat * (1 - y_hat)`
- Make sure your loss includes the `0.5` factor

## Part 2: Model Comparison

After completing your from-scratch implementation, choose a **different classification model** from [scikit-learn](https://scikit-learn.org/stable/supervised_learning.html) (e.g., Decision Tree, Random Forest, SVM, k-Nearest Neighbors, etc. — but **not** Logistic Regression, since that is essentially what you just built).

In `comparison.py` (check your work with `python -m pytest test_comparison.py -v`):

1. **Describe your chosen model** — Write a short paragraph (3–5 sentences) as a comment at the top of the file explaining how the model works and why you chose it.
2. **Implement it** — Train your chosen sklearn model on the same breast cancer dataset (use `load_data()` from `binary_classification.py`).
3. **Compare** — Print the test accuracy of both your from-scratch model and the sklearn model, and write a brief comment (2–3 sentences) discussing which performed better and why that might be.

> **Using AI for this part:** It is totally OK to use an LLM (ChatGPT, Codex, Claude Code, etc.) to write `comparison.py`. But you must give it context first: paste `test_comparison.py` and `binary_classification.py` into the chat, or open this folder in Codex / Claude Code so it can read them. You will also have to pick your scikit-learn model yourself in order to drive the code generation. As always, include your transcript and be ready to explain the code verbally.

## Part 3: Short Written Report (40 points)

Write a **very short report** (a few paragraphs, 1–2 pages with figures) in LaTeX and commit it in a folder called `report/` containing:

- `report.tex` — your LaTeX source
- `report.pdf` — the compiled PDF
- the images your report uses (at minimum `training_results.png`, which `binary_classification.py` saves when you run it)

Your report should briefly cover: (1) what you built and how you trained it, (2) your results — include your training plot as a figure and a table comparing the train/test accuracy of your from-scratch model and your sklearn model, and (3) a short discussion of which model did better and why that might be.

New to LaTeX? Read `overleaf-tutorial.pdf` (starting a project in Overleaf, inserting images and tables, exporting). A complete example on a *different* dataset is in `sample_report/` — `sample_report.tex`, the compiled `sample_report.pdf`, and `make_figures.py`, the script that produced its numbers and figures. Use it as a template for format and length, not content.

## Grading

| Component | Points |
|-----------|--------|
| `sigmoid` passes all tests | 12 |
| `forward` passes all tests | 16 |
| `compute_loss` passes all tests | 12 |
| `compute_gradients` passes all tests | 24 |
| Training loop runs and achieves >90% accuracy | 16 |
| Model comparison (description, implementation, and comparison) | 20 |
| Written report (`report.tex`, `report.pdf`, figures, results table, discussion) — graded by hand | 40 |
| **Total** | **140** |

## Submission

Commit and push your completed `binary_classification.py`, `comparison.py`, and `report/` folder (`.tex`, `.pdf`, and images) to the default branch of this repository. Classroom 50 automatically runs the tests on every push and reports your score for the code (the written report is graded by hand); you can push as many times as you like before the deadline.
