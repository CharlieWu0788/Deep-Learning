"""
Sample Report: experiments and figures
COMP 395 – Deep Learning

Compares several binary classifiers against a random forest on the UCI
Ionosphere dataset (351 radar returns, 34 features, labels good/bad).
Produces accuracy_comparison.png, training_loss.png and results.txt,
which sample_report.tex uses.

Run with: python make_figures.py   (needs internet once, to download the data)
"""

import torch
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, SURFACE = "#0b0b0b", "#898781", "#fcfcfb"


def load_data():
    """Load Ionosphere, split 80/20, normalize with training statistics."""
    data = fetch_openml("ionosphere", version=1, as_frame=False)
    X = data.data.astype("float32")
    y = (data.target == "g").astype("float32")  # 1 = good return, 0 = bad

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    std[std == 0] = 1.0  # one feature is constant
    return (X_train - mean) / std, (X_test - mean) / std, y_train, y_test


def train_neuron(X, y, alpha=0.01, n_epochs=100):
    """The same single sigmoid neuron + MSE + SGD built in Lab 2."""
    X, y = torch.tensor(X), torch.tensor(y)
    w, b = torch.zeros(X.shape[1]), torch.tensor(0.0)
    losses = []
    for _ in range(n_epochs):
        total = 0.0
        for x_i, y_i in zip(X, y):
            y_hat = 1 / (1 + torch.exp(-(torch.dot(w, x_i) + b)))
            total += (0.5 * (y_hat - y_i) ** 2).item()
            delta = (y_hat - y_i) * y_hat * (1 - y_hat)
            w, b = w - alpha * delta * x_i, b - alpha * delta
        losses.append(total / len(y))
    return w, b, losses


def neuron_accuracy(X, y, w, b):
    z = torch.tensor(X) @ w + b
    return ((z >= 0).float() == torch.tensor(y)).float().mean().item()


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    w, b, losses = train_neuron(X_train, y_train)
    results = [("Sigmoid neuron (from scratch)",
                neuron_accuracy(X_train, y_train, w, b),
                neuron_accuracy(X_test, y_test, w, b), None)]

    models = [
        ("Logistic regression", LogisticRegression(max_iter=1000)),
        ("k-nearest neighbors (k=5)", KNeighborsClassifier(n_neighbors=5)),
        ("SVM (RBF kernel)", SVC()),
        ("Decision tree", DecisionTreeClassifier(random_state=42)),
        ("Random forest (200 trees)", RandomForestClassifier(n_estimators=200, random_state=42)),
    ]
    for name, model in models:
        model.fit(X_train, y_train)
        cv = cross_val_score(model, X_train, y_train, cv=5).mean()
        results.append((name, model.score(X_train, y_train), model.score(X_test, y_test), cv))

    with open("results.txt", "w") as f:
        f.write(f"train n={len(y_train)}, test n={len(y_test)}\n")
        f.write(f"{'model':32s} train   test    5-fold CV\n")
        for name, tr, te, cv in results:
            cv_s = "  --  " if cv is None else f"{cv:.3f}"
            f.write(f"{name:32s} {tr:.3f}   {te:.3f}   {cv_s}\n")
    print(open("results.txt").read())

    # Figure 1: test accuracy, random forest highlighted
    order = sorted(results, key=lambda r: r[2])
    fig, ax = plt.subplots(figsize=(6.5, 3.2), facecolor=SURFACE)
    names = [r[0] for r in order]
    accs = [100 * r[2] for r in order]
    colors = [ORANGE if "forest" in n else BLUE for n in names]
    # Dots rather than bars: the axis does not start at zero, and bar length
    # would exaggerate the differences.
    ax.scatter(accs, names, s=70, color=colors, zorder=3)
    for name, acc in zip(names, accs):
        ax.text(acc + 0.5, name, f"{acc:.1f}", va="center", color=INK, fontsize=9)
    ax.set_xlim(80, 100)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlabel("Test accuracy (%)", color=INK)
    ax.grid(axis="x", color="#e8e7e3", linewidth=0.8)
    ax.legend(handles=[plt.Line2D([], [], marker="o", linestyle="", color=ORANGE),
                       plt.Line2D([], [], marker="o", linestyle="", color=BLUE)],
              labels=["Random forest", "Other methods"], frameon=False,
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig("accuracy_comparison.png", dpi=200)

    # Figure 2: training loss of the from-scratch neuron
    fig, ax = plt.subplots(figsize=(6.5, 3.0), facecolor=SURFACE)
    ax.plot(range(1, len(losses) + 1), losses, color=BLUE, linewidth=2)
    ax.set_xlabel("Epoch", color=INK)
    ax.set_ylabel("Average MSE loss", color=INK)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#e8e7e3", linewidth=0.8)
    style(ax)
    fig.tight_layout()
    fig.savefig("training_loss.png", dpi=200)
