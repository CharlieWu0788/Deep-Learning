#!/usr/bin/env bash
#
# Run the from-scratch lab and the PyTorch version, then check that they
# learn the same model.
#
# Usage:
#   ./compare_implementations.sh [LAB_FILE] [TORCH_FILE]
#
# Defaults: binary_classification.py and binary_classification_pytorch.py
# in the same folder as this script. Set PYTHON=... to pick an interpreter.
# Add --compare-only to skip running each script's own main block.
#
# Exit code 0 means the two implementations agree.

set -euo pipefail

COMPARE_ONLY=0
ARGS=()
for a in "$@"; do
    if [[ "$a" == "--compare-only" ]]; then COMPARE_ONLY=1; else ARGS+=("$a"); fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_FILE="${ARGS[0]:-$SCRIPT_DIR/binary_classification.py}"
TORCH_FILE="${ARGS[1]:-$SCRIPT_DIR/binary_classification_pytorch.py}"
PYTHON="${PYTHON:-python3}"

# Save plots to disk instead of opening windows
export MPLBACKEND=Agg

for f in "$LAB_FILE" "$TORCH_FILE"; do
    if [[ ! -f "$f" ]]; then
        echo "Error: file not found: $f" >&2
        exit 2
    fi
done

rule() { printf '%*s\n' 70 '' | tr ' ' '='; }

run_script() {
    local title="$1" file="$2"
    rule; echo " $title"; echo " $file"; rule
    local start=$SECONDS
    ( cd "$(dirname "$file")" && "$PYTHON" "$(basename "$file")" )
    printf '\n[%s finished in about %d s]\n\n' "$title" "$((SECONDS - start))"
}

if [[ $COMPARE_ONLY -eq 0 ]]; then
    run_script "1/3  Lab: gradient descent by hand" "$LAB_FILE"
    run_script "2/3  PyTorch: nn.Module + torch.optim" "$TORCH_FILE"
fi

rule; echo " 3/3  Side-by-side comparison"; rule

"$PYTHON" - "$LAB_FILE" "$TORCH_FILE" <<'PYEOF'
import importlib.util, sys, time
import torch

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

lab = load("lab", sys.argv[1])
pt = load("pt", sys.argv[2])

X_train, X_test, y_train, y_test, _ = lab.load_data()

# --- Check 1: one hand-computed gradient vs. autograd -----------------------
try:
    x0, y0 = X_train[0], y_train[0]
    w0 = torch.randn(X_train.shape[1]) * 0.1
    b0 = torch.tensor(0.05)
    dw_hand, db_hand = lab.compute_gradients(x0, y0, lab.forward(x0, w0, b0))

    model = pt.BinaryClassifier(X_train.shape[1])
    with torch.no_grad():
        model.linear.weight.copy_(w0.unsqueeze(0))
        model.linear.bias.fill_(b0.item())
    loss = 0.5 * torch.nn.MSELoss()(model(x0.unsqueeze(0)), y0.unsqueeze(0))
    loss.backward()
    dw_auto = model.linear.weight.grad.squeeze(0)
    db_auto = model.linear.bias.grad.squeeze(0)
    g_w = (dw_hand - dw_auto).abs().max().item()
    g_b = abs(float(db_hand) - db_auto.item())
except Exception as e:
    print("The lab file still has TODOs (or an error in them):")
    print(f"  {type(e).__name__}: {e}")
    print("Complete the lab file before comparing.")
    sys.exit(3)

print("One-sample gradient check (random w, b)")
print(f"  max |dw_hand - dw_autograd| = {g_w:.2e}")
print(f"      |db_hand - db_autograd| = {g_b:.2e}\n")

# --- Check 2: full training runs --------------------------------------------
t = time.perf_counter()
w, b, lab_losses = lab.train(X_train, y_train, alpha=0.01, n_epochs=100, verbose=False)
t_lab = time.perf_counter() - t

t = time.perf_counter()
model, pt_losses = pt.train(X_train, y_train, alpha=0.01, n_epochs=100, verbose=False)
t_pt = time.perf_counter() - t

lab_tr = lab.accuracy(y_train, lab.predict(X_train, w, b))
lab_te = lab.accuracy(y_test, lab.predict(X_test, w, b))
pt_tr = pt.accuracy(y_train, pt.predict(model, X_train))
pt_te = pt.accuracy(y_test, pt.predict(model, X_test))

w_diff = (model.w - w).abs().max().item()
b_diff = abs(model.b.item() - float(b))
loss_diff = max(abs(a - c) for a, c in zip(lab_losses, pt_losses))

row = "  {:<22}{:>14}{:>14}"
print(row.format("", "Lab", "PyTorch"))
print(row.format("Loss, epoch 1", f"{lab_losses[0]:.6f}", f"{pt_losses[0]:.6f}"))
print(row.format("Loss, epoch 100", f"{lab_losses[-1]:.6f}", f"{pt_losses[-1]:.6f}"))
print(row.format("Training accuracy", f"{lab_tr:.4f}", f"{pt_tr:.4f}"))
print(row.format("Test accuracy", f"{lab_te:.4f}", f"{pt_te:.4f}"))
print(row.format("Training time (s)", f"{t_lab:.2f}", f"{t_pt:.2f}"))
print()
print(f"  max weight difference   = {w_diff:.2e}")
print(f"  bias difference         = {b_diff:.2e}")
print(f"  max per-epoch loss diff = {loss_diff:.2e}")
print()

TOL = 1e-4   # float32 rounding stays far below this
checks = {
    "gradients match":   max(g_w, g_b) < 1e-5,
    "weights match":     w_diff < TOL,
    "bias matches":      b_diff < TOL,
    "loss curves match": loss_diff < TOL,
    "accuracies match":  lab_tr == pt_tr and lab_te == pt_te,
}
for name, ok in checks.items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

if all(checks.values()):
    print("\nThe two implementations learn the same model.")
    sys.exit(0)
print("\nThe implementations differ. Check the lab's loss, gradients, and update.")
sys.exit(1)
PYEOF
