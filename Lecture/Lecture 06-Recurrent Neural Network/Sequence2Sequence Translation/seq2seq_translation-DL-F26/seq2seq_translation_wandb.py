"""
English -> Spanish Translation with a Seq2Seq RNN, with Weights & Biases Tracking
COMP 395 – Deep Learning

THIS FILE IS COMPLETE. Run it, let it train, and bring `translation_model.pt` to class.

The model is the encoder-decoder from the slides:
  * the ENCODER (an RNN) reads the English sentence and ends in one hidden state;
  * that state starts the DECODER (a second RNN), which writes Spanish one word at a
    time, feeding each word it writes back in as its next input, until it writes <eos>.

The structure is the same as linear_baseline_wandb.py: Part 1 is the experiment
function (model, loss, optimizer), Part 3 is train(..., run=None), Part 5 sets up the
W&B run inside try/finally. Two things are new because this run takes hours:

  * A checkpoint is saved after every epoch. If your laptop sleeps or you stop the
    run, start it again with --resume and it picks up where it left off.
  * At the end the model AND its vocabularies are saved to translation_model.pt, so
    you can load it in class and translate your own sentences.

Data: ~118,000 English-Spanish sentence pairs from Tatoeba (via manythings.org/anki,
CC-BY 2.0 FR), downloaded once into ./data.

Typical runs (run `python seq2seq_translation_wandb.py --help` for all options):
    python seq2seq_translation_wandb.py --smoke_test   # ~1 minute: checks everything works, estimates run time
    python seq2seq_translation_wandb.py                # the real run (hours)
    python seq2seq_translation_wandb.py --resume       # continue after a stop

ALWAYS run the smoke test first. A smoke test is a short run that checks
nothing is broken before you commit hours to the real run (the name comes from
hardware: switch it on and check nothing starts smoking). It does not save a checkpoint
or create a W&B run.
On a Mac, keep it awake with:   caffeinate -i python seq2seq_translation_wandb.py
"""

import argparse
import math
import os
import random
import re
import time
import unicodedata
import urllib.request
import zipfile
from collections import Counter

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")   # save figures without opening windows
import matplotlib.pyplot as plt
import wandb
from torch.utils.data import TensorDataset, DataLoader

PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<sos>", "<eos>", "<unk>"]


# =============================================================================
# PART 1: The Experiment (model, loss, optimizer)
# =============================================================================

class Seq2Seq(nn.Module):
    """Encoder RNN -> final hidden state -> decoder RNN (as on the slides)."""

    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim=128, hidden_size=256, cell="gru"):
        super().__init__()
        RNN = nn.GRU if cell == "gru" else nn.LSTM
        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim, padding_idx=PAD)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=PAD)
        self.encoder = RNN(embed_dim, hidden_size, batch_first=True)
        self.decoder = RNN(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, tgt_vocab_size)

    def encode(self, src):
        """English ids (batch, src_len) -> final hidden state. This is the context c."""
        _, state = self.encoder(self.src_embedding(src))
        return state

    def forward(self, src, tgt_in):
        """Training: the decoder is fed the CORRECT previous word (teacher forcing).
        Returns scores (batch, tgt_len, tgt_vocab_size)."""
        out, _ = self.decoder(self.tgt_embedding(tgt_in), self.encode(src))
        return self.fc(out)

    @torch.no_grad()
    def generate(self, src, max_len=20):
        """Translation: the decoder is fed its OWN previous word. Returns ids (batch, max_len)."""
        state = self.encode(src)
        word = torch.full((src.size(0), 1), SOS, dtype=torch.long, device=src.device)
        outputs = []
        for _ in range(max_len):
            out, state = self.decoder(self.tgt_embedding(word), state)
            word = self.fc(out).argmax(dim=-1)          # most likely next word
            outputs.append(word)
        return torch.cat(outputs, dim=1)


def seq2seq_experiment(src_vocab_size, tgt_vocab_size, embed_dim=128, hidden_size=256, cell="gru", lr=1e-3):
    """Encoder-decoder RNN: how well does one fixed-size vector carry a sentence?"""
    model = Seq2Seq(src_vocab_size, tgt_vocab_size, embed_dim, hidden_size, cell)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)     # don't score the padding
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, loss_fn, optimizer


# =============================================================================
# PART 2: Data Loading
# =============================================================================

DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip"


def download_pairs(data_dir="./data", data_file=None):
    """Return a list of (english, spanish) strings. Downloads ~2.6 MB once."""
    if data_file is None:
        os.makedirs(data_dir, exist_ok=True)
        data_file = os.path.join(data_dir, "spa-eng", "spa.txt")
        if not os.path.exists(data_file):
            zip_path = os.path.join(data_dir, "spa-eng.zip")
            print(f"Downloading {DATA_URL} ...")
            urllib.request.urlretrieve(DATA_URL, zip_path)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(data_dir)
    pairs = []
    with open(data_file, encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) >= 2:
                pairs.append((cols[0], cols[1]))
    return pairs


def tokenize(text):
    """Lowercase, split punctuation off words. Keeps Spanish accents and ¿ ¡."""
    text = unicodedata.normalize("NFC", text.lower().strip())
    text = re.sub(r"([.!?,¿¡])", r" \1 ", text)
    return text.split()


def build_vocab(sentences, min_count=2):
    """Words seen at least min_count times get an id; everything else becomes <unk>."""
    counts = Counter(w for s in sentences for w in s)
    words = SPECIALS + sorted(w for w, c in counts.items() if c >= min_count)
    return {w: i for i, w in enumerate(words)}


def encode_src(words, vocab, max_len):
    """English: ids, LEFT-padded to max_len (same as Day 2, so the last step is the last word)."""
    ids = [vocab.get(w, UNK) for w in words]
    return [PAD] * (max_len - len(ids)) + ids


def encode_tgt(words, vocab, max_len):
    """Spanish: <sos> ids <eos>, RIGHT-padded to max_len + 2."""
    ids = [SOS] + [vocab.get(w, UNK) for w in words] + [EOS]
    return ids + [PAD] * (max_len + 2 - len(ids))


def load_data(max_len=10, max_pairs=None, val_fraction=0.05, seed=0, data_dir="./data", data_file=None):
    """Tokenize, keep pairs of at most max_len words, split, build vocabularies, make tensors."""
    pairs = [(tokenize(en), tokenize(es)) for en, es in download_pairs(data_dir, data_file)]
    pairs = [(en, es) for en, es in pairs if 0 < len(en) <= max_len and 0 < len(es) <= max_len]
    random.Random(seed).shuffle(pairs)
    if max_pairs:
        pairs = pairs[:max_pairs]
    n_val = max(1, int(len(pairs) * val_fraction))
    train_pairs, val_pairs = pairs[n_val:], pairs[:n_val]

    src_vocab = build_vocab([en for en, _ in train_pairs])
    tgt_vocab = build_vocab([es for _, es in train_pairs])

    def to_tensors(ps):
        src = torch.tensor([encode_src(en, src_vocab, max_len) for en, _ in ps])
        tgt = torch.tensor([encode_tgt(es, tgt_vocab, max_len) for _, es in ps])
        return src, tgt

    src_train, tgt_train = to_tensors(train_pairs)
    src_val, tgt_val = to_tensors(val_pairs)
    return src_train, tgt_train, src_val, tgt_val, src_vocab, tgt_vocab


# =============================================================================
# PART 3: Training Loop
# =============================================================================

def train(model, loss_fn, optimizer, src_train, tgt_train, n_epochs=20, batch_size=64,
          clip=1.0, verbose=True, run=None, src_val=None, tgt_val=None, seed=0,
          device="cpu", start_epoch=0, losses=None, checkpoint=None, tgt_vocab=None):
    """
    Train the model with the given loss and optimizer.

    Returns:
        losses: list of average training loss per epoch
    """
    loader = DataLoader(TensorDataset(src_train, tgt_train), batch_size=batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(seed + start_epoch))
    losses = losses or []

    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_loss, n_tokens, t0 = 0.0, 0, time.time()

        for src_batch, tgt_batch in loader:
            src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
            tgt_in, tgt_out = tgt_batch[:, :-1], tgt_batch[:, 1:]   # predict each next word

            # Forward pass
            scores = model(src_batch, tgt_in)
            loss = loss_fn(scores.reshape(-1, scores.size(-1)), tgt_out.reshape(-1))

            # Backward pass and update
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)   # Day 2: stops one bad batch wrecking the weights
            optimizer.step()

            k = (tgt_out != PAD).sum().item()
            epoch_loss += loss.item() * k
            n_tokens += k

        avg_loss = epoch_loss / n_tokens
        losses.append(avg_loss)

        # Validation metrics (one greedy translation pass over the validation set)
        metrics = {"train/loss": avg_loss, "epoch_minutes": (time.time() - t0) / 60}
        if src_val is not None:
            metrics["val/loss"] = evaluate_loss(model, loss_fn, src_val, tgt_val, device)
            word_acc, bleu = translation_scores(model, src_val, tgt_val, device)
            metrics["val/word_acc"], metrics["val/bleu"] = word_acc, bleu

        # Log this epoch to wandb (skipped when no run is passed)
        if run is not None:
            run.log(metrics, step=epoch)

        if verbose:
            msg = f"Epoch {epoch + 1:3d}/{n_epochs}  loss {avg_loss:.4f}  ({metrics['epoch_minutes']:.1f} min)"
            if src_val is not None:
                msg += f"  val word acc {metrics['val/word_acc']:.3f}  BLEU {metrics['val/bleu']:.1f}"
            print(msg, flush=True)

        # Save a checkpoint so the run can be resumed with --resume
        if checkpoint is not None:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "epoch": epoch + 1, "losses": losses}, checkpoint)

    return losses


# =============================================================================
# PART 4: Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_loss(model, loss_fn, src, tgt, device, batch_size=512):
    """Average loss per word on a whole dataset."""
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(src), batch_size):
        s, t = src[i:i + batch_size].to(device), tgt[i:i + batch_size].to(device)
        scores = model(s, t[:, :-1])
        k = (t[:, 1:] != PAD).sum().item()
        total += loss_fn(scores.reshape(-1, scores.size(-1)), t[:, 1:].reshape(-1)).item() * k
        n += k
    return total / n


@torch.no_grad()
def predict(model, src, device, batch_size=512):
    """Greedy translations (word ids) for every row of src. The model feeds itself."""
    model.eval()
    max_len = src.size(1) + 1          # room for every Spanish word plus <eos>
    return torch.cat([model.generate(src[i:i + batch_size].to(device), max_len).cpu()
                      for i in range(0, len(src), batch_size)])


def strip(ids):
    """Word ids -> list of words up to <eos> (no padding or <sos>)."""
    out = []
    for i in ids:
        if i == EOS:
            break
        if i not in (PAD, SOS):
            out.append(i)
    return out


def ngrams(words, n):
    return Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))


def bleu(hyps, refs, max_n=4):
    """Corpus BLEU (0-100), the score translation papers report (Papineni et al., 2002).

    For n = 1..4: what fraction of the model's n-word chunks also appear in the
    reference (anywhere in the sentence, each reference chunk usable once)?
    BLEU = geometric mean of those 4 precisions x a penalty for translations that are
    too short. Unlike word accuracy, one extra word does not ruin the rest of the sentence.
    """
    matches, totals = [0] * max_n, [0] * max_n
    hyp_len = ref_len = 0
    for h, r in zip(hyps, refs):
        hyp_len, ref_len = hyp_len + len(h), ref_len + len(r)
        for n in range(1, max_n + 1):
            h_grams, r_grams = ngrams(h, n), ngrams(r, n)
            matches[n - 1] += sum(min(c, r_grams[g]) for g, c in h_grams.items())
            totals[n - 1] += max(len(h) - n + 1, 0)
    if hyp_len == 0:
        return 0.0
    # Skip n-gram sizes that never occur (very short sentences); smooth a zero count to 0.1
    log_p = [math.log(max(m, 0.1) / t) for m, t in zip(matches, totals) if t > 0]
    brevity = min(1.0, math.exp(1 - ref_len / hyp_len))
    return 100 * brevity * math.exp(sum(log_p) / len(log_p))


def translation_scores(model, src, tgt, device):
    """Translate src and compare with tgt. Returns (word accuracy, BLEU)."""
    pred, ref = predict(model, src, device), tgt[:, 1:]
    mask = ref != PAD
    word_acc = ((pred == ref) & mask).sum().item() / mask.sum().item()
    return word_acc, bleu([strip(p) for p in pred.tolist()], [strip(r) for r in ref.tolist()])


def scores_by_length(model, src, tgt, device):
    """Exact match, word accuracy and BLEU, grouped by English sentence length -> the bottleneck curve."""
    pred, ref = predict(model, src, device), tgt[:, 1:]
    src_len = (src != PAD).sum(dim=1)
    mask = ref != PAD
    word_ok = (pred == ref) | ~mask
    rows = []
    for L in sorted(set(src_len.tolist())):
        sel = src_len == L
        rows.append([L, int(sel.sum()),
                     word_ok[sel].all(dim=1).float().mean().item(),                      # whole sentence right
                     ((pred[sel] == ref[sel]) & mask[sel]).sum().item() / mask[sel].sum().item(),
                     bleu([strip(p) for p in pred[sel].tolist()], [strip(r) for r in ref[sel].tolist()])])
    return rows   # [english_length, n_sentences, exact_match, word_accuracy, bleu]


def ids_to_words(ids, vocab):
    inv = {i: w for w, i in vocab.items()}
    words = []
    for i in ids:
        if i == EOS:
            break
        if i not in (PAD, SOS):
            words.append(inv.get(int(i), "<unk>"))
    return " ".join(words)


def translate(model, sentence, src_vocab, tgt_vocab, max_len=10, device="cpu"):
    """English string -> Spanish string."""
    words = tokenize(sentence)[-max_len:]
    src = torch.tensor([encode_src(words, src_vocab, max_len)])
    return ids_to_words(predict(model, src, device)[0].tolist(), tgt_vocab)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# PART 5: Main (run training and evaluation)
# =============================================================================

EXAMPLES = ["I am hungry.", "Where is the library?", "We see the sky.",
            "She doesn't like coffee.", "Tom wants to go home now.",
            "I think it will rain tomorrow afternoon."]


def smoke_test(src_train, tgt_train, src_vocab, tgt_vocab, args, device):
    """A short run that checks every piece works before the real run, then estimates its length."""
    print("\nSMOKE TEST")
    passed = True

    def check(ok, message):
        nonlocal passed
        passed = passed and ok
        print(("  \u2713 " if ok else "  \u2717 ") + message)

    # 1. The data loaded and has the expected shapes
    check(len(src_train) > 1000 and src_train.size(1) == args.max_len and tgt_train.size(1) == args.max_len + 2,
          f"data: {len(src_train)} training pairs, English {tuple(src_train.shape)}, Spanish {tuple(tgt_train.shape)}")

    # 2. One forward pass gives one score per Spanish word per vocabulary entry
    torch.manual_seed(args.seed)
    model, loss_fn, optimizer = seq2seq_experiment(len(src_vocab), len(tgt_vocab), args.embed_dim,
                                                   args.hidden_size, args.cell, args.lr)
    model.to(device)
    src, tgt = src_train[:32].to(device), tgt_train[:32].to(device)
    scores = model(src, tgt[:, :-1])
    check(tuple(scores.shape) == (32, args.max_len + 1, len(tgt_vocab)),
          f"forward pass: scores {tuple(scores.shape)}")

    # 3. The model can memorize one small batch (if it can't, something is wrong with training)
    model.train()
    first = None
    for _ in range(100):
        scores = model(src, tgt[:, :-1])
        loss = loss_fn(scores.reshape(-1, scores.size(-1)), tgt[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        first = first if first is not None else loss.item()
    check(loss.item() < 0.5 * first, f"memorize one batch: loss {first:.2f} -> {loss.item():.2f}")

    # 4. Translation runs, and saving + loading gives the same model
    before = translate(model, "We see the sky.", src_vocab, tgt_vocab, args.max_len, device)
    path = "smoke_test_model.pt"
    torch.save(model.state_dict(), path)
    model.load_state_dict(torch.load(path, map_location=device))
    os.remove(path)
    after = translate(model, "We see the sky.", src_vocab, tgt_vocab, args.max_len, device)
    check(before == after, f'translate + save/load: "We see the sky." -> "{after}"')

    # 5. Time a short training run to estimate the real one
    n = min(2000, len(src_train))
    t0 = time.time()
    train(model, loss_fn, optimizer, src_train[:n], tgt_train[:n], n_epochs=1,
          batch_size=args.batch_size, clip=args.clip, verbose=False, device=device)
    seconds_per_pair = (time.time() - t0) / n
    epoch_minutes = seconds_per_pair * len(src_train) / 60
    print(f"\n  One epoch on all {len(src_train)} pairs: about {epoch_minutes:.1f} min")
    print(f"  {args.n_epochs} epochs (--n_epochs {args.n_epochs}): about {epoch_minutes * args.n_epochs / 60:.1f} hours"
          " (plus a little for validation)")

    if passed:
        print("\nSMOKE TEST PASSED. Start the real run with the same options, without --smoke_test.")
    else:
        print("\nSMOKE TEST FAILED. Fix the \u2717 lines (or ask in class) before starting the real run.")


def parse_args():
    p = argparse.ArgumentParser(description="English->Spanish seq2seq RNN with wandb logging")
    p.add_argument("--n_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--cell", choices=["gru", "lstm"], default="gru")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--max_len", type=int, default=10, help="keep sentences of at most this many words")
    p.add_argument("--max_pairs", type=int, default=None, help="use only this many pairs (for quick tests)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--data_file", type=str, default=None, help="use a local tab-separated file instead of downloading")
    p.add_argument("--smoke_test", action="store_true", help="~1 minute check that everything works; estimates run time")
    p.add_argument("--resume", action="store_true", help="continue from checkpoint.pt")
    p.add_argument("--wandb_project", type=str, default="comp395-seq2seq")
    p.add_argument("--wandb_run_name", type=str, default="seq2seq_gru")
    p.add_argument("--no_wandb", action="store_true", help="disable wandb logging")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = get_device()
    print("Device:", device)

    # Load data
    print("Loading data...")
    src_train, tgt_train, src_val, tgt_val, src_vocab, tgt_vocab = load_data(
        args.max_len, args.max_pairs, seed=args.seed, data_dir=args.data_dir, data_file=args.data_file)
    print(f"Training pairs: {len(src_train)}   Validation pairs: {len(src_val)}")
    print(f"English vocabulary: {len(src_vocab)}   Spanish vocabulary: {len(tgt_vocab)}")

    if args.smoke_test:
        smoke_test(src_train, tgt_train, src_vocab, tgt_vocab, args, device)
        raise SystemExit

    torch.manual_seed(args.seed)
    model, loss_fn, optimizer = seq2seq_experiment(len(src_vocab), len(tgt_vocab), args.embed_dim,
                                                   args.hidden_size, args.cell, args.lr)
    model.to(device)

    # Resume from a checkpoint if asked
    start_epoch, losses = 0, []
    checkpoint = "checkpoint.pt"
    if args.resume and os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, losses = ckpt["epoch"], ckpt["losses"]
        print(f"Resuming after epoch {start_epoch}")

    # Use W&B's default entity (or WANDB_ENTITY); a login username may not be a valid entity.
    # Config records the choices that define this run.
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        group="seq2seq",
        config={"experiment": "seq2seq_translation",
                "n_params": sum(p.numel() for p in model.parameters()),
                "cell": args.cell, "hidden_size": args.hidden_size, "embed_dim": args.embed_dim,
                "loss": type(loss_fn).__name__,
                "optimizer": type(optimizer).__name__, "optimizer_lr": args.lr, "clip": args.clip,
                "n_epochs": args.n_epochs, "batch_size": args.batch_size, "max_len": args.max_len,
                "seed": args.seed, "dataset": "tatoeba-spa-eng",
                "n_train": len(src_train), "n_val": len(src_val)},
        mode="disabled" if args.no_wandb else None,
        resume="allow" if args.resume else None,
    )
    cfg = run.config

    try:
        # Train (per-epoch metrics are logged inside train)
        print("\nTraining...")
        losses = train(model, loss_fn, optimizer, src_train, tgt_train,
                       n_epochs=cfg.n_epochs, batch_size=cfg.batch_size, clip=cfg.clip,
                       run=run, src_val=src_val, tgt_val=tgt_val, seed=cfg.seed, device=device,
                       start_epoch=start_epoch, losses=losses, checkpoint=checkpoint)

        # Evaluate
        print("\nEvaluating...")
        val_loss = evaluate_loss(model, loss_fn, src_val, tgt_val, device)
        val_acc, val_bleu = translation_scores(model, src_val, tgt_val, device)
        by_len = scores_by_length(model, src_val, tgt_val, device)
        print(f"Validation loss: {val_loss:.4f}   word accuracy: {val_acc:.4f}   BLEU: {val_bleu:.1f}")
        print("\nEnglish length   sentences   exact match   word accuracy    BLEU")
        for L, n, exact, word, b in by_len:
            print(f"{L:14d}   {n:9d}   {exact:11.3f}   {word:13.3f}   {b:5.1f}")

        print("\nExample translations:")
        examples = [[en, translate(model, en, src_vocab, tgt_vocab, cfg.max_len, device)] for en in EXAMPLES]
        for en, es in examples:
            print(f"  {en:45s} -> {es}")

        # Final results go in the run summary (shown in the runs table)
        run.summary["val_loss"] = val_loss
        run.summary["val_word_acc"] = val_acc
        run.summary["val_bleu"] = val_bleu
        run.summary["final_loss"] = losses[-1]

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        axes[0].plot(range(1, len(losses) + 1), losses, "o-")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Training loss (per word)")
        axes[0].set_title("Training Loss")

        axes[1].plot([r[0] for r in by_len], [r[2] for r in by_len], "o-", label="whole sentence right")
        axes[1].plot([r[0] for r in by_len], [r[3] for r in by_len], "s-", label="words right")
        axes[1].set_xlabel("English sentence length (words)")
        axes[1].set_ylabel("Validation accuracy")
        axes[1].set_ylim(0, 1)
        axes[1].set_title("Accuracy vs. length")
        axes[1].legend()

        axes[2].plot([r[0] for r in by_len], [r[4] for r in by_len], "o-")
        axes[2].set_xlabel("English sentence length (words)")
        axes[2].set_ylabel("Validation BLEU")
        axes[2].set_ylim(0, 100)
        axes[2].set_title("BLEU vs. length (compare: Bahdanau Fig. 2)")

        plt.tight_layout()
        plt.savefig("translation_results.png", dpi=150)
        print("\nPlot saved to translation_results.png")

        # Log the figure and tables
        run.log({"translation_results": wandb.Image(fig),
                 "scores_by_length": wandb.Table(
                     columns=["english_length", "n_sentences", "exact_match", "word_accuracy", "bleu"], data=by_len),
                 "examples": wandb.Table(columns=["english", "spanish"], data=examples)})

        # Save the model WITH its vocabularies, so it can be loaded in class
        torch.save({"model": model.state_dict(), "src_vocab": src_vocab, "tgt_vocab": tgt_vocab,
                    "config": dict(cfg)}, "translation_model.pt")
        print("Model saved to translation_model.pt")
        if not args.no_wandb:
            artifact = wandb.Artifact("seq2seq-translation", type="model")
            artifact.add_file("translation_model.pt")
            run.log_artifact(artifact)
    finally:
        run.finish()
