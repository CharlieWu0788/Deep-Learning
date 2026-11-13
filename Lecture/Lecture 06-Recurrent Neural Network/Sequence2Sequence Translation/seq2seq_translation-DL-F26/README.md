# English → Spanish Translation with a Seq2Seq RNN
COMP 395 – Deep Learning

Train an encoder–decoder RNN at home, then bring the trained model to the next class.

## Files

| File | What it does |
|---|---|
| `seq2seq_translation_wandb.py` | Downloads the data, trains the model, logs to W&B, saves `translation_model.pt` |
| `translate.py` | Loads `translation_model.pt` and translates your own sentences (used in class) |

## Setup

Use your PyTorch environment. You also need `matplotlib` and `wandb`, which you already installed for earlier assignments.
Keep this folder by itself. The data (~3 MB) downloads into `./data` the first time you run the script.

## Steps

**1. Smoke test (about 1 minute).** This checks that everything works and estimates how long training will take on your computer.

```
python seq2seq_translation_wandb.py --smoke_test
```

Every line should show ✓. If the estimate is longer than you can leave your computer running, lower the number of epochs in step 2 (for example, `--n_epochs 10`).

**2. Train (a few hours).** Plug in your laptop and keep it awake.

```
python seq2seq_translation_wandb.py                          # Windows / Linux
caffeinate -i python seq2seq_translation_wandb.py            # Mac (stops the Mac from sleeping)
```

Each epoch prints the training loss, validation word accuracy and validation BLEU. These also appear in W&B under the project `comp395-seq2seq`.

Runs use your W&B account's default entity (project workspace). To choose another workspace, set `WANDB_ENTITY` to its existing name. Your login username is not necessarily a valid entity.

**3. If training stops** (sleep, crash, Ctrl-C), continue from the last finished epoch:

```
python seq2seq_translation_wandb.py --resume
```

**4. Try it:**

```
python translate.py "We see the sky."
python translate.py            # type sentences; press Enter on an empty line to quit
```

## Bring to class

- `translation_model.pt` (the trained model and its vocabularies)
- `translation_results.png` (the loss curve, plus accuracy and BLEU by sentence length; compare the BLEU plot with Figure 2 of the Bahdanau paper)

## Files the script creates

`data/`, `checkpoint.pt`, `translation_model.pt`, `translation_results.png`, `wandb/`. When the unit is over, you can delete all of them except `translation_model.pt`.

## Useful options

Run `python seq2seq_translation_wandb.py --help` to see all options.

| Option | Default | |
|---|---|---|
| `--n_epochs` | 20 | Fewer epochs = shorter run |
| `--cell` | `gru` | `lstm` to try an LSTM |
| `--hidden_size` | 256 | Size of the context vector *c* |
| `--max_len` | 10 | Longest sentence (in words) to keep |
| `--no_wandb` | off | Train without logging to W&B |

Data: English–Spanish sentence pairs from the Tatoeba Project (tatoeba.org), via manythings.org/anki, CC-BY 2.0 FR.
