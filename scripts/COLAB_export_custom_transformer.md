# Export the Custom Transformer (Track B) from Colab → dashboard

The dashboard auto-loads the Custom Transformer once these **3 files** exist in
`AIRD-social-Monitoring/models/CustomTransformer_TrackB/`:

```
state_dict.pt     # trained weights
word2idx.json     # the vocab map used to tokenise
config.json       # architecture + label order (so inference rebuilds the model)
```

The architecture in `ct_model.py` (server) is identical to the notebook's
`CustomTransformer`, so a notebook-trained `state_dict` loads directly.

---

## Step 1 — add ONE export cell to `AIRD_Project_Notebook.ipynb`

Paste this **right after the Custom Transformer training cell** (the one that
fills `ct_models_pt`) and **before** the cleanup cell that does `del word2idx`.
It exports the *actual trained* Track B model — no retraining:

```python
# === Export Custom Transformer (Track B) for the AIRD dashboard ===
import json, os, torch

OUT = f'{BASE_DIR}/models/CustomTransformer_TrackB'   # lands in your Drive
os.makedirs(OUT, exist_ok=True)

m = ct_models_pt['TrackB']                # trained Track B model (CPU copy)
torch.save(m.state_dict(), f'{OUT}/state_dict.pt')

json.dump(word2idx, open(f'{OUT}/word2idx.json', 'w'))   # vocab built from X_tr_o

json.dump({
    "vocab_size":  VOCAB_SIZE,            # 30000
    "max_len":     MAX_LEN,               # 256
    "num_classes": NUM_CLASSES,           # 7
    "embed_dim":   300, "num_heads": 6, "ff_dim": 1200, "num_blocks": 4,
    "labels": ["Anxiety", "Bipolar", "Depression", "Normal",
               "Personality Disorder", "Stress", "Suicidal"],
}, open(f'{OUT}/config.json', 'w'), indent=2)

print('Saved:', os.listdir(OUT))
```

> If `word2idx` was already deleted, rebuild it (it is deterministic — built from
> the original-train split `X_tr_o`):
> ```python
> from collections import Counter
> freq = Counter()
> for t in X_tr_o: freq.update(t.split())
> word2idx = {w: i for i, w in enumerate(['<PAD>', '<OOV>'] +
>             [w for w, _ in freq.most_common(VOCAB_SIZE - 2)])}
> ```
> And if `ct_models_pt` is gone, just re-run the Custom-Transformer training cell first.

## Step 2 — download the folder

In Colab: `Files` panel → `models/CustomTransformer_TrackB/` → download all three
files (or zip it: `!cd {BASE_DIR}/models && zip -r ct.zip CustomTransformer_TrackB`).

## Step 3 — drop it into the dashboard

Place the folder here:

```
AIRD-social-Monitoring/models/CustomTransformer_TrackB/
    state_dict.pt
    word2idx.json
    config.json
```

That's it — restart the app (or it's picked up on next load) and the **Custom
Transformer (Track B)** option goes live in the Emotion Partner selector.
Verify with:

```bash
python -c "import inference; print([m for m in inference.available_models() if m['key']=='customtransformer'])"
# -> [{'key': 'customtransformer', 'name': 'Custom Transformer (Track B)', 'live': True}]

python -c "import inference; print(inference.analyze('i feel hopeless and want to give up', 'customtransformer')['top_emotion'])"
```

---

### Alternative — retrain from scratch on Colab
If you'd rather retrain cleanly (CUDA), upload `ct_model.py` next to
`scripts/train_transformer.py` and run:
```bash
python scripts/train_transformer.py --data /content/drive/MyDrive/AIRD_Project/data --epochs 80 --patience 12
```
It writes the same three files to `models/CustomTransformer_TrackB/`.
(`--max_len` defaults to 256; the deploy serves at 128 via the `MAX_LEN` env, but
the model's own `config.json` length is what inference uses.)
