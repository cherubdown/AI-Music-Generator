**Bach Music Generator – Project Specification**  
**Version:** 1.0  
**Date:** February 2026  
**Goal:** Build a complete Python application that trains a neural network on the **JSB-Chorales** dataset (Bach’s 4-part chorales) and generates new music in Bach’s style. Output: new MIDI files (and optionally rendered scores/audio).

---

### 1. Project Overview

The application will:
1. Download / load the canonical JSB-Chorales dataset.
2. Pre-process the data (including key transposition for data augmentation).
3. Train a generative model (start with LSTM, later upgrade to Transformer).
4. Generate new 4-part chorales of arbitrary length.
5. Export the generated music as standard MIDI (and optionally MusicXML / audio preview).

Target users: anyone who wants a working “Bach-style composer” that can be extended (style transfer, conditioning on a melody, etc.).

---

### 2. Dataset

**Recommended sources (in order of preference)**

| Source | Format | Resolution | Notes | Link |
|--------|--------|------------|-------|------|
| **MusPy** (best) | `muspy.Music` objects | Original | Built-in loader, easy conversion to piano-roll / event / note representations, PyTorch/TensorFlow datasets | `pip install muspy` |
| czhuang/JSB-Chorales-dataset | pickle / .npz / JSON | 16th, 8th, quarter | Classic split from Boulanger-Lewandowski (2012): 229 train / 76 valid / 77 test | https://github.com/czhuang/JSB-Chorales-dataset |
| ageron/handson-ml2 | CSV (one per chorale) | 16th notes | 4 columns = S A T B MIDI pitches (0 = rest) | https://github.com/ageron/handson-ml2/tree/master/datasets/jsb_chorales |
| music21 corpus | MusicXML / MIDI | Original | 382 chorales, very clean, but you have to convert yourself | `music21.corpus.chorales.Iterator()` |

**Data augmentation (highly recommended)**
- Transpose every chorale to all 12 major/minor keys → ×12 dataset size.
- (Optional) random small time stretches / rhythmic variations.

---

### 3. Technology Stack (Recommended)

**Core**
- Python 3.10+
- **MusPy** – symbolic music toolkit (dataset + representations + conversion)
- **music21** – MIDI / MusicXML export, score rendering, analysis
- **PyTorch** (or TensorFlow) – model training
- **NumPy / Pandas** – data handling

**Optional / Nice-to-have**
- `pretty_midi` / `mido` – alternative MIDI I/O
- `gradio` or `streamlit` – web UI for generation
- `torchsummary` / `tensorboard` – training monitoring
- `fluidsynth` + `pygame` – real-time audio preview

**Requirements file skeleton**
```txt
muspy
music21
torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  # or cuda
numpy
pandas
gradio          # optional
matplotlib
tqdm
```

---

### 4. Project Structure

```
bach-ai-generator/
├── data/                  # raw + processed
│   └── jsb_chorales/      # gitignored or downloaded by script
├── src/
│   ├── data/              # Dataset, preprocessing, augmentation
│   ├── models/            # LSTM, Transformer, etc.
│   ├── training/          # Trainer, loss, metrics
│   ├── generation/        # sampling, post-processing
│   └── utils/             # MIDI conversion, visualization
├── notebooks/             # exploration
├── outputs/
│   ├── models/            # checkpoints
│   └── generated/         # .mid files
├── config.yaml            # hyperparameters
├── requirements.txt
├── train.py
├── generate.py
├── app.py                 # Gradio/Streamlit UI
└── README.md
```

---

### 5. Data Pipeline (Step-by-Step)

1. **Load**
   ```python
   import muspy
   dataset = muspy.JSBChoralesDataset(root="data/jsb_chorales", download_and_extract=True)
   ```

2. **Convert to ML-ready representation** (choose one)
   - **Piano-roll** (recommended for beginners) → shape `(time, 128, 4)` or flattened `(time, 512)`
   - **Event** (good for Transformers)
   - **Note** (list of (time, pitch, duration, velocity))
   - **Chord sequence** (4 pitches per timestep → vocabulary of ~130^4 is too big → usually one-hot per voice or embedding)

3. **Augmentation**
   - Transpose every chorale to all 12 keys (MusPy has `music.transpose()` helper).

4. **PyTorch Dataset**
   ```python
   torch_dataset = dataset.to_pytorch_dataset(
       representation="pianoroll",
       factory=lambda m: muspy.to_pianoroll_representation(m, encode_velocity=False)
   )
   ```

5. **Batching & padding** – use `torch.nn.utils.rnn.pad_sequence` or fixed-length windows (e.g., 64–128 timesteps).

---

### 6. Model Architecture Suggestions

**Starter (easy)**
- 2–3 layer LSTM (or GRU) with 512 hidden units.
- Input: one-hot or embedded chord (4 voices concatenated) or separate heads per voice.
- Output: softmax per voice (or multi-label for simultaneous notes).

**Better (recommended)**
- Causal Transformer (or Music Transformer / Perceiver AR style).
- Use **miditok** or MusPy’s event representation for tokenization.

**Advanced**
- Conditional model (condition on a soprano melody).
- Coconet-style (parallel Gibbs sampling) – great for chorales.

---

### 7. Training

- **Loss**: Cross-entropy per voice (or per note in event representation).
- **Optimizer**: AdamW (lr=1e-3 → 5e-4 with scheduler).
- **Schedule**: ReduceLROnPlateau + early stopping.
- **Metrics**: perplexity per voice, accuracy of next chord, optional music-theory metrics (parallel fifths/octaves, voice leading, etc.).
- **Hyperparameters** (store in `config.yaml`):
  - sequence length, batch size, epochs, hidden size, layers, dropout, etc.

---

### 8. Generation

1. Seed with first 4–8 chords of a real chorale (or random valid starting chord).
2. Autoregressive sampling (temperature, top-k, nucleus).
3. (Optional) post-processing:
   - Resolve voice crossings.
   - Enforce no parallel fifths/octaves (simple rule-based filter).
   - Quantize to 16th notes.

4. Convert back to MIDI:
   ```python
   music = muspy.from_pianoroll_representation(generated_roll)
   muspy.write_midi("generated.mid", music)
   # or use music21 to create a proper four-part score
   ```

---

### 9. Evaluation & Visualization

- **Quantitative**
  - Perplexity on held-out test set.
  - Note accuracy, chord accuracy.
- **Qualitative**
  - Listen to generated MIDI (use MuseScore / VLC / fluidsynth).
  - Plot piano-rolls of original vs. generated.
  - Optional: run music21 analysis (key, roman numerals, voice leading).

---

### 10. User Interface (Phase 2)

**CLI**
```bash
python generate.py --length 128 --temperature 0.9 --seed "bwv269"
```

**Web UI (Gradio)**
- Slider for length, temperature, top-k.
- Option to upload a soprano melody (future).
- “Generate” button → plays MIDI in browser.

---

### 11. Roadmap / Milestones

| Milestone | Deliverable | Time estimate |
|-----------|-------------|---------------|
| 1         | Dataset loader + piano-roll conversion + transposition | 1–2 days |
| 2         | LSTM baseline (train + generate simple chorales) | 3–4 days |
| 3         | Transformer + better tokenization | 1 week |
| 4         | Post-processing rules + evaluation | 2–3 days |
| 5         | Gradio UI + export options | 2 days |
| 6         | (Optional) conditioning on melody, style transfer | future |

---

### 12. Resources & References

- Original paper: Boulanger-Lewandowski et al. (2012) – “Modeling Temporal Dependencies in High-Dimensional Sequences”
- MusPy documentation: https://muspy.readthedocs.io
- Coconet (DeepBach inspiration): https://github.com/magenta/magenta/tree/main/magenta/models/coconet
- Excellent tutorial series: “Composing Bach-Like Music with Neural Networks” (YouTube + code on GitHub)
- music21 Bach corpus examples

---

**Next Step for You**

1. Clone the repo structure above.
2. Run `pip install muspy music21 torch` and download the dataset with MusPy (it does it automatically).
3. Start with the **piano-roll LSTM** baseline – you’ll have working generated Bach chorales in < 1 day.

Copy the entire content above into a file named **`BACH_AI_GENERATOR_SPEC.md`** and you have a complete, actionable specification.

Happy composing! If you want the full starter code skeleton (data loader + LSTM model + generation script) next, just say the word.