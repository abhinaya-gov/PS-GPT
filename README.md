# PS-GPT
# Ponniyin Selvan GPT ⚔️

A **character-level GPT (Generative Pre-trained Transformer)** trained from scratch on Tamil text from *Ponniyin Selvan*, the famous historical novel written by **Kalki Krishnamurthy**.

This project implements a **decoder-only transformer architecture** in PyTorch that learns to generate Tamil text in the style of the novel.

The model learns **character-by-character language modeling** and can generate new Tamil sentences after training.

---

# 📖 About the Project

*Ponniyin Selvan* is one of the most celebrated Tamil historical novels, following the story of **Arulmozhivarman (Raja Raja Chola I)** and the political intrigue of the Chola dynasty.

This project trains a **GPT-style transformer from scratch** on Tamil text extracted from the novel.
Instead of word tokens, the model operates at the **character level**, allowing it to learn the structure of Tamil Unicode characters directly.

The model learns to predict:

next_character | previous_characters

and generates Tamil text autoregressively.

This implementation is inspired by **Andrej Karpathy's nanoGPT**, but implemented from scratch in **PyTorch**.

---

# 🏗️ Architecture

The model is a **decoder-only transformer** composed of stacked transformer blocks.

## Core Components

| Component            | Description                                            |
| -------------------- | ------------------------------------------------------ |
| Token Embedding      | Converts character tokens into embedding vectors       |
| Positional Embedding | Encodes token position within the context window       |
| Self Attention       | Multi-head causal attention using Q, K, V projections  |
| Feed Forward Network | Two-layer MLP with ReLU and 4× hidden expansion        |
| Transformer Block    | LayerNorm → Self Attention → Residual → FFN → Residual |
| Language Model Head  | Linear projection from embeddings to vocabulary        |

The model uses **causal masking** so tokens only attend to previous tokens.

---

# ⚙️ Hyperparameters

| Hyperparameter              | Value |
| --------------------------- | ----- |
| Batch Size                  | 32    |
| Context Length (Block Size) | 512   |
| Embedding Dimension         | 512   |
| Attention Heads             | 16    |
| Transformer Layers          | 6     |
| Dropout                     | 0.2   |
| Learning Rate               | 3e-4  |
| Optimizer                   | AdamW |
| Epochs                      | 25    |

Total parameters: **~20M parameters**.

---

# 📁 Project Structure
```
Ponniyin-Selvan-GPT/
│
├── ponniayan.txt        # Tamil text dataset
├── model_training.ipynb # Main notebook (training + generation)
├── PS_weights.pt        # Saved model weights
└── README.md
```

---

# 📊 Dataset

Source: Tamil text from **Ponniyin Selvan**

| Property                 | Value                                 |
| ------------------------ | ------------------------------------- |
| Tokenization             | Character level                       |
| Train / Validation Split | 90% / 10%                             |
| Vocabulary               | Unique Tamil characters + punctuation |

The vocabulary includes:

* Tamil vowels
* Tamil consonants
* vowel modifiers
* punctuation
* whitespace

This allows the model to learn **Tamil grammar and structure directly from characters**.

---

# 🚀 Getting Started

## Install Dependencies

pip install torch

---

## Training

Run the training notebook or script.

python train.py

The model automatically detects **Apple Silicon GPU (MPS)** if available:

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

Training prints loss per epoch.

Example:

Epoch 1 | Loss: 2.2711 |<br>
Epoch 10 | Loss: 1.0144 |<br>
Epoch 25 | Loss: 0.7713 | <br>

---

# 🧠 Text Generation

After training, the model can generate Tamil text.

Example:

context = torch.zeros((1,1), dtype=torch.long).to(DEVICE) <br>
generated = model.generate(context, max_new_tokens=500) <br>
print(decode(generated[0].tolist())) <br>

Example generated output:

```
அத்தியாயம் 12 - பிறை ஒன்றும் சொல்லித்தானே! அதற்காக “அம்மா, என்ன? நீ ஒப்புக்கொள்ள வேண்டாம்!” என்று தேவராளன் ஆபரணம் இருவர் இச்சமயம் தொடங்கினான்.
“உண்மையானால் இளவரசர் சேர்த்துக் கொண்டு போ! அந்த வீரப் புரடல் எதற்காகக் கிடைக்கிறார்?”
“எனக்கு எச்சரிக்கை செய்திருக்கிறாய். வருகிறவரை கடலில் போகிறேன்.”
“இருந
================================================================================

நாராயணா! உன்னைப் போல் யாரேனும் கொல்லிற்று”
“வேறு எப்படி நாடெங்கும் அப்படித்தான் எறிந்தேன்? எப்போது?”
“தாங்கள் கேட்கச் சொல்கிறேன்! விஷத்தைப்பற்றி நாங்கள் எங்கேயோ கேட்கிறோம்.”
“பெண்ணே! இளவரசே! எத்தனையோ ஓர் உறைத்துக் கொண்டிருக்கிறேன். ஆமாதிரி இந்தக் குற்றம் சொல்கிறீர்களே! உண்மையானதற்காக இந்த மரத்தை எனக
================================================================================

“தேவி! இடந்தான் ஏன் சொன்னேனே? இன்று ஆண்டுகளுக்கு முக்கியம் என்ன செய்து பார்க்கிறேன்!” என்று வந்தியத்தேவன் சிரித்தான்.
மதுராந்தகத் தேவரின் பூங்குழலி ஜயகோஷம் என்று இளவரசர் கம்பீரமான திரும்பிப் பார்த்தான். தேவரின் வாழ்க்கை வந்தியத்தேவனை ரொம்பக் கயிறுகிறது. ஆனால் கந்தமாறன் அங்கே இருந்திருக்கிறான், அவனுட
================================================================================
```


The text resembles the **style and structure of the original novel**.

---

# 🔬 Key Concepts Implemented

This project implements several fundamental transformer concepts:

* Character-level language modeling
* Token + positional embeddings
* Multi-head scaled dot-product attention

Attention(Q,K,V) = softmax(QKᵀ / √dₖ) V

* Causal masking for autoregressive generation
* Residual connections
* Layer normalization
* Transformer feed-forward networks
* AdamW optimizer

---

# ⚡ Hardware

The project supports:

* **Apple Silicon GPU (MPS)** – fastest
* **CPU fallback**

Example device detection:

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

---

# 📚 References

* **Attention Is All You Need** — Vaswani et al., 2017
* **nanoGPT** — Andrej Karpathy
* **Ponniyin Selvan** — Kalki Krishnamurthy

---

# 📜 License

This project is released under the **MIT License**.
