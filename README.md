# 🧠 LLM Preference Modeling & Reward Ranking (Kaggle)

This project builds a hybrid machine learning system to predict human preferences in pairwise “chatbot battles” between Large Language Models.

🏆 Current Rank: **42 / 228 (Top 18%)**  
📊 Best Validation Log Loss: **1.0092**

---

## 📌 Problem Overview

Given:
- A prompt
- Two model responses (A and B)
- Human-labeled winner (A, B, or Tie)

Goal:
Predict which response humans would prefer.

This is effectively a **reward modeling / ranking problem**.

---

## 🔍 Key Insights

### 1️⃣ Verbosity Bias
Exploratory analysis revealed:

- When Model A wins, it was longer **61.34%** of the time.
- When Model B wins, it was longer **61.59%** of the time.

Conclusion: Response length is a dominant signal.

---

## 🚀 Approach

### Step 1: Data Cleaning
- JSON decoding
- Unicode normalization
- Whitespace standardization
- Code block preservation
- Control character removal

---

### Step 2: Data Augmentation (Swap Trick)

To enforce positional invariance:

- Swapped Response A and B
- Flipped labels accordingly
- Doubled dataset size

Training samples:
Original: 57,477
Augmented: 114,954

---

### Step 3: Feature Engineering

#### Human-Centric Meta Features (36 total)
- Length difference
- Word count difference
- Unique word count
- Punctuation counts
- URL detection
- LaTeX detection
- Code block detection
- Capitalization ratio
- Sentence count
- Formatting signals

---

### Step 4: TF-IDF Difference Vectors

Built 5,000-dimensional TF-IDF vectors and computed:

X = Vector_A − Vector_B

This captures semantic differences between responses.

Final feature matrix
114,954 samples × 5,036 features


---

### Step 5: XGBoost Model

- `n_estimators=500`
- `max_depth=6`
- `learning_rate=0.05`
- GPU-accelerated (CUDA)
- Regularization + subsampling

Validation Log Loss:
1.0166

---

### Step 6: DeBERTa-v3 Cross-Encoder

Model: `microsoft/deberta-v3-base`

Input format:
[CLS] Prompt [SEP] Response A [SEP] Response B [SEP]


Training:
- Gradient accumulation
- Cosine LR scheduler
- Mixed precision (FP16)
- Max sequence length: 512

---

### Step 7: Model Ensembling

Performed weighted blending search:

Best Blend:
0.85 XGBoost
0.15 DeBERTa

Final Validation Log Loss:
1.0092

---

## 🛠 Tech Stack

- Python
- XGBoost (CUDA)
- PyTorch
- HuggingFace Transformers
- DeBERTa-v3
- TF-IDF (scikit-learn)
- Feature Engineering
- Model Blending

---

## 📈 Results

- Top 18% Kaggle leaderboard (42 / 228)
- Successfully combined traditional ML + transformer fine-tuning
- Built full end-to-end submission pipeline

---

## 🧪 Future Improvements

- Full cross-validation
- Larger transformer (DeBERTa-large)
- Knowledge distillation
- Rank-based loss optimization
- Advanced ensembling

---

## 💡 Key Learnings

- Human preference modeling requires both semantic understanding and bias awareness.
- Hybrid systems (tree models + transformers) can outperform single-model solutions.
- Ensembling remains one of the strongest practical improvements in Kaggle competitions.

---

