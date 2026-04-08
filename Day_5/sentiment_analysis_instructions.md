# Sentiment Analysis with GloVe Embeddings — Session Instructions

---

## **Section 1: Setup and Understanding the Data**

### 🎯 Objective
Import necessary libraries and load the IMDB movie review dataset.

### 📝 Tasks
1. Import all required libraries
2. Check device availability (GPU or CPU)
3. Load the IMDB dataset and convert labels to numeric

### 🤔 Think About It
What makes a review positive vs negative? The words used, right? That's why embeddings are perfect — they capture word meaning!

---

## **Section 2: Text Preprocessing**

### 🎯 Objective
Clean and prepare the text data for our model.

**Steps:**
1. Convert to lowercase
2. Remove punctuation (`re.sub`)
3. Collapse whitespace
4. Split into word tokens
5. Build `word_to_idx` / `idx_to_word` dictionaries

### 🤔 Think About It
Why preprocess? Because `"Movie"`, `"movie"`, and `"movie!"` should all be treated as the same word!

---

## **Section 3: Loading Pre-trained GloVe Embeddings**

### 🎯 Objective
Load pre-trained GloVe embeddings and build an embedding matrix for our vocabulary.

**GloVe (Global Vectors for Word Representation)** was trained on Wikipedia + Gigaword.
- Each word is a 50-dimensional vector
- Captures rich semantic relationships — `"excellent"` and `"fantastic"` will be close together!

**Strategy:**
- If a word in our vocabulary has a GloVe vector → use it
- If not → initialise with a small random vector

---

## **Section 4: Preparing Data for Training**

### 🎯 Objective
Convert reviews to fixed-length index sequences and split into train / test sets.

**Why fixed length?** Neural networks need consistent input shapes.
**Padding** — add zeros at the end for short reviews.
**Truncating** — cut off long reviews at `max_length`.

---

## **Section 5: Building the Sentiment Classifier**

### 🎯 Objective
Create a neural network that uses GloVe embeddings to classify sentiment.

**Architecture:**
```
Input: [batch_size, max_length]         word index sequences
  ↓  Embedding Layer (freeze=False)
[batch_size, max_length, 50]            one vector per word
  ↓  Mean Pooling  →  [batch_size, 50]  average across all words
  ↓  Linear (50 → 128) + ReLU + Dropout(0.3)
  ↓  Linear (128 → 2)
Output logits: [batch_size, 2]          [negative_score, positive_score]
```

**Why average pooling?** We need ONE vector for the entire review.
**Why `freeze=False`?** We start with GloVe knowledge and fine-tune for sentiment.

---

## **Section 6: Training Setup**

Define the loss function, optimiser, and number of epochs before running the training loop.

- **Loss function:** `CrossEntropyLoss` — ideal for multi-class (here, 2-class) classification
- **Optimiser:** `Adam` with learning rate `0.001`
- **Epochs:** 5 passes over the full training set

---

## **Section 7: The Training Loop**

### 🎯 Objective
Train the classifier and track loss + accuracy each epoch.

**Training loop pattern** (same structure you saw in previous sessions):
1. `model.train()` → enable dropout
2. Mini-batch forward pass → compute loss → backprop → update weights
3. `model.eval()` + `torch.no_grad()` → evaluate on test set

**Expected results:** 85–90%+ test accuracy on the full IMDB dataset after a few epochs.

---

## **Section 8: Visualising Training Progress**

Plot loss and accuracy curves for both the training and test sets across all epochs.

**What to look for:**
- **Loss should decrease** over time
- **Accuracy should increase** over time
- **Small gap between train and test** = good generalisation; large gap = overfitting

---

## **Section 9: Detailed Evaluation**

### 🎯 Objective
Go beyond accuracy — look at precision, recall, F1, and where the model goes wrong.

| Metric | Meaning |
|---|---|
| **Precision** | Of all predicted Positive, how many really were? |
| **Recall** | Of all actual Positives, how many did we catch? |
| **F1-Score** | Harmonic mean of precision and recall |
| **Confusion Matrix** | Where the model confuses the two classes |

---

## **Section 10: Testing on New Reviews**

### 🎯 Objective
Create a reusable `predict_sentiment()` function and test on brand-new reviews.

### 🤔 Think About It
- Does your model generalise well to different writing styles?
- What about sarcasm? Mixed reviews? Very short reviews?

---

## **Section 12: Saving Your Model**

### 🎯 Objective
Persist the trained model so it can be reloaded later without retraining.

A **checkpoint** bundles:
- Model weights (`state_dict`)
- Vocabulary mappings
- Hyperparameters (embedding dim, max_length, etc.)
- Training history
