# Results — Iteration 1

**Model:** LinearSVC, C=1.0  
**Features:** Bag of Words (CountVectorizer)  
**Dataset:** UCI SMS Spam Collection — 5,574 messages (80/20 split)

---

## Accuracy

**Test accuracy: 0.9830**

---

## Classification Report

|          | Precision | Recall | F1     | Support |
|----------|-----------|--------|--------|---------|
| Ham      | 0.9834    | 1.0000 | 0.9916 | 966     |
| Spam     | 1.0000    | 0.8725 | 0.9319 | 149     |
| Accuracy |           |        | 0.9830 | 1115    |

---

## Key Findings

**Precision vs. Recall tradeoff:**  
Spam precision is perfect (1.00) — every message flagged as spam actually was spam, meaning zero false positives (Type I errors). However, spam recall is 0.87, meaning ~13% of spam messages slipped through as ham (Type II errors). For a spam filter, this is the preferred tradeoff: it's better to occasionally miss spam than to incorrectly delete legitimate mail.

**Class imbalance:**  
The dataset is heavily skewed — 966 ham vs. 149 spam in the test set (~87% ham). High overall accuracy (98.3%) is partly a reflection of this imbalance; a naive classifier that predicted "ham" for everything would score ~87%. The F1 scores per class are the more meaningful metric here.

**Bag of Words is sufficient:**  
A simple word-count representation with a linear kernel achieves strong results on this dataset, consistent with the book's observation that text data is often linearly separable in high-dimensional feature spaces.

---

## Next iteration candidates

- Decision threshold sweep — lower the classification threshold to improve spam recall
- N-grams (bigrams) — "free prize", "you won" are stronger signals than individual words
- Class weights — penalize missed spam more heavily during training (directly targets recall)
