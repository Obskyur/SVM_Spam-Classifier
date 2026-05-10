# SVM Spam Classifier

A hands-on project exploring Support Vector Machines (SVM) for binary text classification, built while working through [*The Hundred-Page Machine Learning Book*](https://themlbook.com/wiki/doku.php) by Andriy Burkov.

## Learning Goals

- Understand how SVMs find a maximum-margin decision boundary between two classes
- Apply bag-of-words feature extraction to convert raw text into numeric feature vectors
- Train a linear SVM on real SMS data and evaluate its performance
- Visualize how model accuracy changes as training set size grows (learning curve)
- Interpret a confusion matrix and understand the tradeoff between precision and recall in spam detection

## Dataset

[SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) — 5,574 SMS messages labeled `ham` (not spam) or `spam`.

## Approach

1. Load and explore the dataset
2. Convert messages to bag-of-words feature vectors using `CountVectorizer`
3. Train a linear SVM (`LinearSVC`) on a training split
4. Evaluate with accuracy, confusion matrix, and precision/recall/F1
5. Plot a learning curve (accuracy vs. training set size)
6. (Optional) Repeat with TF-IDF features and compare results

## Setup

```bash
pip install -r requirements.txt
jupyter notebook spam_classifier.ipynb
```

## Project Structure

```
svm-spam-classifier/
├── data/               # Raw dataset (downloaded on first run)
├── spam_classifier.ipynb
├── requirements.txt
└── README.md
```
