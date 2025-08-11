# BBC News Classification

This project classifies BBC news articles into 5 categories:

- **Business**
- **Entertainment**
- **Politics**
- **Sport**
- **Tech**

It uses two models:

1. **Baseline** – Logistic Regression with TF-IDF vectors
2. **Transformer** – Fine-tuned DistilBERT from Hugging Face

It also includes a minimal **Streamlit app** for live text classification.

---

## Dataset

The dataset comes from the [BBC News Classification Dataset – Kaggle](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news) (or equivalent source).

**Folder Structure:**

```bash
bbc/
├── business/
│   ├── file1.txt
│   ├── file2.txt
│   └── ...
├── entertainment/
│   ├── file1.txt
│   └── ...
├── politics/
│   ├── file1.txt
│   └── ...
├── sport/
│   ├── file1.txt
│   └── ...
└── tech/
    ├── file1.txt
    └── ...
```

## How to Run the Notebook

1. **Open `assignment.ipynb` in Jupyter or VSCode and run cells sequentially.**
   - The cell with `!pip install ...` is optional if you preinstalled dependencies.
2. **The notebook will:**
   - Clean text and split into train/test sets.
   - Train a TF-IDF + LogisticRegression baseline and save it as `baseline_tfidf_logreg.joblib`.
   - Fine-tune `distilbert-base-uncased` and save the model under `distilbert_bbc_saved/`.

> **Note:** DistilBERT training requires significant CPU resources; a GPU is recommended (use a CUDA-enabled setup or Google Colab).

---

## How to Run the Streamlit App

1. **Ensure the following files exist (produced by the notebook):**
   - `baseline_tfidf_logreg.joblib`
   - `distilbert_bbc_saved/` (optional, if using BERT)
2. **Run the app:**
   ```bash
   streamlit run app.py
   ```
