# Text Similarity (NLP)

Compute **semantic similarity** between pairs of texts using **Sentence-Transformers** (RoBERTa-based embeddings) and **cosine similarity**.

This repository is primarily implemented as a Jupyter Notebook and produces an `output.csv` file containing similarity scores.

## What this project does

Given a dataset containing two text fields (`text1`, `text2`), the notebook:

1. Loads the dataset (`Text_Similarity_Dataset.csv`).
2. Cleans the text (lowercasing and removing extra symbols).
3. Encodes each sentence with a pretrained Sentence-Transformer model.
4. Computes **cosine similarity** between the two embeddings.
5. Normalizes similarity from `[-1, 1]` to `[0, 1]` using min-max scaling: `(sim + 1) / 2`.
6. Saves predictions to `output.csv` with columns:
   - `Unique_ID`
   - `similarity`

## Repository contents

- `submission.ipynb` — main notebook (end-to-end pipeline)
- `Text_Similarity_Dataset.csv` — input dataset (text pairs)
- `output.csv` — generated output (similarity per `Unique_ID`)

## Requirements

You can run this in Google Colab or locally.

Typical dependencies:

- Python 3.8+
- `transformers`
- `sentence-transformers`
- `torch`
- `pandas`, `numpy`

Install (baseline):

```bash
pip install transformers sentence-transformers torch pandas numpy
```

## How to run

### Option A: Google Colab

1. Open `submission.ipynb` in Colab.
2. Update any local path / Google Drive `cd ...` cell to match your environment.
3. Run all cells.
4. Confirm `output.csv` is generated.

### Option B: Run locally

1. Clone the repo.
2. Install requirements.
3. Launch Jupyter:

```bash
jupyter notebook
```

4. Open `submission.ipynb` and run all cells.

## Model used

The notebook uses:

- `stsb-roberta-base-v2` via `SentenceTransformer(...)`

This model produces **768-dimensional** sentence embeddings.

## Notes / improvements

- Encoding row-by-row can be slow for large datasets. For speed, batch encoding is recommended (`model.encode(list_of_sentences, batch_size=..., show_progress_bar=True)`).
- Consider saving embeddings to disk if you iterate frequently.
- Add a `requirements.txt` for reproducibility.

## License

Add a `LICENSE` file if you plan to reuse/distribute this project.
