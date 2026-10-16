# ict4403-fgai-assignment2
# Shakespeare Next-Word Text Generation

## Problem and objective

This project compares four recurrent neural network models for word-level autoregressive text generation: Simple RNN, Bidirectional RNN, LSTM, and GRU. Each model uses a sequence of previous words to predict the next word, then generates text one word at a time.

## Dataset

**Tiny Shakespeare** is a collection of Shakespearean play text, including dialogue and speaker labels.

- Source: [Tiny Shakespeare dataset](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt)
- Size: 1,115,394 characters
- Processed tokens: 250,172
- Training/validation split: 90% / 10%, in text order

The notebook downloads the dataset directly from its source.

## Preprocessing and input sequences

Text was lowercased and tokenized into words and common punctuation. The vocabulary contains the 7,999 most common training tokens and an `<unk>` token, for a total vocabulary size of 8,000. Tokens outside the vocabulary are mapped to `<unk>`.

Each model receives a 20-token sequence and predicts the next token. Windows were first created with a stride of 5. To keep CPU training time manageable, every fourth training window was used, giving 11,257 training examples. All 5,000 validation examples were used for evaluation.

## Models and training

The four models use 64-dimensional word embeddings and 128 hidden units. They were trained with:

- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 128
- Epochs: 10
- Loss: Cross-entropy
- Device: CPU

The Bidirectional RNN reads each 20-token input window in both directions. Its input contains only words before the target.

## Results

| Model | Validation loss | Perplexity | Validation accuracy | Training time (seconds) |
|---|---:|---:|---:|---:|
| LSTM | 5.9124 | 369.6010 | 14.04% | 82.33 |
| GRU | 5.9142 | 370.2428 | 14.02% | 63.66 |
| Simple RNN | 5.9176 | 371.5118 | 13.62% | 40.86 |
| Bidirectional RNN | 6.2756 | 531.4409 | 11.24% | 73.58 |

See [`metrics.csv`](metrics.csv) for the recorded values and [`model_comparison.png`](model_comparison.png) for a plot.

## Sampling-temperature analysis

Text was generated from the same 20-token seed at temperatures 0.5, 1.0, and 1.5. The samples are in [`generated_samples.txt`](generated_samples.txt).

- **0.5:** outputs were more repetitive, though some preserved speaker labels and dialogue-like patterns.
- **1.0:** outputs were more varied, with occasional Shakespearean phrases, but coherence was inconsistent.
- **1.5:** outputs used a wider range of words, including rare or archaic terms, but were generally less coherent.

## Comparison and conclusion

LSTM had the lowest epoch-10 validation loss and perplexity and the highest validation accuracy. GRU was very close on both validation metrics and trained faster. Simple RNN trained fastest, while Bidirectional RNN had the weakest epoch-10 validation results.

Based on the measured validation results, LSTM was the strongest overall model in this run, although its lead over GRU was small. Generated passages from all four models were still fragmentary. Training loss continued to fall while validation loss began rising for several models, suggesting overfitting in later epochs. Increasing training time alone did not consistently improve generation quality.

## Files

- [`IA2_FGAI_Assignment2.ipynb`](IA2_FGAI_Assignment2.ipynb): preprocessing, model training, evaluation, and generation
- [`metrics.csv`](metrics.csv): quantitative comparison
- [`generated_samples.txt`](generated_samples.txt): samples at each temperature
- [`model_comparison.png`](model_comparison.png): comparison plot

## Run the notebook

Open the notebook in Google Colab and run its cells from top to bottom. The notebook downloads the dataset from the source link above.

Dependencies: PyTorch, pandas, and Matplotlib. Google Colab includes these packages in its standard runtime.
