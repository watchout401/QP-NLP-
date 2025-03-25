# Quantitative Prediction

This project trains a Transformer based model to classify numerical magnitudes in comments and headlines.

## Colab Setup

- Use T4 GPU for hardware accelaration
- Install dependicies

```python
!pip install transformers
```

## Dataset loading

```Python
!gdown --folder --id {1mUiiYJsfHgI3y-LxhoqigmunxCBhT8Yw}
```

- This downloads dataset from google drive.
- This dataset has six files
- train, test, dev files for both healine and comment Data.
- Each sample includes
  - `masked`: The sentence where the number is masked.
  - `comment/headline`: The original text.
  - `magnitude`: The target class label (There are 8 class labels).

## Baseline Model

This implementation is based on **BERT** (`bert-base-uncased`), as referenced in the original paper. The model is fine-tuned for **predicting the magnitude of masked number** in both **headlines** and **comments**.

- Two seperate `BERT` models are fine-tuned for momments data and headline data.

## Dataset preparation

- The dataset is preprocessed and tokenized using **BERT tokenizer** to convert text into numerical inputs.
- The custom pytorch dataset handles tokenization using bert.
- 3 PyTorch datasets are made for each headline and comment data (train, test, dev).

## Training & Evaluation

To start training, run the cell with below contents:

```python
comment_trainer.train()
headline_trainer.train()
```

- This will train both comments and headline models.

For evaluation use

```python
comment_trainer.evaluate(comment_test)
headline_trainer.evaluate(headline_test)
```

- The primary evaluation metric used is `macro-f1` score.

## Using saved model

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("./results/best_model")
```

This now can be used to predict magnitude of a masked text.
