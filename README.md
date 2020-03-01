
<p align="center">
    <br>
    <img src="misc/logo.svg" alt="Bernie Logo" width="150"/>
    <br>
<p>

<p align="center">
    <a href="https://pypi.python.org/pypi/ernie/"><img alt="Downloads" src="https://img.shields.io/pypi/dm/ernie.svg?style=flat-square"></a>
    <a href="https://pypi.python.org/pypi/ernie/"><img alt="PyPi" src="https://img.shields.io/pypi/v/ernie.svg?style=flat-square"></a>
    <!--<a href="https://github.com/brunneis/ernie/releases"><img alt="GitHub releases" src="https://img.shields.io/github/release/brunneis/ernie.svg?style=flat-square"></a>-->
    <a href="https://github.com/brunneis/ernie/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/github/license/brunneis/ernie.svg?style=flat-square&color=blue"></a>
</p>

<h3 align="center">
<b>BERT's best friend.</b>
</h3>

[![](https://sourcerer.io/fame/brunneis/brunneis/ernie/images/0)](https://sourcerer.io/fame/brunneis/brunneis/ernie/links/0)[![](https://sourcerer.io/fame/brunneis/brunneis/ernie/images/1)](https://sourcerer.io/fame/brunneis/brunneis/ernie/links/1)[![](https://sourcerer.io/fame/brunneis/brunneis/ernie/images/2)](https://sourcerer.io/fame/brunneis/brunneis/ernie/links/2)[![](https://sourcerer.io/fame/brunneis/brunneis/ernie/images/3)](https://sourcerer.io/fame/brunneis/brunneis/ernie/links/3)[![](https://sourcerer.io/fame/brunneis/brunneis/ernie/images/4)](https://sourcerer.io/fame/brunneis/brunneis/ernie/links/4)[![](https://sourcerer.io/fame/brunneis/brunneis/ernie/images/5)](https://sourcerer.io/fame/brunneis/brunneis/ernie/links/5)[![](https://sourcerer.io/fame/brunneis/brunneis/ernie/images/6)](https://sourcerer.io/fame/brunneis/brunneis/ernie/links/6)[![](https://sourcerer.io/fame/brunneis/brunneis/ernie/images/7)](https://sourcerer.io/fame/brunneis/brunneis/ernie/links/7)

# Installation
```bash
pip install ernie
```

# Fine-Tuning
<a href="https://colab.research.google.com/drive/10lmqZyAHFP_-x4LxIQxZCavYpPqcR28c"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg?style=flat-square"></a>

## Sentence Classification
```python
from ernie import SentenceClassifier, Models
import pandas as pd

tuples = [("This is a positive example. I'm very happy today.", 1),
          ("This is a negative sentence. Everything was wrong today at work.", 0)]

df = pd.DataFrame(tuples)
classifier = SentenceClassifier(model_name=Models.BertBaseUncased, max_length=128, labels_no=2)
classifier.load_dataset(df, validation_split=0.2)
classifier.fine_tune(epochs=4, learning_rate=2e-5, training_batch_size=32, validation_batch_size=64)
```

# Prediction
## Predict a single text
```python
text = "Oh, that's great!"

# It returns a tuple with the prediction
probabilities = classifier.predict_one(text)
```

## Predict multiple texts
```python
texts = ["Oh, that's great!", "That's really bad"]

# It returns a generator of tuples with the predictions
probabilities = classifier.predict(texts)
```

## Prediction Strategies
If the length in tokens of the texts is greater than the `max_length` with which the model has been fine-tuned, they will be truncated. To avoid losing information you can use a split strategy and aggregate the predictions in different ways.

### Split Strategies
- `GroupedSentencesWithoutUrls`. The text will be divided in groups of sentences with a lenght in tokens similar to `max_length`.

### Aggregation Strategies
- `Mean`: the prediction of the text will be the mean of the predictions of the splits.
- `MeanTop5`: the mean is computed over the 5 higher predictions only.
- `MeanTop10`: the mean is computed over the 10 higher predictions only.
- `MeanTop15`: the mean is computed over the 15 higher predictions only.
- `MeanTop20`: the mean is computed over the 20 higher predictions only.


```python
from ernie import SplitStrategies, AggregationStrategies
texts = ["Oh, that's great!", "That's really bad"]

probabilities = classifier.predict(texts,
                                   split_strategy=SplitStrategies.GroupedSentencesWithoutUrls,
                                   aggregation_strategy=AggregationStrategies.Mean) 
```

# Save and restore a fine-tuned model
## Save model
```python
classifier.dump('./model')
```

## Load model
```python
classifier = SentenceClassifier(model_path='./model')
```

# Supported Models
## BERT
- `BertBaseUncased`
- `BertBaseCased`
- `BertLargeUncased`
- `BertLargeCased`

## RoBERTa
- `RobertaBaseCased`
- `RobertaLargeCased`

## XLNet
- `XLNetBaseCased`
- `XLNetLargeCased`

## DistilBERT
- `DistilBertBaseUncased`
- `DistilBertBaseMultilingualCased`
