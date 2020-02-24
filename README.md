
<p align="center">
    <br>
    <img src="misc/logo.svg" alt="Bernie Logo" width="150"/>
    <br>
<p>

<p align="center">
    <a href="https://pypi.python.org/pypi/ernie/"><img alt="Downloads" src="https://img.shields.io/pypi/dm/ernie.svg?style=flat-square"></a>
    <a href="https://pypi.python.org/pypi/ernie/"><img alt="PyPi" src="https://img.shields.io/pypi/v/ernie.svg?style=flat-square"></a>
    <a href="https://github.com/brunneis/ernie/releases"><img alt="GitHub releases" src="https://img.shields.io/github/release/brunneis/ernie.svg?style=flat-square"></a>
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

# Usage
<a href="https://colab.research.google.com/drive/10lmqZyAHFP_-x4LxIQxZCavYpPqcR28c"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg?style=flat-square"></a>

## Binary Classification
```python
from ernie import BinaryClassifier, Models
import pandas as pd

tuples = [("This is a positive example. I'm very happy today.", 1),
          ("This is a negative sentence. Everything was wrong today at work.", 0)]

df = pd.DataFrame(tuples)
classifier = BinaryClassifier(model_name=Models.BertBaseUncased, max_length=128)
classifier.load_dataset(df, validation_split=0.2)
classifier.fine_tune(epochs=4, learning_rate=2e-5, training_batch_size=32, validation_batch_size=64)

sentence = "Oh, that's great!"
probabilities = classifier.predict_one(sentence) # It returns a tuple with the prediction
```

> You can use `classifier.predict` to predict several sentences at a time. The method will return a generator.

# Save and restore a fine-tuned model
## Save model
```python
classifier.dump('./model')
```

## Load model
```python
classifier = BinaryClassifier(model_path='./model')
```

# Supported Models
```python
>>> from ernie import ModelsByFamily
>>> print(ModelsByFamily.Supported)
{'bert-base-cased', 'xlnet-base-cased', 'roberta-large', 'bert-base-uncased', 'roberta-base', 'bert-large-cased', 'distilbert-base-uncased', 'distilbert-base-multilingual-cased', 'xlnet-large-cased', 'bert-large-uncased'}
```
