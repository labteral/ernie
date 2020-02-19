
<p align="center">
    <br>
    <img src="misc/logo.svg" alt="Bernie Logo" width="150"/>
    <br>
<p>

<p align="center">
    <a href="https://github.com/brunneis/ernie/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/brunneis/ernie.svg?style=flat-square&color=blue">
    </a>
    <a href="https://github.com/brunneis/ernie/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/brunneis/ernie.svg?style=flat-square">
    </a>
</p>

<h3 align="center">
<b>BERT's best friend.</b>
</h3>

[![](https://sourcerer.io/fame/brunneis/brunneis/ernie/images/0)](https://sourcerer.io/fame/brunneis/brunneis/ernie/links/0)[![](https://sourcerer.io/fame/brunneis/brunneis/ernie/images/1)](https://sourcerer.io/fame/brunneis/brunneis/ernie/links/1)[![](https://sourcerer.io/fame/brunneis/brunneis/ernie/images/2)](https://sourcerer.io/fame/brunneis/brunneis/ernie/links/2)[![](https://sourcerer.io/fame/brunneis/brunneis/ernie/images/3)](https://sourcerer.io/fame/brunneis/brunneis/ernie/links/3)[![](https://sourcerer.io/fame/brunneis/brunneis/ernie/images/4)](https://sourcerer.io/fame/brunneis/brunneis/ernie/links/4)[![](https://sourcerer.io/fame/brunneis/brunneis/ernie/images/5)](https://sourcerer.io/fame/brunneis/brunneis/ernie/links/5)[![](https://sourcerer.io/fame/brunneis/brunneis/ernie/images/6)](https://sourcerer.io/fame/brunneis/brunneis/ernie/links/6)[![](https://sourcerer.io/fame/brunneis/brunneis/ernie/images/7)](https://sourcerer.io/fame/brunneis/brunneis/ernie/links/7)

## Installation
```bash
pip install ernie
```

## Binary Classification
```python
from ernie import BinaryClassifier, Models
import pandas as pd

tuples = [("This is a positive example. I'm very happy today.", 1),
          ("This is a negative sentence. Everything was wrong today at work.", 0)]

df = pd.DataFrame(tuples)

classifier = BinaryClassifier(model=Models.BertBaseUncased)
classifier.load_dataset(df, validation_size=0.2)
classifier.train(epochs=4, training_batch_size=32, validation_batch_size=64)

sentence = "Oh, that's great!"
probabilities = classifier.predict(sentence)
```