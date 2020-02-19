#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ernie import BinaryClassifier, Models
import pandas as pd

tuples = [("This is a positive example. I'm very happy today.", 1),
          ("This is a negative sentence. Everything was wrong today at work.", 0)]

df = pd.DataFrame(tuples)

classifier = BinaryClassifier(model=Models.BertBaseUncased)
classifier.load_dataset(df)
classifier.train()

sentence = "Oh, that's great!"
probability = classifier.predict(sentence)[1]
print(f"\"{sentence}\": {probability} [{'positive' if probability >= 0.5 else 'negative'}]")