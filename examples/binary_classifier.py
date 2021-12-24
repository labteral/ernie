#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ernie import SentenceClassifier, Models
import pandas as pd

tuples = [
    ("This is a positive example. I'm very happy today.", 1),
    ("This is a negative sentence. Everything was wrong today at work.", 0)
]

df = pd.DataFrame(tuples)

classifier = SentenceClassifier(
    model_name=Models.BertBaseUncased, max_length=128, labels_no=2)
classifier.load_dataset(df, validation_split=0.2)
classifier.fine_tune(epochs=4, learning_rate=2e-5,
                     training_batch_size=32, validation_batch_size=64)

sentence = "Oh, that's great!"
probability = classifier.predict_one(sentence)[1]
print(
    f"\"{sentence}\": {probability} "
    f"[{'positive' if probability >= 0.5 else 'negative'}]"
)
