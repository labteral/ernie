#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pandas as pd
from ernie import SentenceClassifier, Models  # noqa: F401


class TestLoadModel(unittest.TestCase):
    tuples = [
        ("This is a negative sentence. Everything was wrong today.", 0),
        ("This is a positive example. I'm very happy today.", 1),
        ("This is a neutral sentence. That's normal.", 2),
    ]
    df = pd.DataFrame(tuples)

    classifier = SentenceClassifier(
        model_name='xlm-roberta-large',
        max_length=128,
        labels_no=3,
    )
    classifier.load_dataset(df, validation_split=0.2)
    classifier.fine_tune(
        epochs=4,
        learning_rate=2e-5,
        training_batch_size=32,
        validation_batch_size=64,
    )

    def test_predict(self):
        text = "Oh, that's great!"
        prediction = self.classifier.predict_one(text)
        self.assertEqual(len(prediction), 3)


if __name__ == '__main__':
    unittest.main()
