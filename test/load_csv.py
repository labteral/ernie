#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from ernie import SentenceClassifier, Models


class TestLoadCsv(unittest.TestCase):
    classifier = SentenceClassifier(
        model_name=Models.BertBaseUncased,
        max_length=128,
        labels_no=2,
    )
    classifier.load_dataset(
        validation_split=0.2,
        csv_path="example.csv",
        read_csv_kwargs={"header": None},
    )
    classifier.fine_tune(
        epochs=4,
        learning_rate=2e-5,
        training_batch_size=32,
        validation_batch_size=64,
    )

    def test_predict(self):
        text = "Oh, that's great!"
        prediction = self.classifier.predict_one(text)
        self.assertEqual(len(prediction), 2)


if __name__ == '__main__':
    unittest.main()
