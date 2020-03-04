#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from ernie import SentenceClassifier, Models


class TestPredict(unittest.TestCase):
    classifier = SentenceClassifier(model_name=Models.BertBaseUncased, max_length=128, labels_no=2)

    def test_batch_predict(self):
        sentences_no = 50
        predictions = list(self.classifier.predict(["this is a test " * 100] * sentences_no))
        self.assertEqual(len(predictions), sentences_no)

    def test_predict_one(self):
        sentences_no = 50
        prediction = self.classifier.predict_one("this is a test " * 100)
        self.assertEqual(len(prediction), 2)


if __name__ == '__main__':
    unittest.main()