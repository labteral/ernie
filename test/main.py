#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pandas as pd
from shutil import rmtree
from ernie import BinaryClassifier


class TestErnie(unittest.TestCase):
    classifier = BinaryClassifier()
    classifier.load_dataset(
        pd.DataFrame([("This is a positive example. I'm very happy today.", 1),
                      ("This is a negative sentence. Everything was wrong today at work.", 0)]))
    classifier.fine_tune()
    sentence = "Oh, that's great!"

    def test_dump_and_load(self):
        prediction = self.classifier.predict_one(self.sentence)
        self.classifier_path = '/tmp/ernie/test'
        self.classifier.dump(self.classifier_path)

        classifier = BinaryClassifier(model_path=self.classifier_path)
        rmtree(self.classifier_path)
        self.assertEqual(self.classifier.predict_one(self.sentence), prediction)


if __name__ == '__main__':
    unittest.main()