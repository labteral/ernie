#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pandas as pd
from shutil import rmtree
from ernie import BinaryClassifier


class TestErnie(unittest.TestCase):

    df = pd.DataFrame([("This is a positive example. I'm very happy today.", 1),
                       ("This is a negative sentence. Everything was wrong today at work.", 0)])
    sentence = "Oh, that's great!"

    def test_prediction_after_reload(self):
        classifier = BinaryClassifier()
        classifier.load_dataset(self.df)
        classifier.fine_tune()
        prediction = classifier.predict_one(self.sentence)

        classifier_path = '/tmp/ernie/test'
        classifier.dump(classifier_path)

        classifier = BinaryClassifier(model_path=classifier_path)
        rmtree(classifier_path)

        self.assertEqual(classifier.predict_one(self.sentence), prediction)


if __name__ == '__main__':
    unittest.main()