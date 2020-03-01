#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pandas as pd
from shutil import rmtree
from ernie import SentenceClassifier, Models


class TestErnie(unittest.TestCase):
    df = pd.DataFrame([("This is a positive example. I'm very happy today.", 1),
                       ("This is a negative sentence. Everything was wrong today at work.", 0)])
    sentence = "Oh, that's great!"

    def test_dump_and_load_bert(self):
        self._test_dump_and_load_model(Models.BertBaseUncased)

    def test_dump_and_load_roberta(self):
        self._test_dump_and_load_model(Models.RobertaBaseCased)

    def test_dump_and_load_xlnet(self):
        self._test_dump_and_load_model(Models.XLNetBaseCased)

    def test_dump_and_load_distilbert(self):
        self._test_dump_and_load_model(Models.DistilBertBaseUncased)

    def _test_dump_and_load_model(self, model_name):
        classifier = SentenceClassifier(model_name=model_name)
        classifier.load_dataset(self.df)
        classifier.fine_tune(epochs=1)

        first_prediction = classifier.predict_one(self.sentence)

        classifier_path = f'/tmp/ernie/test/{model_name}'
        classifier.dump(classifier_path)
        classifier = SentenceClassifier(model_path=classifier_path)
        rmtree(classifier_path)

        second_prediction = classifier.predict_one(self.sentence)

        self.assertEqual(first_prediction, second_prediction)


if __name__ == '__main__':
    unittest.main()