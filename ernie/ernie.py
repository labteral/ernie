#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from transformers import (AutoTokenizer, TFAutoModelForSequenceClassification, BertTokenizer,
                          TFBertForSequenceClassification, RobertaTokenizer, TFRobertaForSequenceClassification,
                          XLNetTokenizer, TFXLNetForSequenceClassification, DistilBertTokenizer,
                          TFDistilBertForSequenceClassification)
from tensorflow import keras
from sklearn.model_selection import train_test_split
from os import makedirs
import time
from shutil import rmtree
import logging

from .models import Models, ModelsByFamily, ModelFamilyNames
from .split_strategies import SplitStrategy, SplitStrategies, RegexExpressions
from .aggregation_strategies import AggregationStrategy, AggregationStrategies
from .helper import get_features, softmax


class SentenceClassifier:
    def __init__(self,
                 model_name=Models.BertBaseUncased,
                 model_path=None,
                 max_length=128,
                 labels_no=2,
                 tokenizer_kwargs=None,
                 model_kwargs=None):
        self._loaded_data = False

        if model_kwargs is None:
            model_kwargs = {}
        model_kwargs['num_labels'] = labels_no

        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
        tokenizer_kwargs['max_len'] = max_length

        if model_path is not None:
            self._load_local_model(model_path)
        else:
            self._load_remote_model(model_name, tokenizer_kwargs, model_kwargs)

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    def load_dataset(self, dataframe=None, csv_path=None, validation_split=0.1):
        if dataframe is None and csv_path is None:
            raise ValueError

        if csv_path is not None:
            raise NotImplementedError

        sentences = list(dataframe[dataframe.columns[0]])
        labels = dataframe[dataframe.columns[1]].values

        training_sentences, validation_sentences, training_labels, validation_labels = train_test_split(
            sentences, labels, test_size=validation_split, shuffle=True)

        logging.disable(logging.WARNING)
        self._training_features = get_features(self._tokenizer, training_sentences, training_labels)
        self._training_size = len(training_sentences)

        self._validation_features = get_features(self._tokenizer, validation_sentences, validation_labels)
        self._validation_split = len(validation_sentences)
        logging.disable(logging.NOTSET)

        logging.info(f'training_size: {self._training_size}')
        logging.info(f'validation_split: {self._validation_split}')

        self._loaded_data = True

    def fine_tune(self,
                  learning_rate=2e-5,
                  epsilon=1e-8,
                  clipnorm=1.0,
                  optimizer_function=keras.optimizers.Adam,
                  optimizer_kwargs=None,
                  loss_function=keras.losses.SparseCategoricalCrossentropy,
                  loss_kwargs=None,
                  accuracy_function=keras.metrics.SparseCategoricalAccuracy,
                  accuracy_kwargs=None,
                  training_batch_size=32,
                  validation_batch_size=64,
                  **kwargs):
        if not self._loaded_data:
            raise Exception('Data has not been loaded.')

        if optimizer_kwargs is None:
            optimizer_kwargs = {'learning_rate': learning_rate, 'epsilon': epsilon, 'clipnorm': clipnorm}
        optimizer = optimizer_function(**optimizer_kwargs)

        if loss_kwargs is None:
            loss_kwargs = {'from_logits': True}
        loss = loss_function(**loss_kwargs)

        if accuracy_kwargs is None:
            accuracy_kwargs = {'name': 'accuracy'}
        accuracy = accuracy_function(**accuracy_kwargs)

        self._model.compile(optimizer=optimizer, loss=loss, metrics=[accuracy])

        training_features = self._training_features.shuffle(self._training_size).batch(training_batch_size).repeat(-1)
        validation_features = self._validation_features.batch(validation_batch_size)

        training_steps = self._training_size // training_batch_size
        if training_steps == 0:
            training_steps = self._training_size
        logging.info(f'training_steps: {training_steps}')

        validation_steps = self._validation_split // validation_batch_size
        if validation_steps == 0:
            validation_steps = self._validation_split
        logging.info(f'validation_steps: {validation_steps}')

        self._model.fit(training_features,
                        validation_data=validation_features,
                        steps_per_epoch=training_steps,
                        validation_steps=validation_steps,
                        **kwargs)

        self._reload_model()

    def predict_one(self, text, split_strategy=None, aggregation_strategy=None):
        return next(
            self.predict([text], batch_size=1, split_strategy=split_strategy,
                         aggregation_strategy=aggregation_strategy))

    def predict(self, texts, batch_size=32, split_strategy=None, aggregation_strategy=None):
        if split_strategy is None:
            yield from self._predict_batch(texts, batch_size)

        else:
            if aggregation_strategy is None:
                aggregation_strategy = AggregationStrategies.Mean

            split_indexes = [0]
            sentences = []
            for text in texts:
                logging.disable(logging.WARNING)
                new_sentences = split_strategy.split(text, self.tokenizer)
                logging.disable(logging.NOTSET)

                split_indexes.append(split_indexes[-1] + len(new_sentences))
                sentences.extend(new_sentences)

            predictions = list(self._predict_batch(sentences, batch_size))
            for i, split_index in enumerate(split_indexes[:-1]):
                stop_index = split_indexes[i + 1]
                yield aggregation_strategy.aggregate(predictions[split_index:stop_index])

    def dump(self, path):
        try:
            makedirs(path)
        except FileExistsError:
            pass
        self._model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)

    def _predict_batch(self, sentences: list, batch_size: int):
        sentences_number = len(sentences)
        if batch_size > sentences_number:
            batch_size = sentences_number

        for i in range(0, sentences_number, batch_size):
            input_ids_list = []
            attention_mask_list = []

            stop_index = i + batch_size
            stop_index = stop_index if stop_index < sentences_number else sentences_number
            for j in range(i, stop_index):
                features = self._tokenizer.encode_plus(sentences[j],
                                                       add_special_tokens=True,
                                                       max_length=self._tokenizer.max_len)
                input_ids, _, attention_mask = features['input_ids'], features['token_type_ids'], features[
                    'attention_mask']

                input_ids = self._list_to_padded_array(features['input_ids'])
                attention_mask = self._list_to_padded_array(features['attention_mask'])

                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)

            input_dict = {'input_ids': np.array(input_ids_list), 'attention_mask': np.array(attention_mask_list)}
            logit_predictions = self._model.predict_on_batch(input_dict)
            yield from ([softmax(logit_prediction) for logit_prediction in logit_predictions[0]])

    def _list_to_padded_array(self, items):
        array = np.array(items)
        padded_array = np.zeros(self._tokenizer.max_len, dtype=np.int)
        padded_array[:array.shape[0]] = array
        return padded_array

    def _load_local_model(self, path):
        self._model = TFAutoModelForSequenceClassification.from_pretrained(path)
        self._tokenizer = AutoTokenizer.from_pretrained(path)

    # The fine-tuned model does not have the same input interface after being
    # exported and loaded again. The model is reloaded just after fine tuning.
    def _reload_model(self):
        model_family = self._get_model_family()
        if model_family == ModelFamilyNames.XLNet:
            temporary_path = f'/tmp/ernie/{self.model.name}'
        else:
            temporary_path = f'/tmp/ernie/{int(round(time.time() * 1000))}'
        self.dump(temporary_path)
        self._load_local_model(temporary_path)
        # Bugfix for XLNet. After reloading the model the cache path of the
        # "spiece.model" from the tokenizer points to this temporary path.
        if model_family != ModelFamilyNames.XLNet:
            rmtree(temporary_path)

    def _load_remote_model(self, model_name, tokenizer_kwargs, model_kwargs):
        if model_name not in ModelsByFamily.Supported:
            raise ValueError(f'Model {model_name} not supported.')

        do_lower_case = False
        if 'uncased' in model_name.lower():
            do_lower_case = True
        tokenizer_kwargs.update({'do_lower_case': do_lower_case})

        self._tokenizer = None
        self._model = None

        if model_name in ModelsByFamily.Bert:
            self._tokenizer = BertTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
            self._model = TFBertForSequenceClassification.from_pretrained(model_name, **model_kwargs)
        elif model_name in ModelsByFamily.Roberta:
            self._tokenizer = RobertaTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
            self._model = TFRobertaForSequenceClassification.from_pretrained(model_name, **model_kwargs)
        elif model_name in ModelsByFamily.XLNet:
            self._tokenizer = XLNetTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
            self._model = TFXLNetForSequenceClassification.from_pretrained(model_name, **model_kwargs)
        elif model_name in ModelsByFamily.DistilBert:
            self._tokenizer = DistilBertTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
            self._model = TFDistilBertForSequenceClassification.from_pretrained(model_name, **model_kwargs)

        assert self._tokenizer and self._model

    def _get_model_family(self):
        model_family = ''.join(self._model.name[2:].split('_')[:2])
        return model_family