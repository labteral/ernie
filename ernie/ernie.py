#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from transformers import *
from sklearn.model_selection import train_test_split
from math import exp


class Models:
    BertBaseUncased = 'bert-base-uncased'
    BertBaseCased = 'bert-base-cased'
    BertLargeUncased = 'bert-large-uncased'
    BertLargeCased = 'bert-large-cased'

    RobertaBaseCased = 'roberta-base'
    RobertaLargeCased = 'roberta-large'

    XLNetBaseCased = 'xlnet-base-cased'
    XLNetLargeCased = 'xlnet-large-cased'

    DistilBertBaseUncased = 'distilbert-base-uncased'
    DistilBertBaseMultilingualCased = 'distilbert-base-multilingual-cased'


class ModelFamilies:
    Bert = set([Models.BertBaseUncased, Models.BertBaseCased, Models.BertLargeUncased, Models.BertLargeCased])
    Roberta = set([Models.RobertaBaseCased, Models.RobertaLargeCased])
    XLNet = set([Models.XLNetBaseCased, Models.XLNetLargeCased])
    DistilBert = set([Models.DistilBertBaseUncased, Models.DistilBertBaseMultilingualCased])
    Supported = set(
        [getattr(Models, model_type) for model_type in filter(lambda x: x[:2] != '__', Models.__dict__.keys())])


def get_features(tokenizer, sentences, max_length, labels=None):
    features = []
    for i, sentence in enumerate(sentences):
        inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=max_length)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        padding_length = max_length - len(input_ids)

        if tokenizer.padding_side == 'right':
            attention_mask = [1] * len(input_ids) + [0] * padding_length
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            token_type_ids = token_type_ids + [tokenizer.pad_token_type_id] * padding_length
        else:
            attention_mask = [0] * padding_length + [1] * len(input_ids)
            input_ids = [tokenizer.pad_token_id] * padding_length + input_ids
            token_type_ids = [tokenizer.pad_token_type_id] * padding_length + token_type_ids

        assert max_length == len(attention_mask) == len(input_ids) == len(token_type_ids)

        feature = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label': int(labels[i]) if labels is not None else -1
        }

        features.append(feature)

    def gen():
        for feature in features:
            yield (
                {
                    'input_ids': feature['input_ids'],
                    'attention_mask': feature['attention_mask'],
                    'token_type_ids': feature['token_type_ids'],
                },
                feature['label'],
            )

    tf_dataset = tf.data.Dataset.from_generator(
        gen,
        ({
            'input_ids': tf.int32,
            'attention_mask': tf.int32,
            'token_type_ids': tf.int32
        }, tf.int64),
        (
            {
                'input_ids': tf.TensorShape([None]),
                'attention_mask': tf.TensorShape([None]),
                'token_type_ids': tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )
    return tf_dataset


def softmax(values):
    exps = [exp(value) for value in values]
    exps_sum = sum(exp_value for exp_value in exps)
    return tuple(map(lambda x: x / exps_sum, exps))


class BinaryClassifier:
    def __init__(self,
                 model=Models.BertBaseUncased,
                 max_length=128,
                 learning_rate=2e-5,
                 epsilon=1e-8,
                 clipnorm=1.0,
                 optimizer_function=tf.keras.optimizers.Adam,
                 optimizer_kwargs=None,
                 loss_function=tf.keras.losses.SparseCategoricalCrossentropy,
                 loss_kwargs=None,
                 accuracy_function=tf.keras.metrics.SparseCategoricalAccuracy,
                 accuracy_kwargs=None,
                 model_path=None):
        self._loaded_data = False

        if model_path is not None:
            # self._model =
            raise NotImplementedError
        else:
            if model not in ModelFamilies.Supported:
                # AutoTokenizer, AutoModel
                raise NotImplementedError
            self._tokenizer, self._model = self._get_model_items(model)

        self._max_length = max_length

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

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    def load_dataset(self, dataframe=None, csv_path=None, validation_split=0.1):
        if dataframe is None and csv_path is None:
            raise ValueError

        if dataframe is not None:
            sentences = list(dataframe[0])
            labels = dataframe[1].values

        elif csv_path is not None:
            raise NotImplementedError

        training_sentences, validation_sentences, training_labels, validation_labels = train_test_split(
            sentences, labels, test_size=validation_split, shuffle=True)

        self._training_features = self._get_features(training_sentences, training_labels)
        self._training_size = len(training_sentences)
        print(f'training_size: {self._training_size}')

        self._validation_features = self._get_features(validation_sentences, validation_labels)
        self._validation_split = len(validation_sentences)
        print(f'validation_split: {self._validation_split}')

        self._loaded_data = True

    def fine_tune(self, training_batch_size=32, validation_batch_size=64, **kwargs):
        if not self._loaded_data:
            return

        training_features = self._training_features.shuffle(self._training_size).batch(training_batch_size).repeat(-1)
        validation_features = self._validation_features.batch(validation_batch_size)

        training_steps = self._training_size // training_batch_size
        if training_steps == 0:
            training_steps = self._training_size
        print(f'training_steps: {training_steps}')

        validation_steps = self._validation_split // validation_batch_size
        if validation_steps == 0:
            validation_steps = self._validation_split
        print(f'validation_steps: {validation_steps}')

        self._model.fit(training_features,
                        validation_data=validation_features,
                        steps_per_epoch=training_steps,
                        validation_steps=validation_steps,
                        **kwargs)

    def predict(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]
        sentences_no = len(sentences)

        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []

        for sentence in sentences:
            features = self._tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=self._max_length)
            input_ids, token_type_ids, attention_mask = features['input_ids'], features['token_type_ids'], features[
                'attention_mask']

            input_ids = self._list_to_padded_array(features['input_ids'])
            token_type_ids = self._list_to_padded_array(features['token_type_ids'])
            attention_mask = self._list_to_padded_array(features['attention_mask'])

            input_ids_list.append(input_ids)
            token_type_ids_list.append(token_type_ids)
            attention_mask_list.append(attention_mask)

        logit_predictions = self._model.predict_on_batch(
            [np.array(attention_mask_list),
             np.array(input_ids_list),
             np.array(token_type_ids_list)])
        softmax_predictions = tuple([softmax(logit_prediction) for logit_prediction in logit_predictions[0]])

        if len(softmax_predictions) == 1:
            return softmax_predictions[0]
        return softmax_predictions

    def dump(self, path):
        raise NotImplementedError

    def _get_features(self, sentences, labels=None):
        features = get_features(tokenizer=self._tokenizer,
                                sentences=sentences,
                                max_length=self._max_length,
                                labels=labels)
        return features

    def _list_to_padded_array(self, items):
        array = np.array(items)
        padded_array = np.zeros(self._max_length, dtype=np.int)
        padded_array[:array.shape[0]] = array
        return padded_array

    @staticmethod
    def _get_model_items(model_name):
        do_lower_case = False
        if 'uncased' in model_name.lower():
            do_lower_case = True

        tokenizer = None
        model = None

        if model_name in ModelFamilies.Bert:
            tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
            model = TFBertForSequenceClassification.from_pretrained(model_name)
        elif model_name in ModelFamilies.Roberta:
            tokenizer = RobertaTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
            model = TFRobertaForSequenceClassification.from_pretrained(model_name)
        elif model_name in ModelFamilies.XLNet:
            tokenizer = XLNetTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
            model = TFXLNetForSequenceClassification.from_pretrained(model_name)
        elif model_name in ModelFamilies.DistilBert:
            tokenizer = DistilBertTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
            model = TFDistilBertForSequenceClassification.from_pretrained(model_name)
        assert tokenizer and model

        return tokenizer, model