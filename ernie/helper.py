#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow import data, TensorShape, int64, int32
from math import exp


def get_features(tokenizer, sentences, labels):
    features = []
    for i, sentence in enumerate(sentences):
        inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=tokenizer.max_len)
        input_ids, token_type_ids = inputs['input_ids'], inputs['token_type_ids']
        padding_length = tokenizer.max_len - len(input_ids)

        if tokenizer.padding_side == 'right':
            attention_mask = [1] * len(input_ids) + [0] * padding_length
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            token_type_ids = token_type_ids + [tokenizer.pad_token_type_id] * padding_length
        else:
            attention_mask = [0] * padding_length + [1] * len(input_ids)
            input_ids = [tokenizer.pad_token_id] * padding_length + input_ids
            token_type_ids = [tokenizer.pad_token_type_id] * padding_length + token_type_ids

        assert tokenizer.max_len == len(attention_mask) == len(input_ids) == len(
            token_type_ids), f'{tokenizer.max_len}, {len(attention_mask)}, {len(input_ids)}, {len(token_type_ids)}'

        feature = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label': int(labels[i])
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

    dataset = data.Dataset.from_generator(
        gen,
        ({
            'input_ids': int32,
            'attention_mask': int32,
            'token_type_ids': int32
        }, int64),
        (
            {
                'input_ids': TensorShape([None]),
                'attention_mask': TensorShape([None]),
                'token_type_ids': TensorShape([None]),
            },
            TensorShape([]),
        ),
    )

    return dataset


def softmax(values):
    exps = [exp(value) for value in values]
    exps_sum = sum(exp_value for exp_value in exps)
    return tuple(map(lambda x: x / exps_sum, exps))
