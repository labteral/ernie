#!/usr/bin/env python
# -*- coding: utf-8 -*-

from statistics import mean


class AggregationStrategy:
    def __init__(self, method, max_items=None, top_items=True, sorting_class_index=1):
        self.method = method
        self.max_items = max_items
        self.top_items = top_items
        self.sorting_class_index = sorting_class_index

    def aggregate(self, softmax_tuples):
        softmax_dicts = []
        for softmax_tuple in softmax_tuples:
            softmax_dict = {}
            for i, probability in enumerate(softmax_tuple):
                softmax_dict[i] = probability
            softmax_dicts.append(softmax_dict)

        if self.max_items is not None:
            softmax_dicts = sorted(softmax_dicts, key=lambda x: x[self.sorting_class_index], reverse=self.top_items)
            if self.max_items < len(softmax_dicts):
                softmax_dicts = softmax_dicts[:self.max_items]

        softmax_list = []
        for key in softmax_dicts[0].keys():
            softmax_list.append(self.method([probabilities[key] for probabilities in softmax_dicts]))
        softmax_tuple = tuple(softmax_list)
        return softmax_tuple


class AggregationStrategies:
    Mean = AggregationStrategy(method=mean)
    MeanTopFiveBinaryClassification = AggregationStrategy(method=mean,
                                                          max_items=5,
                                                          top_items=True,
                                                          sorting_class_index=1)
    MeanTopTenBinaryClassification = AggregationStrategy(method=mean,
                                                         max_items=10,
                                                         top_items=True,
                                                         sorting_class_index=1)
    MeanTopFifteenBinaryClassification = AggregationStrategy(method=mean,
                                                             max_items=15,
                                                             top_items=True,
                                                             sorting_class_index=1)
    MeanTopTwentyBinaryClassification = AggregationStrategy(method=mean,
                                                            max_items=20,
                                                            top_items=True,
                                                            sorting_class_index=1)