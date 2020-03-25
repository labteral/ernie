#!/usr/bin/env python
# -*- coding: utf-8 -*-


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

    AlbertBaseCased = 'albert-base-v1'
    AlbertLargeCased = 'albert-large-v1'
    AlbertXLargeCased = 'albert-xlarge-v1'
    AlbertXXLargeCased = 'albert-xxlarge-v1'

    AlbertBaseCased2 = 'albert-base-v2'
    AlbertLargeCased2 = 'albert-large-v2'
    AlbertXLargeCased2 = 'albert-xlarge-v2'
    AlbertXXLargeCased2 = 'albert-xxlarge-v2'


class ModelsByFamily:
    Bert = set([Models.BertBaseUncased, Models.BertBaseCased, Models.BertLargeUncased, Models.BertLargeCased])
    Roberta = set([Models.RobertaBaseCased, Models.RobertaLargeCased])
    XLNet = set([Models.XLNetBaseCased, Models.XLNetLargeCased])
    DistilBert = set([Models.DistilBertBaseUncased, Models.DistilBertBaseMultilingualCased])
    Albert = set([
        Models.AlbertBaseCased, Models.AlbertLargeCased, Models.AlbertXLargeCased, Models.AlbertXXLargeCased,
        Models.AlbertBaseCased2, Models.AlbertLargeCased2, Models.AlbertXLargeCased2, Models.AlbertXXLargeCased2
    ])
    Supported = set(
        [getattr(Models, model_type) for model_type in filter(lambda x: x[:2] != '__', Models.__dict__.keys())])