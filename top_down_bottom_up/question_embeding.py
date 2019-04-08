# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from config.config import cfg

from pytorch_pretrained_bert.modeling import BertModel#, BertPreTrainedModel, BertConfig


def build_question_encoding_module(method, par, num_vocab):
    if method == "default_que_embed":
        return QuestionEmbeding(num_vocab, **par)
    elif method == "att_que_embed":
        return AttQuestionEmbedding(num_vocab, **par)
    elif method == "bert_embed":
        # config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        # num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
        return BertForVQA(**par)
    else:
        raise NotImplementedError(
            "unknown question encoding model %s" % method)

class BertForVQA(nn.Module):
  def __init__(self, **kwargs):
    super(BertForVQA, self).__init__()
    self.text_out_dim = kwargs['q_embedding_size']
    self.bert = BertModel.from_pretrained(kwargs['pretrained_bert_dir'])
    self.dropout = nn.Dropout(kwargs['hidden_dropout_prob'])
    self.transform = nn.Linear(kwargs['hidden_size'], self.text_out_dim)
    # self.apply(self.init_bert_weights)

  def forward(self, input_text, bert_ids, attention_mask=None):
    sequence_output, _ = self.bert(bert_ids, token_type_ids=None, attention_mask=attention_mask, output_all_encoded_layers=False)
    first_token_tensor = sequence_output[:, 0]
    first_token_tensor = F.normalize(first_token_tensor, p=2, dim=1)
    # print("bert sentence embedding: ", first_token_tensor)
    pooled_output = self.dropout(first_token_tensor)
    q_embedding = self.transform(pooled_output)
    return q_embedding

class QuestionEmbeding(nn.Module):
    def __init__(self, num_vocab, **kwargs):
        super(QuestionEmbeding, self).__init__()
        self.text_out_dim = kwargs['LSTM_hidden_size']
        #self.num_vocab = kwargs['num_vocab']
        self.embedding_dim = kwargs['embedding_dim']
        self.embedding = nn.Embedding(
            num_vocab, kwargs['embedding_dim'])
        self.gru = nn.GRU(
            input_size=kwargs['embedding_dim'],
            hidden_size=kwargs['LSTM_hidden_size'],
            num_layers=kwargs['LSTM_layer'],
            dropout=kwargs['dropout'],
            batch_first=True)
        self.batch_first = True

        if 'embedding_init' in kwargs and kwargs['embedding_init'] is not None:
            self.embedding.weight.data.copy_(
                torch.from_numpy(kwargs['embedding_init']))

    def forward(self, input_text, **kwargs):
        embeded_txt = self.embedding(input_text)
        out, hidden_state = self.gru(embeded_txt)
        res = out[:, -1]
        return res


class AttQuestionEmbedding(nn.Module):
    def __init__(self, num_vocab, **kwargs):
        super(AttQuestionEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_vocab, kwargs['embedding_dim'])
        self.LSTM = nn.LSTM(input_size=kwargs['embedding_dim'],
                            hidden_size=kwargs['LSTM_hidden_size'],
                            num_layers=kwargs['LSTM_layer'],
                            batch_first=True)
        self.Dropout = nn.Dropout(p=kwargs['dropout'])
        self.conv1 = nn.Conv1d(
            in_channels=kwargs['LSTM_hidden_size'],
            out_channels=kwargs['conv1_out'],
            kernel_size=kwargs['kernel_size'],
            padding=kwargs['padding'])
        self.conv2 = nn.Conv1d(
            in_channels=kwargs['conv1_out'],
            out_channels=kwargs['conv2_out'],
            kernel_size=kwargs['kernel_size'],
            padding=kwargs['padding'])
        self.text_out_dim = kwargs['LSTM_hidden_size'] * kwargs['conv2_out']

        if 'embedding_init_file' in kwargs \
                and kwargs['embedding_init_file'] is not None:
            if os.path.isabs(kwargs['embedding_init_file']):
                embedding_file = kwargs['embedding_init_file']
            else:
                embedding_file = os.path.join(
                    cfg.data.data_root_dir, kwargs['embedding_init_file'])
            embedding_init = np.load(embedding_file)
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_init))

    def forward(self, input_text, **kwargs):
        batch_size, _ = input_text.data.shape
        embed_txt = self.embedding(input_text)          # N * T * embedding_dim

        # self.LSTM.flatten_parameters()
        lstm_out, _ = self.LSTM(embed_txt)  # N * T * LSTM_hidden_size
        lstm_drop = self.Dropout(lstm_out)  # N * T * LSTM_hidden_size
        lstm_reshape = lstm_drop.permute(0, 2, 1)  # N * LSTM_hidden_size * T

        qatt_conv1 = self.conv1(lstm_reshape)  # N x conv1_out x T
        qatt_relu = F.relu(qatt_conv1)
        qatt_conv2 = self.conv2(qatt_relu)  # N x conv2_out x T

        qtt_softmax = F.softmax(qatt_conv2, dim=2)
        # N * conv2_out * LSTM_hidden_size
        qtt_feature = torch.bmm(qtt_softmax, lstm_drop)
        # N * (conv2_out * LSTM_hidden_size)
        qtt_feature_concat = qtt_feature.view(batch_size, -1)

        return qtt_feature_concat
