# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from config.collections import AttrDict

# --------------------------------------------------------------------------- #
# optimizer options:
# --------------------------------------------------------------------------- #

Adamax_par = AttrDict()
Adamax_par.lr = 0.01
Adamax_par.weight_decay = 0
Adamax_par.eps = 0.00000001

OPTIMIZER = {
    'Adamax': Adamax_par
}


# --------------------------------------------------------------------------- #
# loss options:
# --------------------------------------------------------------------------- #

LOSS = {
    'logit': AttrDict(),
    'softmax': AttrDict()
}


# --------------------------------------------------------------------------- #
# image feature encoding options:
# --------------------------------------------------------------------------- #
finetune_faster_rcnn_fpn_fc7 = AttrDict()
finetune_faster_rcnn_fpn_fc7.weights_file = ""
finetune_faster_rcnn_fpn_fc7.bias_file = ""

default_feature = AttrDict()

IMAGE_FEATURE_MODEL = {
    "finetune_faster_rcnn_fpn_fc7": finetune_faster_rcnn_fpn_fc7,
    "default": default_feature
}


# --------------------------------------------------------------------------- #
# question embedding model options:
# --------------------------------------------------------------------------- #

# Attention question embedding model
att_que_embed = AttrDict()
att_que_embed.embedding_dim = 300
att_que_embed.LSTM_hidden_size = 1024
att_que_embed.LSTM_layer = 1
att_que_embed.dropout = 0
att_que_embed.conv1_out = 512
att_que_embed.conv2_out = 2
att_que_embed.kernel_size = 1
att_que_embed.padding = 0
att_que_embed.embedding_init_file = "vqa2.0_glove.6B.300d.txt.npy"

# GRU
default_que_embed = AttrDict()
default_que_embed.embedding_dim = 300
default_que_embed.LSTM_hidden_size = 1024
default_que_embed.LSTM_layer = 1
default_que_embed.dropout = 0
default_que_embed.embedding_init_file = "vqa2.0_glove.6B.300d.txt.npy"

# BERT
bert_embed = AttrDict()
bert_embed.attention_probs_dropout_prob = 0.1
bert_embed.hidden_act = "gelu"
bert_embed.hidden_dropout_prob = 0.1
bert_embed.hidden_size = 768
bert_embed.initializer_range = 0.02
bert_embed.intermediate_size = 3072
bert_embed.max_position_embeddings = 512
bert_embed.num_attention_heads = 12
bert_embed.num_hidden_layers = 12
bert_embed.type_vocab_size = 2
bert_embed.vocab_size = 30522
bert_embed.q_embedding_size = 2048
bert_embed.pretrained_bert_dir = '/u/lxiao/VQA/pythia/pretrained_bert'

QUESTION_MODEL = {
    "att_que_embed": att_que_embed,
    "default_que_embed": default_que_embed,
    "bert_embed": bert_embed
}


# --------------------------------------------------------------------------- #
# modal combine options:
# --------------------------------------------------------------------------- #

non_linear_elmt_multiply = AttrDict()
non_linear_elmt_multiply.hidden_size = 5000
non_linear_elmt_multiply.dropout = 0

MFH = AttrDict()

MFH.order = 1
MFH.hidden_sizes = [5000]
MFH.dropout = 0.1
MFH.pool_size = 5

MODAL_COMBINE = {
    'non_linear_elmt_multiply': non_linear_elmt_multiply,
    "MFH": MFH
}


# --------------------------------------------------------------------------- #
# linear_transform options
# --------------------------------------------------------------------------- #
linear_transform = AttrDict()
linear_transform.out_dim = 1

conv_transform = AttrDict()
conv_transform.out_dim = 2
conv_transform.hidden_dim = 512


# --------------------------------------------------------------------------- #
# logit classifier
# --------------------------------------------------------------------------- #

logit_classifier = AttrDict()
logit_classifier.txt_hidden_dim = 300
logit_classifier.img_hidden_dim = 5000


# --------------------------------------------------------------------------- #
# weigt_norm_classifier
# --------------------------------------------------------------------------- #
weight_norm_classifier = AttrDict()
weight_norm_classifier.hidden_dim = 5000
weight_norm_classifier.dropout = 0.1


# --------------------------------------------------------------------------- #
# linear_classifier
# --------------------------------------------------------------------------- #
linear_classifier = AttrDict()


# --------------------------------------------------------------------------- #
# Adamax optimizer
# --------------------------------------------------------------------------- #
adamax_opt = AttrDict()
adamax_opt.lr = 0.01
adamax_opt.weight_decay = 0
adamax_opt.eps = 0.00000001

# SUMMARY of all model parameters
MODEL_TYPE_PAR_DICT = {
    'linear_transform': linear_transform,
    'conv_transform': conv_transform,
    'non_linear_elmt_multiply': MODAL_COMBINE['non_linear_elmt_multiply'],
    'MFH': MODAL_COMBINE['MFH'],
    'att_que_embed': QUESTION_MODEL['att_que_embed'],
    'bert_embed': QUESTION_MODEL['bert_embed'],
    'logit_classifier': logit_classifier,
    'Adamax': adamax_opt,
    'default_image': IMAGE_FEATURE_MODEL['default'],
    'finetune_faster_rcnn_fpn_fc7':
        IMAGE_FEATURE_MODEL['finetune_faster_rcnn_fpn_fc7'],
    'weight_norm_classifier': weight_norm_classifier,
    'linear_classifier': linear_classifier
}


class ModelParPair(AttrDict):

    IMMUTABLE = '__immutable__'

    def __init__(self, model_type):
        super(ModelParPair, self).__init__()

        self.method = model_type
        if self.method not in MODEL_TYPE_PAR_DICT:
            exit("unkown model type %s, please check \
                 config/function_config_lib.py for allowed options")
        self.par = MODEL_TYPE_PAR_DICT[self.method]

    def update_type(self, updated_pair_type):
        if updated_pair_type != self.method:
            self.method = updated_pair_type
            self.par = MODEL_TYPE_PAR_DICT[self.method]

    def is_immutable(self):
        return self.__dict__[ModelParPair.IMMUTABLE]
