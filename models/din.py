# -*- coding:utf-8 -*-
"""
Author:
    Yukun Zheng, zyk265182@alibaba-inc.com
"""

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Dense, Concatenate, Lambda, Multiply, Add
from tensorflow.python.keras.models import Model, load_model

from deepctr.layers.core import DNN, PredictionLayer, LocalActivationUnit
from layers.layers import AttentionSequencePoolingLayer

from inputs.dataset import Dataset
from base_model import BaseModel

class DIN(BaseModel):
    def __init__(self, args):
        super(DIN, self).__init__(args)
        self.need_neighbors = False
        self.column_groups = {'user': ['user_id', 'user_gender', 'user_age'],
                              'query': ['query'],
                              'item': ['item_id', 'item_brand', 'item_seller', 'item_cate',
                                       'item_price'],
                              'hist': ['hist', 'hist_brand', 'hist_seller', 'hist_cate',  'hist_price']}
        self.initialize_model()
        self.dataset = Dataset(args,
                               train_files=args.train_files,
                               dev_files=args.dev_files,
                               test_files=args.test_files,
                               features=self.feature_columns_dict,
                               need_neighbor=self.need_neighbors)

    def create_model(self):
        group_embed_dict = self.group_embed_dict
        att_query = Concatenate(axis=-1)([group_embed_dict['user'], group_embed_dict['query'], group_embed_dict['item']])
        att_key = group_embed_dict['hist']
        att_embed_size = att_key.get_shape().as_list()[-1]
        att_query = DNN(hidden_units=[att_embed_size], activation='dice', dropout_rate=self.dropout_rate)(att_query)
        att_key_mask = Lambda(tf.not_equal, arguments={'y': 0})(self.features['hist'])
        att_item_embed = AttentionSequencePoolingLayer(att_hidden_units=(96, 24), att_activation='dice',
                                                       dropout_rate=self.dropout_rate)(inputs=[att_query, att_key], mask=[att_key_mask])

        group_embed_dict['hist'] = Lambda(K.mean, arguments={'axis': 1, 'keepdims': True})(group_embed_dict['hist'])

        '''concatenate all features for the prediction layer'''
        deep_input_emb = Concatenate(axis=-1)([group_embed_dict['user'],
                                               group_embed_dict['item'],
                                               group_embed_dict['query'],
                                               att_item_embed])
        self.deep_input_emb = Lambda(K.squeeze, arguments={'axis': 1})(deep_input_emb)


    def load_model(self, model_path):
        return load_model(model_path, custom_objects={'DNN': DNN,
                                                      'PredictionLayer': PredictionLayer,
                                                      'AttentionSequencePoolingLayer': AttentionSequencePoolingLayer,
                                                      'LocalActivationUnit': LocalActivationUnit})
