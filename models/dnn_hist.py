# -*- coding:utf-8 -*-
"""
Author:
    Yukun Zheng, zyk265182@alibaba-inc.com
"""

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Dense, Concatenate, Lambda, Multiply, Add
from tensorflow.python.keras.models import Model, load_model

from deepctr.layers.core import DNN, PredictionLayer
from inputs.dataset import Dataset
from base_model import BaseModel

class DNN_hist(BaseModel):
    def __init__(self, args):
        super(DNN_hist, self).__init__(args)
        self.need_neighbors = False
        self.column_groups = {'user': ['user_id', 'user_gender', 'user_age'],
                              'query': ['query'],
                              'item': ['item_id', 'item_brand', 'item_seller', 'item_cate', 'item_price'],
                              'hist': ['hist', 'hist_brand', 'hist_seller', 'hist_cate', 'hist_price']}
        self.initialize_model()
        self.dataset = Dataset(args,
                               train_files=args.train_files,
                               dev_files=args.dev_files,
                               test_files=args.test_files,
                               features=self.feature_columns_dict,
                               need_neighbor=self.need_neighbors)


    def create_model(self):
        group_embed_dict = self.group_embed_dict
        # group_embed_dict['hist'] = Lambda(K.mean, arguments={'axis': 1, 'keepdims': True})(group_embed_dict['hist'])

        hist_embed = group_embed_dict['hist']
        hist_mask = Lambda(K.not_equal, arguments={'y': 0})(self.features['hist'])
        hist_mask = Lambda(K.cast, arguments={'dtype': 'float32'})(hist_mask)
        hist_len = Lambda(tf.reduce_sum, arguments={'axis': 1, 'keepdims': True})(hist_mask)

        hist_mask = Lambda(K.reshape, arguments={'shape': [-1, self.max_hist_len, 1]})(hist_mask)
        hist_mask = Lambda(tf.tile, arguments={'multiples': [1, 1, hist_embed.get_shape().as_list()[-1]]})(hist_mask)
        hist_embed = Multiply()([hist_embed, hist_mask])
        hist_embed_sum = Lambda(K.sum, arguments={'axis': 1, 'keepdims': True})(hist_embed)

        hist_len = Lambda(K.reshape, arguments={'shape': [-1, 1, 1]})(hist_len)
        hist_len = Lambda(tf.tile, arguments={'multiples': [1, 1, hist_embed.get_shape().as_list()[-1]]})(hist_len)
        hist_len = Lambda(lambda x:x+1e-6)(hist_len)
        hist_embed_mean = Lambda(lambda inputs: inputs[0] / inputs[1])([hist_embed_sum, hist_len])
        # mul_item_embed = Multiply()([group_embed_dict['item'], hist_embed_mean])

        '''concatenate all features for the prediction layer'''
        deep_input_emb = Concatenate(axis=-1)([group_embed_dict['user'],
                                               group_embed_dict['item'],
                                               group_embed_dict['query'],
                                               hist_embed_mean])
        self.deep_input_emb = Lambda(K.squeeze, arguments={'axis': 1})(deep_input_emb)


    def load_model(self, model_path):
        return load_model(model_path, custom_objects={'DNN': DNN,
                                                      'PredictionLayer': PredictionLayer})
