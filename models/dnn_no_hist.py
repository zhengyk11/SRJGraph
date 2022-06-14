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


class DNN_no_hist(BaseModel):
    def __init__(self, args):
        super(DNN_no_hist, self).__init__(args)
        self.need_neighbors = False
        self.column_groups = {'user': ['user_id', 'user_gender', 'user_age'],
                              'query': ['query'],
                              'item': ['item_id', 'item_brand', 'item_seller', 'item_cate', 'item_price']}
        self.initialize_model()
        self.dataset = Dataset(args,
                               train_files=args.train_files,
                               dev_files=args.dev_files,
                               test_files=args.test_files,
                               features=self.feature_columns_dict,
                               need_neighbor=self.need_neighbors)

    def create_model(self):
        '''concatenate all features for the prediction layer'''
        deep_input_emb = Concatenate(axis=-1)(self.group_embed_dict.values())
        self.deep_input_emb = Lambda(K.squeeze, arguments={'axis': 1})(deep_input_emb)


    def load_model(self, model_path):
        return load_model(model_path, custom_objects={'DNN': DNN,
                                                      'PredictionLayer': PredictionLayer})
