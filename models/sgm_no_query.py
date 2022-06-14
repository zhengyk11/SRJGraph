# -*- coding:utf-8 -*-
"""
Author:
    Yukun Zheng, zyk265182@alibaba-inc.com
"""
import numpy as np

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Dense, Concatenate, Lambda, Multiply, Add
from tensorflow.python.keras.models import Model, load_model

from deepctr.layers.core import DNN, PredictionLayer
from base_model import BaseModel
from inputs.dataset import Dataset
from layers.layers import AggregationLayer, Transformer


class SGM_no_query(BaseModel):
    def __init__(self, args):
        super(SGM_no_query, self).__init__(args)
        self.need_neighbors = True
        self.column_groups = {'user': ['user_id', 'user_gender', 'user_age'],
                              'query': ['query'],
                              'item': ['item_id', 'item_brand', 'item_seller', 'item_cate', 'item_price']}
        if len(self.user_neighbor_nums) >= 1 and np.prod(self.user_neighbor_nums[:1]) > 0:
            self.column_groups.update({'ui': ['ui', 'ui_brand', 'ui_seller', 'ui_cate', 'ui_price'], 'ui_query': ['ui_query']})
        if len(self.user_neighbor_nums) >= 2 and np.prod(self.user_neighbor_nums[:2]) > 0:
            self.column_groups.update({'uiu': ['uiu', 'uiu_gender', 'uiu_age'], 'uiu_query': ['uiu_query']})
        if len(self.user_neighbor_nums) >= 3 and np.prod(self.user_neighbor_nums[:3]) > 0:
            self.column_groups.update({'uiui': ['uiui', 'uiui_brand', 'uiui_seller', 'uiui_cate', 'uiui_price'], 'uiui_query': ['uiui_query']})
        if len(self.user_neighbor_nums) >= 4 and np.prod(self.user_neighbor_nums[:4]) > 0:
            self.column_groups.update({'uiuiu': ['uiuiu', 'uiuiu_gender', 'uiuiu_age'], 'uiuiu_query': ['uiuiu_query']})
        if len(self.item_neighbor_nums) >= 1 and np.prod(self.item_neighbor_nums[:1]) > 0:
            self.column_groups.update({'iu': ['iu', 'iu_gender', 'iu_age'], 'iu_query': ['iu_query']})
        if len(self.item_neighbor_nums) >= 2 and np.prod(self.item_neighbor_nums[:2]) > 0:
            self.column_groups.update({'iui': ['iui', 'iui_brand', 'iui_seller', 'iui_cate', 'iui_price'], 'iui_query': ['iui_query']})
        if len(self.item_neighbor_nums) >= 3 and np.prod(self.item_neighbor_nums[:3]) > 0:
            self.column_groups.update({'iuiu': ['iuiu', 'iuiu_gender', 'iuiu_age'], 'iuiu_query': ['iuiu_query']})
        if len(self.item_neighbor_nums) >= 4 and np.prod(self.item_neighbor_nums[:4]) > 0:
            self.column_groups.update({'iuiui': ['iuiui', 'iuiui_brand', 'iuiui_seller', 'iuiui_cate', 'iuiui_price'], 'iuiui_query': ['iuiui_query']})

        self.initialize_model()
        self.dataset = Dataset(args,
                               train_files=args.train_files,
                               dev_files=args.dev_files,
                               test_files=args.test_files,
                               features=self.feature_columns_dict,
                               need_neighbor=self.need_neighbors)


    def create_model(self):
        group_embed_dict = self.group_embed_dict
        target_uqi = Concatenate(axis=2)([group_embed_dict['user'], Lambda(tf.zeros_like)(group_embed_dict['query']), group_embed_dict['item']])

        # uqi_embed_size = target_uqi.get_shape().as_list()[-1]
        # user_embed_size = group_embed_dict['user'].get_shape().as_list()[-1]
        # item_embed_size = group_embed_dict['item'].get_shape().as_list()[-1]
        # query_embed_size = group_embed_dict['query'].get_shape().as_list()[-1]

        agg_layer_u2i = AggregationLayer(self.dropout_rate)
        agg_layer_i2u = AggregationLayer(self.dropout_rate)
        agg_user_neighbor_nums = [1] + self.user_neighbor_nums
        agg_item_neighbor_nums = [1] + self.item_neighbor_nums
        
        agg_layer_outs = {}
        '''u--i--u--i<--u'''
        if 'uiuiu' in group_embed_dict:
            inputs = [target_uqi, self.group_embed_dict['uiu'],
                      self.features['uiui'], self.group_embed_dict['uiui'], Lambda(tf.zeros_like)(self.group_embed_dict['uiui_query']),
                      self.features['uiuiu']]
            if 'uiuiu' in agg_layer_outs:
                inputs += [agg_layer_outs['uiuiu']]
                is_leaf = False
            else:
                inputs += [self.group_embed_dict['uiuiu'], Lambda(tf.zeros_like)(self.group_embed_dict['uiuiu_query'])]
                is_leaf = True
            self.logger.info('aggregating u--i--u--i<--u')
            agg_layer_outs['uiui'] = agg_layer_u2i(inputs, is_leaf=is_leaf, agg_type='u2i', order=4, neighbor_nums=agg_user_neighbor_nums)
            
        '''u--i--u<--i--u'''
        if 'uiui' in group_embed_dict:
            inputs = [target_uqi, self.group_embed_dict['ui'],
                      self.features['uiu'], self.group_embed_dict['uiu'], Lambda(tf.zeros_like)(self.group_embed_dict['uiu_query']),
                      self.features['uiui']]
            if 'uiui' in agg_layer_outs:
                inputs += [agg_layer_outs['uiui']]
                is_leaf = False
            else:
                inputs += [self.group_embed_dict['uiui'], Lambda(tf.zeros_like)(self.group_embed_dict['uiui_query'])]
                is_leaf = True
            self.logger.info('aggregating u--i--u<--i--u')
            agg_layer_outs['uiu'] = agg_layer_i2u(inputs, is_leaf=is_leaf, agg_type='i2u', order=3, neighbor_nums=agg_user_neighbor_nums)

        '''u--i<--u--i--u'''
        if 'uiu' in group_embed_dict:
            inputs = [target_uqi, self.group_embed_dict['user'],
                      self.features['ui'], self.group_embed_dict['ui'], Lambda(tf.zeros_like)(self.group_embed_dict['ui_query']),
                      self.features['uiu']]
            if 'uiu' in agg_layer_outs:
                inputs += [agg_layer_outs['uiu']]
                is_leaf = False
            else:
                inputs += [self.group_embed_dict['uiu'], Lambda(tf.zeros_like)(self.group_embed_dict['uiu_query'])]
                is_leaf = True
            self.logger.info('aggregating u--i<--u--i--u')
            agg_layer_outs['ui'] = agg_layer_u2i(inputs, is_leaf=is_leaf, agg_type='u2i', order=2, neighbor_nums=agg_user_neighbor_nums)

        '''u<--i--u--i--u'''
        if 'ui' in group_embed_dict:
            inputs = [target_uqi, self.group_embed_dict['item'],
                      self.features['user_id'], self.group_embed_dict['user'], Lambda(tf.zeros_like)(self.group_embed_dict['query']),
                      self.features['ui']]
            if 'ui' in agg_layer_outs:
                inputs += [agg_layer_outs['ui']]
                is_leaf = False
            else:
                inputs += [self.group_embed_dict['ui'], Lambda(tf.zeros_like)(self.group_embed_dict['ui_query'])]
                is_leaf = True
            self.logger.info('aggregating u<--i--u--i--u')
            agg_layer_outs['u'] = agg_layer_i2u(inputs, is_leaf=is_leaf, agg_type='i2u', order=1, neighbor_nums=agg_user_neighbor_nums)

        '''i--u--i--u<--i'''
        if 'iuiui' in group_embed_dict:
            inputs = [target_uqi, self.group_embed_dict['iui'],
                      self.features['iuiu'], self.group_embed_dict['iuiu'], Lambda(tf.zeros_like)(self.group_embed_dict['iuiu_query']),
                      self.features['iuiui']]
            if 'iuiui' in agg_layer_outs:
                inputs += [agg_layer_outs['iuiui']]
                is_leaf = False
            else:
                inputs += [self.group_embed_dict['iuiui'], Lambda(tf.zeros_like)(self.group_embed_dict['iuiui_query'])]
                is_leaf = True
            self.logger.info('aggregating i--u--i--u<--i')
            agg_layer_outs['iuiu'] = agg_layer_i2u(inputs, is_leaf=is_leaf, agg_type='i2u', order=4, neighbor_nums=agg_item_neighbor_nums)

        '''i--u--i<--u--i'''
        if 'iuiu' in group_embed_dict:
            inputs = [target_uqi, self.group_embed_dict['iu'],
                      self.features['iui'], self.group_embed_dict['iui'], Lambda(tf.zeros_like)(self.group_embed_dict['iui_query']),
                      self.features['iuiu']]
            if 'iuiu' in agg_layer_outs:
                inputs += [agg_layer_outs['iuiu']]
                is_leaf = False
            else:
                inputs += [self.group_embed_dict['iuiu'], Lambda(tf.zeros_like)(self.group_embed_dict['iuiu_query'])]
                is_leaf = True
            self.logger.info('aggregating i--u--i<--u--i')
            agg_layer_outs['iui'] = agg_layer_u2i(inputs, is_leaf=is_leaf, agg_type='u2i', order=3, neighbor_nums=agg_item_neighbor_nums)

        '''i--u<--i--u--i'''
        if 'iui' in group_embed_dict:
            inputs = [target_uqi, self.group_embed_dict['item'],
                      self.features['iu'], self.group_embed_dict['iu'], Lambda(tf.zeros_like)(self.group_embed_dict['iu_query']),
                      self.features['iui']]
            if 'iui' in agg_layer_outs:
                inputs += [agg_layer_outs['iui']]
                is_leaf = False
            else:
                inputs += [self.group_embed_dict['iui'], Lambda(tf.zeros_like)(self.group_embed_dict['iui_query'])]
                is_leaf = True
            self.logger.info('aggregating i--u<--i--u--i')
            agg_layer_outs['iu'] = agg_layer_i2u(inputs, is_leaf=is_leaf, agg_type='i2u', order=2, neighbor_nums=agg_item_neighbor_nums)

        '''i<--u--i--u--i'''
        if 'iu' in group_embed_dict:
            inputs = [target_uqi, self.group_embed_dict['user'],
                      self.features['item_id'], self.group_embed_dict['item'], Lambda(tf.zeros_like)(self.group_embed_dict['query']),
                      self.features['iu']]
            if 'iu' in agg_layer_outs:
                inputs += [agg_layer_outs['iu']]
                is_leaf = False
            else:
                inputs += [self.group_embed_dict['iu'], Lambda(tf.zeros_like)(self.group_embed_dict['iu_query'])]
                is_leaf = True
            self.logger.info('aggregating i<--u--i--u--i')
            agg_layer_outs['i'] = agg_layer_u2i(inputs, is_leaf=is_leaf, agg_type='u2i', order=1, neighbor_nums=agg_item_neighbor_nums)

        deep_input_emb = [group_embed_dict['user'], group_embed_dict['item'], group_embed_dict['query']]
        if 'u' in agg_layer_outs:
            self.logger.info('adding agg layer outs ["u"] into deep_input_emb')
            deep_input_emb.append(agg_layer_outs['u'])
        if 'i' in agg_layer_outs:
            deep_input_emb.append(agg_layer_outs['i'])
            self.logger.info('adding agg layer outs ["i"] into deep_input_emb')
        deep_input_emb = Concatenate(axis=-1)(deep_input_emb)
        self.deep_input_emb = Lambda(K.squeeze, arguments={'axis': 1})(deep_input_emb)


    def load_model(self, model_path):
        return load_model(model_path, custom_objects={'DNN': DNN,
                                                      'PredictionLayer': PredictionLayer,
                                                      'Transformer': Transformer,
                                                      'AggregationLayer': AggregationLayer})
