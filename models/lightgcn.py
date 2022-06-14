# -*- coding:utf-8 -*-
import numpy as np

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer, Dense, Concatenate, Lambda, Multiply, Add, Dot, LeakyReLU, Flatten, \
    Softmax, Reshape
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.layers.base import Layer

from deepctr.layers.core import DNN, PredictionLayer
from dnn_no_hist import BaseModel
from inputs.dataset import Dataset
from layers.layers import AggregationLayer, Transformer, NGCFLayer


class GCN(BaseModel):
    def __init__(self, args):
        super(GCN, self).__init__(args)
        self.need_neighbors = True
        self.dataset = Dataset(args, train_files=args.train_files, dev_files=args.dev_files, test_files=args.test_files,
                               need_neighbor=self.need_neighbors)
        self.column_groups = {'user': ['user_id', 'user_gender', 'user_age'],
                              'query': ['query'],
                              'item': ['item_id', 'item_brand', 'item_seller', 'item_cate', 'item_price']}
        if len(self.user_neighbor_nums) >= 1 and np.prod(self.user_neighbor_nums[:1]) > 0:
            self.column_groups.update({'ui': ['ui', 'ui_brand', 'ui_seller', 'ui_cate', 'ui_price']})
        if len(self.user_neighbor_nums) >= 2 and np.prod(self.user_neighbor_nums[:2]) > 0:
            self.column_groups.update({'uiu': ['uiu', 'uiu_gender', 'uiu_age']})
        if len(self.user_neighbor_nums) >= 3 and np.prod(self.user_neighbor_nums[:3]) > 0:
            self.column_groups.update({'uiui': ['uiui', 'uiui_brand', 'uiui_seller', 'uiui_cate', 'uiui_price']})
        if len(self.user_neighbor_nums) >= 4 and np.prod(self.user_neighbor_nums[:4]) > 0:
            self.column_groups.update({'uiuiu': ['uiuiu', 'uiuiu_gender', 'uiuiu_age']})
        if len(self.item_neighbor_nums) >= 1 and np.prod(self.item_neighbor_nums[:1]) > 0:
            self.column_groups.update({'iu': ['iu', 'iu_gender', 'iu_age']})
        if len(self.item_neighbor_nums) >= 2 and np.prod(self.item_neighbor_nums[:2]) > 0:
            self.column_groups.update({'iui': ['iui', 'iui_brand', 'iui_seller', 'iui_cate', 'iui_price']})
        if len(self.item_neighbor_nums) >= 3 and np.prod(self.item_neighbor_nums[:3]) > 0:
            self.column_groups.update({'iuiu': ['iuiu', 'iuiu_gender', 'iuiu_age']})
        if len(self.item_neighbor_nums) >= 4 and np.prod(self.item_neighbor_nums[:4]) > 0:
            self.column_groups.update({'iuiui': ['iuiui', 'iuiui_brand', 'iuiui_seller', 'iuiui_cate', 'iuiui_price']})

        self.initialize_model()

    def create_model(self):
        group_embed_dict = self.group_embed_dict

        agg_layer_u2i = lightLayer(self.dropout_rate)
        agg_layer_i2u = lightLayer(self.dropout_rate)
        agg_user_neighbor_nums = [1] + self.user_neighbor_nums
        agg_item_neighbor_nums = [1] + self.item_neighbor_nums
        user_dense_layer = Dense(48)
        item_dense_layer = Dense(48)

        agg_layer_outs = {}
        '''u--i--u--i<--u'''
        if 'uiuiu' in group_embed_dict:
            inputs = [self.features['uiui'], item_dense_layer(group_embed_dict['uiui']), self.features['uiuiu']]
            if 'uiuiu' in agg_layer_outs:
                inputs += [agg_layer_outs['uiuiu']]
            else:
                inputs += [user_dense_layer(group_embed_dict['uiuiu'])]
            self.logger.info('aggregating u--i--u--i<--u')
            agg_layer_outs['uiui'] = agg_layer_u2i(inputs, order=4, neighbor_nums=agg_user_neighbor_nums)

        '''u--i--u<--i--u'''
        if 'uiui' in group_embed_dict:
            inputs = [self.features['uiu'], user_dense_layer(group_embed_dict['uiu']), self.features['uiui']]
            if 'uiui' in agg_layer_outs:
                inputs += [agg_layer_outs['uiui']]
            else:
                inputs += [item_dense_layer(group_embed_dict['uiui'])]
            self.logger.info('aggregating u--i--u<--i--u')
            agg_layer_outs['uiu'] = agg_layer_i2u(inputs, order=3, neighbor_nums=agg_user_neighbor_nums)

        '''u--i<--u--i--u'''
        if 'uiu' in group_embed_dict:
            inputs = [self.features['ui'], item_dense_layer(group_embed_dict['ui']), self.features['uiu']]
            if 'uiu' in agg_layer_outs:
                inputs += [agg_layer_outs['uiu']]
            else:
                inputs += [user_dense_layer(group_embed_dict['uiu'])]
            self.logger.info('aggregating u--i<--u--i--u')
            agg_layer_outs['ui'] = agg_layer_u2i(inputs, order=2, neighbor_nums=agg_user_neighbor_nums)

        '''u<--i--u--i--u'''
        if 'ui' in group_embed_dict:
            inputs = [self.features['user_id'], user_dense_layer(group_embed_dict['user']), self.features['ui']]
            if 'ui' in agg_layer_outs:
                inputs += [agg_layer_outs['ui']]
            else:
                inputs += [item_dense_layer(group_embed_dict['ui'])]
            self.logger.info('aggregating u<--i--u--i--u')
            agg_layer_outs['u'] = agg_layer_i2u(inputs, order=1, neighbor_nums=agg_user_neighbor_nums)
            agg_layer_outs['u'] = Lambda(K.squeeze, arguments={'axis': 2})(agg_layer_outs['u'])

        '''i--u--i--u<--i'''
        if 'iuiui' in group_embed_dict:
            inputs = [self.features['iuiu'], user_dense_layer(group_embed_dict['iuiu']), self.features['iuiui']]
            if 'iuiui' in agg_layer_outs:
                inputs += [agg_layer_outs['iuiui']]
            else:
                inputs += [item_dense_layer(group_embed_dict['iuiui'])]
            self.logger.info('aggregating i--u--i--u<--i')
            agg_layer_outs['iuiu'] = agg_layer_i2u(inputs, order=4, neighbor_nums=agg_item_neighbor_nums)

        '''i--u--i<--u--i'''
        if 'iuiu' in group_embed_dict:
            inputs = [self.features['iui'], item_dense_layer(group_embed_dict['iui']), self.features['iuiu']]
            if 'iuiu' in agg_layer_outs:
                inputs += [agg_layer_outs['iuiu']]
            else:
                inputs += [user_dense_layer(group_embed_dict['iuiu'])]
            self.logger.info('aggregating i--u--i<--u--i')
            agg_layer_outs['iui'] = agg_layer_u2i(inputs, order=3, neighbor_nums=agg_item_neighbor_nums)

        '''i--u<--i--u--i'''
        if 'iui' in group_embed_dict:
            inputs = [self.features['iu'], user_dense_layer(group_embed_dict['iu']), self.features['iui']]
            if 'iui' in agg_layer_outs:
                inputs += [agg_layer_outs['iui']]
            else:
                inputs += [item_dense_layer(group_embed_dict['iui'])]
            self.logger.info('aggregating i--u<--i--u--i')
            agg_layer_outs['iu'] = agg_layer_i2u(inputs, order=2, neighbor_nums=agg_item_neighbor_nums)

        '''i<--u--i--u--i'''
        if 'iu' in group_embed_dict:
            inputs = [self.features['item_id'], item_dense_layer(group_embed_dict['item']), self.features['iu']]
            if 'iu' in agg_layer_outs:
                inputs += [agg_layer_outs['iu']]
            else:
                inputs += [user_dense_layer(group_embed_dict['iu'])]
            self.logger.info('aggregating i<--u--i--u--i')
            agg_layer_outs['i'] = agg_layer_u2i(inputs, order=1, neighbor_nums=agg_item_neighbor_nums)
            agg_layer_outs['i'] = Lambda(K.squeeze, arguments={'axis': 2})(agg_layer_outs['i'])

        agg_layer_outs['u'] = Add()([user_dense_layer(group_embed_dict['user']), agg_layer_outs['u']])
        agg_layer_outs['i'] = Add()([item_dense_layer(group_embed_dict['item']), agg_layer_outs['i']])
        deep_input_emb = Multiply()([agg_layer_outs['u'], agg_layer_outs['i']])
        deep_input_emb = [group_embed_dict['user'], group_embed_dict['item'], group_embed_dict['query']] + [
            deep_input_emb]
        deep_input_emb = Concatenate(axis=-1)(deep_input_emb)
        self.deep_input_emb = Lambda(K.squeeze, arguments={'axis': 1})(deep_input_emb)

    def load_model(self, model_path):
        return load_model(model_path, custom_objects={'DNN': DNN,
                                                      'PredictionLayer': PredictionLayer,
                                                      'lightLayer': lightLayer})


class lightLayer(Layer):
    def __init__(self, dropout_rate=0., **kwargs):
        super(lightLayer, self).__init__(**kwargs)
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        super(lightLayer, self).build(input_shape)
        self.embed_size = int(input_shape[1][-1])

    def call(self, inputs, order=None, neighbor_nums=None, **kwargs):
        node_id, node_embed, neighbor_id, neighbor_embed = inputs

        node_embed_expand = tf.reshape(node_embed, [-1, np.prod(neighbor_nums[:order]), 1, self.embed_size])
        neighbor_embed = tf.reshape(neighbor_embed,
                                    [-1, np.prod(neighbor_nums[:order]), neighbor_nums[order], self.embed_size])
        neighbor_embed = tf.concat([node_embed_expand, neighbor_embed], axis=2)

        # node_embed_expand = tf.tile(node_embed_expand, [1, 1, neighbor_nums[order] + 1, 1])

        neighbor_embed1 = neighbor_embed

        Query_mask = tf.expand_dims(tf.not_equal(node_id, 0), axis=2)
        Query_mask = tf.cast(Query_mask, dtype=tf.float32)
        neighbor_mask = tf.not_equal(neighbor_id, 0)
        neighbor_mask = tf.cast(neighbor_mask, dtype=tf.float32)
        neighbor_mask = tf.reshape(neighbor_mask, (-1, np.prod(neighbor_nums[:order]), neighbor_nums[order]))
        neighbor_mask = tf.concat([Query_mask, neighbor_mask], axis=-1)
        neighbor_mask = tf.expand_dims(neighbor_mask, axis=2)

        neighbor_embed1 = Dot(axes=(3, 2))([neighbor_mask, neighbor_embed1])

        neighbor_mask_sum = Lambda(tf.reduce_sum, arguments={'axis': -1, 'keepdims': True})(neighbor_mask)
        neighbor_mask_sum = Lambda(lambda x: x + 1e-6)(neighbor_mask_sum)
        neighbor_embed1 = Lambda(lambda inputs: inputs[0] / inputs[1])([neighbor_embed1, neighbor_mask_sum])
        neighbor_embed = neighbor_embed1
        Query_mask = tf.expand_dims(Query_mask, axis=2)
        agg_node_embed = Multiply()([neighbor_embed, Query_mask])
        agg_node_embed = LeakyReLU()(agg_node_embed)

        return agg_node_embed

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[2][-1], input_shape[0][-1])

    def get_config(self, ):
        config = {'dropout_rate': self.dropout_rate}
        base_config = super(lightLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))