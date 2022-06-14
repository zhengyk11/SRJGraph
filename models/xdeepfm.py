# -*- coding:utf-8 -*-

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Dense, Concatenate, Lambda, Multiply, Add, BatchNormalization
from tensorflow.python.keras.models import Model, load_model

from deepctr.layers.core import DNN, PredictionLayer
from deepctr.layers.interaction import CIN

from inputs.dataset import Dataset
from base_model import BaseModel

class xDeepFM(BaseModel):
    def __init__(self, args):
        super(xDeepFM, self).__init__(args)
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
        feature_embed_dict = self.feature_embed_dict
        group_embed_dict = self.group_embed_dict

        '''concatenate all features for the deep layer'''
        deep_input_emb = Concatenate(axis=-1)([group_embed_dict['user'],
                                               group_embed_dict['item'],
                                               group_embed_dict['query']])
        self.deep_input_emb = Lambda(tf.squeeze, arguments={'axis': 1})(deep_input_emb)
        
        '''concatenate all features for the FM layer'''
        self.fm_input = Concatenate(axis=-2)([feature_embed_dict['user_id'], feature_embed_dict['user_gender'],
                                         feature_embed_dict['user_age'], group_embed_dict['query'],
                                         feature_embed_dict['item_id'], feature_embed_dict['item_brand'],
                                         feature_embed_dict['item_seller'], feature_embed_dict['item_cate'],
                                         feature_embed_dict['item_price']])

    def prediction_layer(self):
        dnn_activation = 'dice'
        l2_reg_dnn = 0.
        dnn_use_bn = False

        '''search'''
        '''deep layer'''
        deep_input_emb_search = BatchNormalization()(self.deep_input_emb, training=None)
        output_search = DNN(self.dnn_hidden_units, dnn_activation, l2_reg_dnn,
                            self.dropout_rate, dnn_use_bn)(deep_input_emb_search)
        final_logit_search = Dense(1, use_bias=False)(output_search)
        '''FM layer'''
        exFM_out_search = CIN((128, 128), 'prelu', True, l2_reg_dnn)(self.fm_input)
        exFM_logit_search = Dense(1, use_bias=False)(exFM_out_search)
        final_logit_search = Add()([final_logit_search, exFM_logit_search])

        output_search = PredictionLayer()(final_logit_search)

        '''recommend'''
        '''deep layer'''
        deep_input_emb_recommend = BatchNormalization()(self.deep_input_emb, training=None)
        output_recommend = DNN(self.dnn_hidden_units, dnn_activation, l2_reg_dnn,
                               self.dropout_rate, dnn_use_bn)(deep_input_emb_recommend)
        final_logit_recommend = Dense(1, use_bias=False)(output_recommend)
        '''FM layer'''
        exFM_out_recommend = CIN((128, 128), 'prelu', True, l2_reg_dnn)(self.fm_input)
        exFM_logit_recommend = Dense(1, use_bias=False)(exFM_out_recommend)
        final_logit_recommend = Add()([final_logit_recommend, exFM_logit_recommend])

        output_recommend = PredictionLayer()(final_logit_recommend)

        query_sum = Lambda(tf.reduce_sum, arguments={'axis': 1, 'keepdims': True})(self.features['query'])
        query_mask_search = Lambda(tf.not_equal, arguments={'y': 0})(query_sum)
        query_mask_recommend = Lambda(tf.equal, arguments={'y': 0})(query_sum)
        query_mask_search = Lambda(tf.cast, arguments={'dtype': 'float32'})(query_mask_search)
        query_mask_recommend = Lambda(tf.cast, arguments={'dtype': 'float32'})(query_mask_recommend)

        output_search = Multiply()([output_search, query_mask_search])
        output_recommend = Multiply()([output_recommend, query_mask_recommend])
        output_all = Add()([output_search, output_recommend])

        self.model = Model(inputs=list(self.features.values()), outputs=output_all)
        return self.model
    
    def load_model(self, model_path):
        return load_model(model_path, custom_objects={'DNN': DNN,
                                                      'PredictionLayer': PredictionLayer,
                                                      'CIN': CIN})
