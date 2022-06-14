# coding=utf-8
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer, Dense, Concatenate, Flatten, Softmax, Lambda, Reshape, Embedding, \
    LeakyReLU, Multiply, Add, Dot

from deepctr.layers import LayerNormalization
from deepctr.layers.core import DNN
from deepctr.layers.core import LocalActivationUnit


class AttentionSequencePoolingLayer(Layer):
    """The Attentional sequence pooling operation used in DIN.

      Input shape
        - A list of three tensor: [query, keys, keys_length]
        - query is a 3D tensor with shape:  ``(batch_size, 1, embedding_size)``
        - keys is a 3D tensor with shape:   ``(batch_size, T, embedding_size)``
        - keys_mask is a 2D tensor with shape: ``(batch_size, 1)``

      Output shape
        - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.

      Arguments
        - **att_hidden_units**:list of positive integer, the attention net layer number and units in each layer.
        - **att_activation**: Activation function to use in attention net.

      References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(self, att_hidden_units=(80, 40), att_activation='sigmoid', dropout_rate=0., **kwargs):
        super(AttentionSequencePoolingLayer, self).__init__(**kwargs)
        self.att_hidden_units = att_hidden_units
        self.att_activation = att_activation
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        super(AttentionSequencePoolingLayer, self).build(input_shape)  # Be sure to call this somewhere!
        self.local_att = LocalActivationUnit(self.att_hidden_units, self.att_activation, l2_reg=0.,
                                             dropout_rate=self.dropout_rate, use_bn=False, seed=1024)

    def call(self, inputs, mask=None, training=None, **kwargs):
        if mask is None:
            raise ValueError(
                "When supports_masking=True,input must support masking")
        queries, keys = inputs
        key_masks = tf.expand_dims(mask[-1], axis=1)
        #
        # else:
        #
        #     queries, keys, keys_length = inputs
        #     hist_len = keys.get_shape()[1]
        #     key_masks = tf.sequence_mask(keys_length, hist_len)

        attention_score = self.local_att([queries, keys], training=training)
        attention_score = tf.transpose(attention_score, (0, 2, 1))
        paddings = tf.ones_like(attention_score) * (-2 ** 32 + 1)
        attention_score = tf.where(key_masks, attention_score, paddings)
        attention_score = tf.nn.softmax(attention_score)
        outputs = tf.matmul(attention_score, keys)
        # if tf.__version__ < '1.13.0':
        #     outputs._uses_learning_phase = attention_score._uses_learning_phase
        # else:
        #     outputs._uses_learning_phase = training is not None
        return outputs

    def compute_output_shape(self, input_shape):
        return (None, 1, input_shape[0][-1])

    def get_config(self, ):
        config = {'att_hidden_units': self.att_hidden_units, 'att_activation': self.att_activation}
        base_config = super(AttentionSequencePoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Transformer(Layer):
    """
      Input shape
        - a list of two 4D tensor with shape ``(batch_size, any, timesteps, input_dim)``.
      Output shape
        - 4D tensor with shape: ``(batch_size, any, timesteps, input_dim)``.
      Arguments
            - **att_embedding_size**: int.The embedding size in multi-head self-attention network.
            - **head_num**: int.The head number in multi-head  self-attention network.
            - **dropout_rate**: float between 0 and 1. Fraction of the units to drop.
            - **use_positional_encoding**: bool. Whether or not use positional_encoding
            - **use_res**: bool. Whether or not use standard residual connections before output.
            - **use_feed_forward**: bool. Whether or not use pointwise feed foward network.
            - **use_layer_norm**: bool. Whether or not use Layer Normalization.
            - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, att_embedding_size, head_num=8, dropout_rate=0.0, use_positional_encoding=True, use_res=True,
                 use_feed_forward=True, use_layer_norm=True, seed=1024, **kwargs):
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.num_units = att_embedding_size * head_num
        self.use_res = use_res
        self.use_feed_forward = use_feed_forward
        self.seed = seed
        self.use_positional_encoding = use_positional_encoding
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.seq_len_max = 4096
        super(Transformer, self).__init__(**kwargs)

    def build(self, input_shape):
        embedding_size = int(input_shape[0][-1])
        if self.num_units != embedding_size:
            raise ValueError(
                "att_embedding_size * head_num must equal the last dimension size of inputs,got %d * %d != %d" % (
                    self.att_embedding_size, self.head_num, embedding_size))
        assert max(int(input_shape[0][2]), int(input_shape[1][2])) <= self.seq_len_max
        self.W_Query = self.add_weight(name='query_weight',
                                       shape=(embedding_size, self.att_embedding_size * self.head_num),
                                       dtype=tf.float32,
                                       initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed))
        self.B_Query = self.add_weight(name='query_bias', shape=(self.att_embedding_size * self.head_num,),
                                       dtype=tf.float32,
                                       initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed))
        self.W_key = self.add_weight(name='key_weight', shape=(embedding_size, self.att_embedding_size * self.head_num),
                                     dtype=tf.float32,
                                     initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed + 1))
        self.B_Key = self.add_weight(name='key_bias', shape=(self.att_embedding_size * self.head_num,),
                                     dtype=tf.float32,
                                     initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed))
        '''
        self.W_Value = self.add_weight(name='value_weight',
                                       shape=(embedding_size, self.att_embedding_size * self.head_num),
                                       dtype=tf.float32,
                                       initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed + 2))
        self.B_Value = self.add_weight(name='value_bias', shape=(self.att_embedding_size * self.head_num,),
                                       dtype=tf.float32,
                                       initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed))
        '''
        self.position_emb = Embedding(self.seq_len_max, embedding_size,
                                      embeddings_initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed))

        if self.use_feed_forward:
            self.fw = self.add_weight('fw', shape=(self.num_units, self.num_units), dtype=tf.float32,
                                      initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
            self.fb = self.add_weight('fb', shape=(self.num_units,), dtype=tf.float32,
                                      initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed)
        self.ln = LayerNormalization()
        self.ln_q = LayerNormalization()
        self.ln_k = LayerNormalization()
        # Be sure to call this somewhere!
        super(Transformer, self).build(input_shape)

    def call(self, inputs, mask=None, training=None, **kwargs):

        # if self.supports_masking:

        queries, keys = inputs
        query_masks, key_masks = mask
        query_masks = tf.cast(query_masks, tf.float32)
        key_masks = tf.cast(key_masks, tf.float32)

        # else:
        #     queries, keys, query_masks, key_masks = inputs
        #     query_masks = tf.sequence_mask(
        #         query_masks, self.seq_len_query, dtype=tf.float32)
        #     key_masks = tf.sequence_mask(
        #         key_masks, self.seq_len_key, dtype=tf.float32)
        #     query_masks = tf.squeeze(query_masks, axis=2)
        #     key_masks = tf.squeeze(key_masks, axis=2)

        values = keys
        if self.use_positional_encoding:
            keys = keys + self.position_emb(tf.expand_dims(tf.range(tf.shape(keys)[2]), 0))

        # print 'queries', queries.get_shape().as_list()
        # print 'keys', keys.get_shape().as_list()
        # print 'values', values.get_shape().as_list()

        querys = tf.nn.bias_add(tf.tensordot(queries, self.W_Query, axes=(-1, 0)), self.B_Query)  # None T_q D*head_num
        keys = tf.nn.bias_add(tf.tensordot(keys, self.W_key, axes=(-1, 0)), self.B_Key)
        # values = tf.nn.bias_add(tf.tensordot(values, self.W_Value, axes=(-1, 0)), self.B_Value)
        
        # head_num*None T_q D
        querys = tf.concat(tf.split(querys, self.head_num, axis=-1), axis=1)
        keys = tf.concat(tf.split(keys, self.head_num, axis=-1), axis=1)
        values = tf.concat(tf.split(values, self.head_num, axis=-1), axis=1)

        # print 'querys', querys.get_shape().as_list()
        # print 'keys', keys.get_shape().as_list()
        # print 'values', values.get_shape().as_list()

        if self.use_layer_norm:
            querys = self.ln_q(querys)
            keys = self.ln_k(keys)

        # head_num*None T_q T_k
        outputs = tf.matmul(querys, keys, transpose_b=True)
        outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
        # print 'outputs', outputs.get_shape().as_list()

        # print 'key_masks', key_masks.get_shape().as_list()
        key_masks = tf.tile(key_masks, [1, self.head_num, 1])
        # print 'key_masks', key_masks.get_shape().as_list()
        key_masks = tf.tile(tf.expand_dims(key_masks, 2), [1, 1, tf.shape(querys)[2], 1])  # (h*N, T_q, T_k)
        # print 'key_masks', key_masks.get_shape().as_list()
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 1), outputs, paddings)  # (h*N, T_q, T_k)

        outputs = tf.nn.softmax(outputs, axis=-1)

        # print 'query_masks', query_masks.get_shape().as_list()
        query_masks = tf.tile(query_masks, [1, self.head_num, 1])  # (h*N, T_q)
        # print 'query_masks', query_masks.get_shape().as_list()
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, 1, tf.shape(keys)[2]])  # (h*N, T_q, T_k)
        # print 'query_masks', query_masks.get_shape().as_list()
        outputs *= query_masks

        outputs = self.dropout(outputs, training=training)

        result = tf.matmul(outputs, values)  # Weighted sum ( h*N, T_q, C/h)
        result = tf.concat(tf.split(result, self.head_num, axis=1), axis=-1)
        # print 'result', result.get_shape().as_list()

        if self.use_feed_forward:
            fw1 = tf.nn.bias_add(tf.tensordot(result, self.fw, axes=[-1, 0]), self.fb)
            fw1 = self.dropout(fw1, training=training)
            if self.use_res:
                result += fw1
            else:
                result = fw1
        if self.use_layer_norm:
            result = self.ln(result)

        # result = tf.squeeze(result, axis=2)
        result = tf.reduce_mean(result, axis=2)
        # return reduce_mean(result, axis=1, keep_dims=True)
        # print 'result', result.get_shape().as_list()
        return result

    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][1], input_shape[0][2], self.num_units)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self, ):
        config = {'att_embedding_size': self.att_embedding_size, 'head_num': self.head_num,
                  'dropout_rate': self.dropout_rate, 'use_res': self.use_res,
                  'use_positional_encoding': self.use_positional_encoding, 'use_feed_forward': self.use_feed_forward,
                  'use_layer_norm': self.use_layer_norm, 'seed': self.seed}
        base_config = super(Transformer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AggregationLayer(Layer):
    def __init__(self, dropout_rate=0., **kwargs):
        super(AggregationLayer, self).__init__(**kwargs)
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        super(AggregationLayer, self).build(input_shape)
        # print (input_shape[0][0], input_shape[2][-1], input_shape[0][-1])

        self.embed_size = int(input_shape[0][-1])
        self.query_embed_size = int(input_shape[4][-1])
        self.dnn_Q = Dense(self.embed_size)
        self.transformer = Transformer(att_embedding_size=self.embed_size, head_num=1,
                                       dropout_rate=self.dropout_rate)

    def call(self, inputs, is_leaf=False, agg_type=None, order=None, neighbor_nums=None, **kwargs):
        if is_leaf:
            target_uqi, pre_embed, node_id, node_embed, node_query, neighbor_id, neighbor_embed, neighbor_query = inputs
        else:
            target_uqi, pre_embed, node_id, node_embed, node_query, neighbor_id, agg_neighbor_embed = inputs

        target_uqi_expand = tf.tile(target_uqi, [1, np.prod(neighbor_nums[:order]), 1])
        target_uqi_expand = tf.expand_dims(target_uqi_expand, axis=2)
        # print 'target_uqi_expand', target_uqi_expand.get_shape().as_list()

        pre_embed_expand = tf.tile(pre_embed, [1, neighbor_nums[order-1], 1])
        node_query_reshape = tf.reshape(node_query, [-1, np.prod(neighbor_nums[:order]), self.query_embed_size])

        if agg_type == 'u2i':
            node_uqi = tf.concat([pre_embed_expand, node_query_reshape, node_embed], axis=2)
        else:
            node_uqi = tf.concat([node_embed, node_query_reshape, pre_embed_expand], axis=2)

        node_uqi = tf.expand_dims(node_uqi, axis=2)
        # print 'node_uqi', node_uqi.get_shape().as_list()

        Query = self.dnn_Q(tf.concat([target_uqi_expand, node_uqi], axis=-1))

        node_embed_expand = tf.tile(node_embed, [1, neighbor_nums[order], 1])
        if is_leaf:
            neighbor_query_reshape = tf.reshape(neighbor_query, [-1, np.prod(neighbor_nums[:order + 1]), self.query_embed_size])
            if agg_type == 'u2i':
                neighbor_uqi = tf.concat([neighbor_embed, neighbor_query_reshape, node_embed_expand], axis=2)
            else:
                neighbor_uqi = tf.concat([node_embed_expand, neighbor_query_reshape, neighbor_embed], axis=2)
        else:
            neighbor_uqi = agg_neighbor_embed
        neighbor_uqi = tf.reshape(neighbor_uqi, [-1, np.prod(neighbor_nums[:order]), neighbor_nums[order], self.embed_size])

        # print 'neighbor_uqi', neighbor_uqi.get_shape().as_list()
        Key = tf.concat([node_uqi, neighbor_uqi], axis=2)

        Query_mask = tf.expand_dims(tf.not_equal(node_id, 0), axis=2)

        neighbor_uqi_concat_mask = tf.not_equal(neighbor_id, 0)
        neighbor_uqi_concat_mask = tf.reshape(neighbor_uqi_concat_mask, (-1, np.prod(neighbor_nums[:order]), neighbor_nums[order]))

        Key_mask = tf.concat([Query_mask, neighbor_uqi_concat_mask], axis=-1)

        # print 'Query_mask', Query_mask.get_shape().as_list()
        # print 'Key_mask', Key_mask.get_shape().as_list()
        agg_node_embed = self.transformer(inputs=[Query, Key],
                                        mask=[Query_mask, Key_mask])
        # print 'agg_node_embed', agg_node_embed.get_shape().as_list()
        # print
        return agg_node_embed

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[2][-1], input_shape[0][-1])

    def get_config(self, ):
        config = {'dropout_rate': self.dropout_rate}
        base_config = super(AggregationLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class NGCFLayer(Layer):
    def __init__(self, dropout_rate=0., **kwargs):
        super(NGCFLayer, self).__init__(**kwargs)
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        super(NGCFLayer, self).build(input_shape)
        self.embed_size = int(input_shape[1][-1])
        self.dnn_W1 = Dense(self.embed_size)
        self.dnn_W2 = Dense(self.embed_size)

    def call(self, inputs, order=None, neighbor_nums=None, **kwargs):
        node_id, node_embed, neighbor_id, neighbor_embed = inputs

        node_embed_expand = tf.reshape(node_embed, [-1, np.prod(neighbor_nums[:order]), 1, self.embed_size])
        neighbor_embed = tf.reshape(neighbor_embed,
                                    [-1, np.prod(neighbor_nums[:order]), neighbor_nums[order], self.embed_size])
        neighbor_embed = tf.concat([node_embed_expand, neighbor_embed], axis=2)

        node_embed_expand = tf.tile(node_embed_expand, [1, 1, neighbor_nums[order] + 1, 1])

        neighbor_embed1 = neighbor_embed
        neighbor_embed2 = Multiply()([neighbor_embed, node_embed_expand])
        neighbor_embed1 = self.dnn_W1(neighbor_embed1)
        neighbor_embed2 = self.dnn_W2(neighbor_embed2)

        Query_mask = tf.expand_dims(tf.not_equal(node_id, 0), axis=2)
        Query_mask = tf.cast(Query_mask, dtype=tf.float32)
        neighbor_mask = tf.not_equal(neighbor_id, 0)
        neighbor_mask = tf.cast(neighbor_mask, dtype=tf.float32)
        neighbor_mask = tf.reshape(neighbor_mask, (-1, np.prod(neighbor_nums[:order]), neighbor_nums[order]))
        neighbor_mask = tf.concat([Query_mask, neighbor_mask], axis=-1)
        neighbor_mask = tf.expand_dims(neighbor_mask, axis=2)

        neighbor_embed1 = Dot(axes=(3, 2))([neighbor_mask, neighbor_embed1])
        neighbor_embed2 = Dot(axes=(3, 2))([neighbor_mask, neighbor_embed2])
        print(neighbor_embed2.get_shape())

        neighbor_mask_sum = Lambda(tf.reduce_sum, arguments={'axis': -1, 'keepdims': True})(neighbor_mask)
        neighbor_mask_sum = Lambda(lambda x: x + 1e-6)(neighbor_mask_sum)
        neighbor_embed1 = Lambda(lambda inputs: inputs[0] / inputs[1])([neighbor_embed1, neighbor_mask_sum])
        neighbor_embed2 = Lambda(lambda inputs: inputs[0] / inputs[1])([neighbor_embed2, neighbor_mask_sum])
        neighbor_embed = Add()([neighbor_embed1, neighbor_embed2])
        Query_mask = tf.expand_dims(Query_mask, axis=2)
        agg_node_embed = Multiply()([neighbor_embed, Query_mask])
        agg_node_embed = LeakyReLU()(agg_node_embed)

        return agg_node_embed

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[2][-1], input_shape[0][-1])

    def get_config(self, ):
        config = {'dropout_rate': self.dropout_rate}
        base_config = super(NGCFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AggregationLayerV1(Layer):
    def __init__(self, dropout_rate=0., **kwargs):
        super(AggregationLayerV1, self).__init__(**kwargs)
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        super(AggregationLayerV1, self).build(input_shape)
        # print (input_shape[0][0], input_shape[2][-1], input_shape[0][-1])

        self.embed_size = int(input_shape[0][-1])
        self.query_embed_size = int(input_shape[4][-1])
        self.dnn_Q = Dense(self.embed_size)
        self.transformer = Transformer(att_embedding_size=self.embed_size, head_num=1,
                                       dropout_rate=self.dropout_rate)

    def call(self, inputs, is_leaf=False, agg_type=None, order=None, neighbor_nums=None, **kwargs):
        if is_leaf:
            target_uqi, pre_embed, node_id, node_embed, node_query, neighbor_id, neighbor_embed, neighbor_query = inputs
        else:
            target_uqi, pre_embed, node_id, node_embed, node_query, neighbor_id, agg_neighbor_embed = inputs
        target_uqi_expand = tf.tile(target_uqi, [1, np.prod(neighbor_nums[:order]), 1])
        target_uqi_expand = tf.expand_dims(target_uqi_expand, axis=2)
        # print 'target_uqi_expand', target_uqi_expand.get_shape().as_list()

        # pre_embed_expand = tf.tile(pre_embed, [1, neighbor_nums[order - 1], 1])
        # node_query_reshape = tf.reshape(node_query, [-1, np.prod(neighbor_nums[:order]), self.query_embed_size])

        # if agg_type == 'u2i':
        #     node_uqi = node_embed # tf.concat([tf.zeros_like(pre_embed_expand), tf.zeros_like(node_query_reshape), node_embed], axis=2)
        # else:
        #     node_uqi = node_embed # tf.concat([node_embed, tf.zeros_like(node_query_reshape), tf.zeros_like(pre_embed_expand)], axis=2)

        node_uqi = tf.expand_dims(node_embed, axis=2)
        # print 'node_uqi', node_uqi.get_shape().as_list()

        Query = self.dnn_Q(tf.concat([target_uqi_expand, node_uqi], axis=-1))

        # node_embed_expand = tf.tile(node_embed, [1, neighbor_nums[order], 1])
        if is_leaf:
            # neighbor_query_reshape = tf.reshape(neighbor_query,
            #                                     [-1, np.prod(neighbor_nums[:order + 1]), self.query_embed_size])
            neighbor_uqi = neighbor_embed
            # if agg_type == 'u2i':
            #     neighbor_uqi = tf.concat([neighbor_embed, tf.zeros_like(neighbor_query_reshape), tf.zeros_like(node_embed_expand)], axis=2)
            # else:
            #     neighbor_uqi = tf.concat([tf.zeros_like(node_embed_expand), tf.zeros_like(neighbor_query_reshape), neighbor_embed], axis=2)
        else:
            neighbor_uqi = agg_neighbor_embed
        neighbor_uqi = tf.reshape(neighbor_uqi,
                                  [-1, np.prod(neighbor_nums[:order]), neighbor_nums[order], self.embed_size])

        # print 'neighbor_uqi', neighbor_uqi.get_shape().as_list()
        Key = tf.concat([node_uqi, neighbor_uqi], axis=2)

        Query_mask = tf.expand_dims(tf.not_equal(node_id, 0), axis=2)

        neighbor_uqi_concat_mask = tf.not_equal(neighbor_id, 0)
        neighbor_uqi_concat_mask = tf.reshape(neighbor_uqi_concat_mask,
                                              (-1, np.prod(neighbor_nums[:order]), neighbor_nums[order]))

        Key_mask = tf.concat([Query_mask, neighbor_uqi_concat_mask], axis=-1)

        # print 'Query_mask', Query_mask.get_shape().as_list()
        # print 'Key_mask', Key_mask.get_shape().as_list()
        agg_node_embed = self.transformer(inputs=[Query, Key],
                                          mask=[Query_mask, Key_mask])
        # print 'agg_node_embed', agg_node_embed.get_shape().as_list()
        # print
        return agg_node_embed

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[2][-1], input_shape[0][-1])

    def get_config(self, ):
        config = {'dropout_rate': self.dropout_rate}
        base_config = super(AggregationLayerV1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AggregationLayerV2(Layer):
    def __init__(self, dropout_rate=0., **kwargs):
        super(AggregationLayerV2, self).__init__(**kwargs)
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        super(AggregationLayerV2, self).build(input_shape)
        # print (input_shape[0][0], input_shape[2][-1], input_shape[0][-1])

        self.embed_size = int(input_shape[0][-1])
        self.node_embed_size = int(input_shape[3][-1])
        self.query_embed_size = int(input_shape[4][-1])
        self.dnn_Q = Dense(self.embed_size)
        self.dnn_out = Dense(self.embed_size)

        self.dnn_agg_linear = Dense(self.embed_size)
        self.dnn_agg_linear_2 = Dense(self.embed_size)
        self.dnn_agg_linear_3 = Dense(self.node_embed_size)
        self.dnn_agg_nonlinear = Dense(self.embed_size)
        self.transformer = Transformer(att_embedding_size=self.embed_size, head_num=1,
                                       dropout_rate=self.dropout_rate)

    def call(self, inputs, is_leaf=False, agg_type=None, order=None, neighbor_nums=None, **kwargs):
        if is_leaf:
            target_uqi, pre_embed, node_id, node_embed, node_query, neighbor_id, neighbor_embed, neighbor_query = inputs
        else:
            target_uqi, pre_embed, node_id, node_embed, node_query, neighbor_id, agg_neighbor_embed = inputs

        target_uqi_expand = tf.tile(target_uqi, [1, np.prod(neighbor_nums[:order]), 1])
        target_uqi_expand = tf.expand_dims(target_uqi_expand, axis=2)
        # print 'target_uqi_expand', target_uqi_expand.get_shape().as_list()

        pre_embed_expand = tf.tile(pre_embed, [1, neighbor_nums[order-1], 1])
        node_query_reshape = tf.reshape(node_query, [-1, np.prod(neighbor_nums[:order]), self.query_embed_size])

        if agg_type == 'u2i':
            node_uqi = tf.concat([pre_embed_expand, node_query_reshape, node_embed], axis=2)
        else:
            node_uqi = tf.concat([node_embed, node_query_reshape, pre_embed_expand], axis=2)

        node_uqi = tf.expand_dims(node_uqi, axis=2)
        # print 'node_uqi', node_uqi.get_shape().as_list()

        Query = self.dnn_Q(tf.concat([target_uqi_expand, node_uqi], axis=-1))

        node_embed_expand = tf.tile(node_embed, [1, neighbor_nums[order], 1])
        if is_leaf:
            neighbor_query_reshape = tf.reshape(neighbor_query, [-1, np.prod(neighbor_nums[:order + 1]), self.query_embed_size])
            if agg_type == 'u2i':
                neighbor_uqi = tf.concat([neighbor_embed, neighbor_query_reshape, node_embed_expand], axis=2)
            else:
                neighbor_uqi = tf.concat([node_embed_expand, neighbor_query_reshape, neighbor_embed], axis=2)
        else:
            neighbor_uqi = agg_neighbor_embed
        neighbor_uqi = tf.reshape(neighbor_uqi, [-1, np.prod(neighbor_nums[:order]), neighbor_nums[order], self.embed_size])

        # print 'neighbor_uqi', neighbor_uqi.get_shape().as_list()
        Key = tf.concat([node_uqi, neighbor_uqi], axis=2)

        Query_mask = tf.expand_dims(tf.not_equal(node_id, 0), axis=2)

        neighbor_uqi_concat_mask = tf.not_equal(neighbor_id, 0)
        neighbor_uqi_concat_mask = tf.reshape(neighbor_uqi_concat_mask, (-1, np.prod(neighbor_nums[:order]), neighbor_nums[order]))

        Key_mask = tf.concat([Query_mask, neighbor_uqi_concat_mask], axis=-1)

        # print 'Query_mask', Query_mask.get_shape().as_list()
        # print 'Key_mask', Key_mask.get_shape().as_list()
        agg_node_embed = self.transformer(inputs=[Query, Key],
                                        mask=[Query_mask, Key_mask])
        # print 'agg_node_embed', agg_node_embed.get_shape().as_list()
        # print

        '''linear'''
        if is_leaf:
            neighbor_embed = neighbor_embed
        else:
            neighbor_embed = self.dnn_agg_linear_3(agg_neighbor_embed)
        node_embed_expand = tf.reshape(node_embed, [-1, np.prod(neighbor_nums[:order]), 1, self.node_embed_size])
        neighbor_embed_reshape = tf.reshape(neighbor_embed,
                                            [-1, np.prod(neighbor_nums[:order]), neighbor_nums[order],
                                             self.node_embed_size])
        neighbor_embed_concat = tf.concat([node_embed_expand, neighbor_embed_reshape], axis=2)

        Query_mask_linear = tf.expand_dims(tf.not_equal(node_id, 0), axis=2)
        Query_mask_linear = tf.cast(Query_mask_linear, dtype=tf.float32)
        neighbor_mask_linear = tf.not_equal(neighbor_id, 0)
        neighbor_mask_linear = tf.cast(neighbor_mask_linear, dtype=tf.float32)
        neighbor_mask_linear = tf.reshape(neighbor_mask_linear,
                                          (-1, np.prod(neighbor_nums[:order]), neighbor_nums[order]))
        neighbor_mask_linear = tf.concat([Query_mask_linear, neighbor_mask_linear], axis=-1)
        neighbor_mask_linear = tf.expand_dims(neighbor_mask_linear, axis=2)

        neighbor_embed1 = Dot(axes=(3, 2))([neighbor_mask_linear, neighbor_embed_concat])

        neighbor_mask_sum = Lambda(tf.reduce_sum, arguments={'axis': -1, 'keepdims': True})(neighbor_mask_linear)
        neighbor_mask_sum = Lambda(lambda x: x + 1e-6)(neighbor_mask_sum)
        neighbor_embed1 = Lambda(lambda inputs: inputs[0] / inputs[1])([neighbor_embed1, neighbor_mask_sum])
        Query_mask_linear_expand = tf.expand_dims(Query_mask_linear, axis=2)
        agg_node_embed_linear = Multiply()([neighbor_embed1, Query_mask_linear_expand])
        agg_node_embed_linear = self.dnn_agg_linear(LeakyReLU()(agg_node_embed_linear))
        agg_node_embed_linear = tf.squeeze(agg_node_embed_linear, axis=2)
        print 'agg_node_embed_linear', agg_node_embed_linear.get_shape().as_list()

        '''linear and nonlinear attention'''
        agg_node_embed_linear_ = self.dnn_agg_linear_2(LeakyReLU()(agg_node_embed_linear))
        agg_node_embed_ = self.dnn_agg_nonlinear(LeakyReLU()(agg_node_embed))

        agg_att_score = tf.nn.sigmoid(tf.reduce_mean(agg_node_embed_linear_ * agg_node_embed_, axis=-1, keepdims=True))
        print 'agg_att_score', agg_att_score.get_shape().as_list()
        agg_node_embed_final = agg_att_score * agg_node_embed_linear + (1. - agg_att_score) * agg_node_embed
        print 'agg_node_embed_final', agg_node_embed_final.get_shape().as_list()
        agg_node_embed_final = tf.reshape(agg_node_embed_final,
                                          [-1, np.prod(neighbor_nums[:order]), self.embed_size])


        return agg_node_embed_final

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[2][-1], input_shape[0][-1])

    def get_config(self, ):
        config = {'dropout_rate': self.dropout_rate}
        base_config = super(AggregationLayerV2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
