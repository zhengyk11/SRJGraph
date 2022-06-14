# -*- coding:utf-8 -*-
# @Organization: Alibaba
# @Author: Yukun Zheng
# @Email: zyk265182@alibaba-inc.com
# @Time: 2020/9/21 14:04

import numpy as np
from deepctr.feature_column import SparseFeat, VarLenSparseFeat


def get_all_feature_columns(emb_init,
                            user_embed_size, item_embed_size, query_embed_size,
                            user_feat_embed_size, item_feat_embed_size,
                            user_vocab_size, item_vocab_size, query_vocab_size,
                            user_gender_vocab_size, user_age_vocab_size,
                            item_brand_vocab_size, item_seller_vocab_size, item_cate_vocab_size,
                            item_cate_level1_vocab_size, item_price_vocab_size,
                            max_q_len, max_hist_len,
                            need_neighbors=False, user_neighbor_nums=None, item_neighbor_nums=None):
    all_feature_columns = [SparseFeat('task', vocabulary_size=2, embedding_dim=user_embed_size,
                                      embedding_name='task', dtype='int64', embeddings_initializer=emb_init,
                                      use_hash=False),
                           SparseFeat('user_id', vocabulary_size=user_vocab_size, embedding_dim=user_embed_size,
                                      embedding_name='user_id', dtype='int64', embeddings_initializer=emb_init,
                                      use_hash=False),
                           SparseFeat('item_id', vocabulary_size=item_vocab_size, embedding_dim=item_embed_size,
                                      embedding_name='item_id', dtype='int64', embeddings_initializer=emb_init,
                                      use_hash=False),
                           VarLenSparseFeat(
                               SparseFeat('query', vocabulary_size=query_vocab_size, embedding_dim=query_embed_size,
                                          embedding_name='query_term', dtype='int64', embeddings_initializer=emb_init,
                                          use_hash=False),
                               maxlen=max_q_len),
                           SparseFeat('user_gender', vocabulary_size=user_gender_vocab_size,
                                      embedding_dim=user_feat_embed_size,
                                      embedding_name='user_gender', dtype='int64', embeddings_initializer=emb_init,
                                      use_hash=False),
                           SparseFeat('user_age', vocabulary_size=user_age_vocab_size,
                                      embedding_dim=user_feat_embed_size,
                                      embedding_name='user_age', dtype='int64', embeddings_initializer=emb_init,
                                      use_hash=False),
                           SparseFeat('item_brand', vocabulary_size=item_brand_vocab_size,
                                      embedding_dim=item_feat_embed_size,
                                      embedding_name='item_brand', dtype='int64', embeddings_initializer=emb_init,
                                      use_hash=False),
                           SparseFeat('item_seller', vocabulary_size=item_seller_vocab_size,
                                      embedding_dim=item_feat_embed_size,
                                      embedding_name='item_seller', dtype='int64', embeddings_initializer=emb_init,
                                      use_hash=False),
                           SparseFeat('item_cate', vocabulary_size=item_cate_vocab_size,
                                      embedding_dim=item_feat_embed_size,
                                      embedding_name='item_cate', dtype='int64', embeddings_initializer=emb_init,
                                      use_hash=False),
                           SparseFeat('item_cate_level1', vocabulary_size=item_cate_level1_vocab_size,
                                      embedding_dim=item_feat_embed_size,
                                      embedding_name='item_cate_level1', dtype='int64', embeddings_initializer=emb_init,
                                      use_hash=False),
                           SparseFeat('item_price', vocabulary_size=item_price_vocab_size,
                                      embedding_dim=item_feat_embed_size,
                                      embedding_name='item_price', dtype='int64', embeddings_initializer=emb_init,
                                      use_hash=False),
                           VarLenSparseFeat(
                               SparseFeat('hist', vocabulary_size=item_vocab_size,
                                          embedding_dim=item_embed_size,
                                          embedding_name='item_id', dtype='int64', embeddings_initializer=emb_init,
                                          use_hash=False),
                               maxlen=max_hist_len),
                           VarLenSparseFeat(
                               SparseFeat('hist_query', vocabulary_size=query_vocab_size,
                                          embedding_dim=query_embed_size,
                                          embedding_name='query_term', dtype='int64', embeddings_initializer=emb_init,
                                          use_hash=False),
                               maxlen=max_q_len * max_hist_len),
                           VarLenSparseFeat(
                               SparseFeat('hist_brand', vocabulary_size=item_brand_vocab_size,
                                          embedding_dim=item_feat_embed_size,
                                          embedding_name='item_brand', dtype='int64', embeddings_initializer=emb_init,
                                          use_hash=False),
                               maxlen=max_hist_len),
                           VarLenSparseFeat(
                               SparseFeat('hist_seller', vocabulary_size=item_seller_vocab_size,
                                          embedding_dim=item_feat_embed_size,
                                          embedding_name='item_seller', dtype='int64', embeddings_initializer=emb_init,
                                          use_hash=False),
                               maxlen=max_hist_len),
                           VarLenSparseFeat(
                               SparseFeat('hist_cate', vocabulary_size=item_cate_vocab_size,
                                          embedding_dim=item_feat_embed_size,
                                          embedding_name='item_cate', dtype='int64', embeddings_initializer=emb_init,
                                          use_hash=False),
                               maxlen=max_hist_len),
                           VarLenSparseFeat(
                               SparseFeat('hist_cate_level1', vocabulary_size=item_cate_level1_vocab_size,
                                          embedding_dim=item_feat_embed_size,
                                          embedding_name='item_cate_level1', dtype='int64',
                                          embeddings_initializer=emb_init,
                                          use_hash=False),
                               maxlen=max_hist_len),
                           VarLenSparseFeat(
                               SparseFeat('hist_price', vocabulary_size=item_price_vocab_size,
                                          embedding_dim=item_feat_embed_size,
                                          embedding_name='item_price', dtype='int64', embeddings_initializer=emb_init,
                                          use_hash=False),
                               maxlen=max_hist_len)]
    if not need_neighbors:
        return all_feature_columns

    if len(user_neighbor_nums) >= 1 and np.prod(user_neighbor_nums[:1]) > 0:
        all_feature_columns += [
            VarLenSparseFeat(
                SparseFeat('ui', vocabulary_size=item_vocab_size,
                           embedding_dim=item_embed_size,
                           embedding_name='item_id', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(user_neighbor_nums[:1])),
            VarLenSparseFeat(
                SparseFeat('ui_query', vocabulary_size=query_vocab_size,
                           embedding_dim=query_embed_size,
                           embedding_name='query_term', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=max_q_len*np.prod(user_neighbor_nums[:1])),
            VarLenSparseFeat(
                SparseFeat('ui_brand', vocabulary_size=item_brand_vocab_size,
                           embedding_dim=item_feat_embed_size,
                           embedding_name='item_brand', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(user_neighbor_nums[:1])),
            VarLenSparseFeat(
                SparseFeat('ui_seller', vocabulary_size=item_seller_vocab_size,
                           embedding_dim=item_feat_embed_size,
                           embedding_name='item_seller', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(user_neighbor_nums[:1])),
            VarLenSparseFeat(
                SparseFeat('ui_cate', vocabulary_size=item_cate_vocab_size,
                           embedding_dim=item_feat_embed_size,
                           embedding_name='item_cate', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(user_neighbor_nums[:1])),
            VarLenSparseFeat(
                SparseFeat('ui_cate_level1', vocabulary_size=item_cate_level1_vocab_size,
                           embedding_dim=item_feat_embed_size,
                           embedding_name='item_cate_level1', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(user_neighbor_nums[:1])),
            VarLenSparseFeat(
                SparseFeat('ui_price', vocabulary_size=item_price_vocab_size,
                           embedding_dim=item_feat_embed_size,
                           embedding_name='item_price', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(user_neighbor_nums[:1]))
        ]
    if len(user_neighbor_nums) >= 2 and np.prod(user_neighbor_nums[:2]) > 0:
        all_feature_columns += [
            VarLenSparseFeat(
                SparseFeat('uiu', vocabulary_size=user_vocab_size,
                           embedding_dim=user_embed_size,
                           embedding_name='user_id', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(user_neighbor_nums[:2])),
            VarLenSparseFeat(
                SparseFeat('uiu_query', vocabulary_size=query_vocab_size,
                           embedding_dim=query_embed_size,
                           embedding_name='query_term', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=max_q_len*np.prod(user_neighbor_nums[:2])),
            VarLenSparseFeat(
                SparseFeat('uiu_gender', vocabulary_size=user_gender_vocab_size,
                           embedding_dim=user_feat_embed_size,
                           embedding_name='user_gender', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(user_neighbor_nums[:2])),
            VarLenSparseFeat(
                SparseFeat('uiu_age', vocabulary_size=user_age_vocab_size,
                           embedding_dim=user_feat_embed_size,
                           embedding_name='user_age', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(user_neighbor_nums[:2]))
        ]
    if len(user_neighbor_nums) >= 3 and np.prod(user_neighbor_nums[:3]) > 0:
        all_feature_columns += [
            VarLenSparseFeat(
                SparseFeat('uiui', vocabulary_size=item_vocab_size,
                           embedding_dim=item_embed_size,
                           embedding_name='item_id', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(user_neighbor_nums[:3])),
            VarLenSparseFeat(
                SparseFeat('uiui_query', vocabulary_size=query_vocab_size,
                           embedding_dim=query_embed_size,
                           embedding_name='query_term', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=max_q_len*np.prod(user_neighbor_nums[:3])),
            VarLenSparseFeat(
                SparseFeat('uiui_brand', vocabulary_size=item_brand_vocab_size,
                           embedding_dim=item_feat_embed_size,
                           embedding_name='item_brand', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(user_neighbor_nums[:3])),
            VarLenSparseFeat(
                SparseFeat('uiui_seller', vocabulary_size=item_seller_vocab_size,
                           embedding_dim=item_feat_embed_size,
                           embedding_name='item_seller', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(user_neighbor_nums[:3])),
            VarLenSparseFeat(
                SparseFeat('uiui_cate', vocabulary_size=item_cate_vocab_size,
                           embedding_dim=item_feat_embed_size,
                           embedding_name='item_cate', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(user_neighbor_nums[:3])),
            VarLenSparseFeat(
                SparseFeat('uiui_cate_level1', vocabulary_size=item_cate_level1_vocab_size,
                           embedding_dim=item_feat_embed_size,
                           embedding_name='item_cate_level1', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(user_neighbor_nums[:3])),
            VarLenSparseFeat(
                SparseFeat('uiui_price', vocabulary_size=item_price_vocab_size,
                           embedding_dim=item_feat_embed_size,
                           embedding_name='item_price', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(user_neighbor_nums[:3]))
        ]
    if len(user_neighbor_nums) >= 4 and np.prod(user_neighbor_nums[:4]) > 0:
        all_feature_columns += [
            VarLenSparseFeat(
                SparseFeat('uiuiu', vocabulary_size=user_vocab_size,
                           embedding_dim=user_embed_size,
                           embedding_name='user_id', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(user_neighbor_nums[:4])),
            VarLenSparseFeat(
                SparseFeat('uiuiu_query', vocabulary_size=query_vocab_size,
                           embedding_dim=query_embed_size,
                           embedding_name='query_term', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=max_q_len*np.prod(user_neighbor_nums[:4])),
            VarLenSparseFeat(
                SparseFeat('uiuiu_gender', vocabulary_size=user_gender_vocab_size,
                           embedding_dim=user_feat_embed_size,
                           embedding_name='user_gender', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(user_neighbor_nums[:4])),
            VarLenSparseFeat(
                SparseFeat('uiuiu_age', vocabulary_size=user_age_vocab_size,
                           embedding_dim=user_feat_embed_size,
                           embedding_name='user_age', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(user_neighbor_nums[:4]))
        ]
    if len(item_neighbor_nums) >= 1 and np.prod(item_neighbor_nums[:1]) > 0:
        all_feature_columns += [
            VarLenSparseFeat(
                SparseFeat('iu', vocabulary_size=user_vocab_size,
                           embedding_dim=user_embed_size,
                           embedding_name='user_id', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(item_neighbor_nums[:1])),
            VarLenSparseFeat(
                SparseFeat('iu_query', vocabulary_size=query_vocab_size,
                           embedding_dim=query_embed_size,
                           embedding_name='query_term', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=max_q_len*np.prod(item_neighbor_nums[:1])),
            VarLenSparseFeat(
                SparseFeat('iu_gender', vocabulary_size=user_gender_vocab_size,
                           embedding_dim=user_feat_embed_size,
                           embedding_name='user_gender', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(item_neighbor_nums[:1])),
            VarLenSparseFeat(
                SparseFeat('iu_age', vocabulary_size=user_age_vocab_size,
                           embedding_dim=user_feat_embed_size,
                           embedding_name='user_age', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(item_neighbor_nums[:1]))
        ]
    if len(item_neighbor_nums) >= 2 and np.prod(item_neighbor_nums[:2]) > 0:
        all_feature_columns += [
            VarLenSparseFeat(
                SparseFeat('iui', vocabulary_size=item_vocab_size,
                           embedding_dim=item_embed_size,
                           embedding_name='item_id', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(item_neighbor_nums[:2])),
            VarLenSparseFeat(
                SparseFeat('iui_query', vocabulary_size=query_vocab_size,
                           embedding_dim=query_embed_size,
                           embedding_name='query_term', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=max_q_len*np.prod(item_neighbor_nums[:2])),
            VarLenSparseFeat(
                SparseFeat('iui_brand', vocabulary_size=item_brand_vocab_size,
                           embedding_dim=item_feat_embed_size,
                           embedding_name='item_brand', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(item_neighbor_nums[:2])),
            VarLenSparseFeat(
                SparseFeat('iui_seller', vocabulary_size=item_seller_vocab_size,
                           embedding_dim=item_feat_embed_size,
                           embedding_name='item_seller', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(item_neighbor_nums[:2])),
            VarLenSparseFeat(
                SparseFeat('iui_cate', vocabulary_size=item_cate_vocab_size,
                           embedding_dim=item_feat_embed_size,
                           embedding_name='item_cate', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(item_neighbor_nums[:2])),
            VarLenSparseFeat(
                SparseFeat('iui_cate_level1', vocabulary_size=item_cate_level1_vocab_size,
                           embedding_dim=item_feat_embed_size,
                           embedding_name='item_cate_level1', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(item_neighbor_nums[:2])),
            VarLenSparseFeat(
                SparseFeat('iui_price', vocabulary_size=item_price_vocab_size,
                           embedding_dim=item_feat_embed_size,
                           embedding_name='item_price', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(item_neighbor_nums[:2]))
        ]
    if len(item_neighbor_nums) >= 3 and np.prod(item_neighbor_nums[:3]) > 0:
        all_feature_columns += [
            VarLenSparseFeat(
                SparseFeat('iuiu', vocabulary_size=user_vocab_size,
                           embedding_dim=user_embed_size,
                           embedding_name='user_id', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(item_neighbor_nums[:3])),
            VarLenSparseFeat(
                SparseFeat('iuiu_query', vocabulary_size=query_vocab_size,
                           embedding_dim=query_embed_size,
                           embedding_name='query_term', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=max_q_len*np.prod(item_neighbor_nums[:3])),
            VarLenSparseFeat(
                SparseFeat('iuiu_gender', vocabulary_size=user_gender_vocab_size,
                           embedding_dim=user_feat_embed_size,
                           embedding_name='user_gender', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(item_neighbor_nums[:3])),
            VarLenSparseFeat(
                SparseFeat('iuiu_age', vocabulary_size=user_age_vocab_size,
                           embedding_dim=user_feat_embed_size,
                           embedding_name='user_age', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(item_neighbor_nums[:3]))
        ]
    if len(item_neighbor_nums) >= 4 and np.prod(item_neighbor_nums[:4]) > 0:
        all_feature_columns += [
            VarLenSparseFeat(
                SparseFeat('iuiui', vocabulary_size=item_vocab_size,
                           embedding_dim=item_embed_size,
                           embedding_name='item_id', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(item_neighbor_nums[:4])),
            VarLenSparseFeat(
                SparseFeat('iuiui_query', vocabulary_size=query_vocab_size,
                           embedding_dim=query_embed_size,
                           embedding_name='query_term', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=max_q_len*np.prod(item_neighbor_nums[:4])),
            VarLenSparseFeat(
                SparseFeat('iuiui_brand', vocabulary_size=item_brand_vocab_size,
                           embedding_dim=item_feat_embed_size,
                           embedding_name='item_brand', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(item_neighbor_nums[:4])),
            VarLenSparseFeat(
                SparseFeat('iuiui_seller', vocabulary_size=item_seller_vocab_size,
                           embedding_dim=item_feat_embed_size,
                           embedding_name='item_seller', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(item_neighbor_nums[:4])),
            VarLenSparseFeat(
                SparseFeat('iuiui_cate', vocabulary_size=item_cate_vocab_size,
                           embedding_dim=item_feat_embed_size,
                           embedding_name='item_cate', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(item_neighbor_nums[:4])),
            VarLenSparseFeat(
                SparseFeat('iuiui_cate_level1', vocabulary_size=item_cate_level1_vocab_size,
                           embedding_dim=item_feat_embed_size,
                           embedding_name='item_cate_level1', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(item_neighbor_nums[:4])),
            VarLenSparseFeat(
                SparseFeat('iuiui_price', vocabulary_size=item_price_vocab_size,
                           embedding_dim=item_feat_embed_size,
                           embedding_name='item_price', dtype='int64', embeddings_initializer=emb_init,
                           use_hash=False),
                maxlen=np.prod(item_neighbor_nums[:4]))
        ]
    return all_feature_columns


if __name__ == '__main__':
    all_feature_columns = get_all_feature_columns('', 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, True,
                                                  [1, 1, 1, 1], [1, 1, 1, 1])
    print len(all_feature_columns)

    all_feature_columns = get_all_feature_columns('', 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, False)
    print len(all_feature_columns)
