# -*- coding:utf-8 -*-
# @Organization: Alibaba
# @Author: Yukun Zheng
# @Email: zyk265182@alibaba-inc.com
# @Time: 2020/10/12 18:01


import os
import pandas as pd
import numpy as np
import random

def get_all(data_tyle_list):
    max_hist_len = 100
    uid_cate_hist = {}
    user_dict = {}
    item_dict = {}
    query_dict = {}
    hist_len = []
    for f in ['back.tsv', 'train.tsv', 'dev.tsv', 'test.tsv']:
        lines = []
        all_data = []
        for data_type in data_tyle_list:
            print data_type, f
            dir = '../dataset/dataset_0929/{}/'.format(data_type)
            all_data += pd.read_csv(os.path.join(dir, f), header=None, sep='\t', index_col=False,
                                    dtype=str).values.tolist()

        # all_data = sorted(all_data, key=lambda x: x[8])
        cnt = 0
        for arr in all_data:
            for i, x in enumerate(arr):
                if not isinstance(x, str):
                    arr[i] = ''
            rn, uid, query, nid, click, pay, add_cart, add_favorite, pv_time = arr[0:9]
            # user_sex, user_age, user_power, user_tag = arr[9:13]
            # query_mlr_score = arr[13]
            # item_brand_id, item_seller_id, item_cate_id, item_cate_level1_id, py_price_list, r_price, item_tag = arr[14:21]

            user_dict[uid] = 0
            item_dict[nid] = 0
            query_dict[query] = 0

    print len(user_dict), len(item_dict), len(query_dict)

def data_analysis():
    sea_user_rn_dict = {}
    rec_user_rn_dict = {}

    sea_user_nid_dict = {}
    rec_user_nid_dict = {}

    user_dict = {}
    item_dict = {}
    query_dict = {}
    search_data = []
    rec_data = []
    # for f in ['back.tsv', 'train.tsv', 'dev.tsv', 'test.tsv']:
    for root, dir, files in os.walk('../dataset/dataset_0929/search'):
        for f in files:
            print os.path.join(root, f)
        # lines = []
        # all_data = []
        # for data_type in ['search', 'recommend']:
        #     print data_type, f
        #     dir = '../dataset/dataset_0929/{}/'.format(data_type)
        #     search_data += pd.read_csv(os.path.join(root, f), header=None, sep='\t', index_col=False,
        #                             dtype=str).values.tolist()
            for line in open(os.path.join(root, f)):
                arr = line.split('\t')
                for i, x in enumerate(arr):
                    if not isinstance(x, str):
                        arr[i] = ''
                rn, uid, query, nid, click, pay, add_cart, add_favorite, pv_time = arr[0:9]
                user_dict[uid] = 0
                item_dict[nid] = 0
                query_dict[query] = 0

                # if data_type == 'search':
                if uid not in sea_user_rn_dict:
                    sea_user_rn_dict[uid] = {}
                if rn not in sea_user_rn_dict[uid]:
                    sea_user_rn_dict[uid][rn] = 0

                if uid not in sea_user_nid_dict:
                    sea_user_nid_dict[uid] = {}
                if click == '1' and nid not in sea_user_nid_dict[uid]:
                    sea_user_nid_dict[uid][nid] = 0

                if click == '1':
                    sea_user_rn_dict[uid][rn] += 1


    for root, dir, files in os.walk('../dataset/dataset_0929/recommend'):
        for f in files:
            print os.path.join(root, f)
        # lines = []
        # all_data = []
        # for data_type in ['search', 'recommend']:
        #     print data_type, f
        #     dir = '../dataset/dataset_0929/{}/'.format(data_type)
            for line in open(os.path.join(root, f)):
                arr = line.split('\t')
                for i, x in enumerate(arr):
                    if not isinstance(x, str):
                        arr[i] = ''
                rn, uid, query, nid, click, pay, add_cart, add_favorite, pv_time = arr[0:9]
                user_dict[uid] = 0
                item_dict[nid] = 0
                query_dict[query] = 0

                # if data_type == 'search':
                if uid not in rec_user_rn_dict:
                    rec_user_rn_dict[uid] = {}
                if rn not in rec_user_rn_dict[uid]:
                    rec_user_rn_dict[uid][rn] = 0

                if uid not in rec_user_nid_dict:
                    rec_user_nid_dict[uid] = {}
                if click == '1' and nid not in rec_user_nid_dict[uid]:
                    rec_user_nid_dict[uid][nid] = 0

                if click == '1':
                    rec_user_rn_dict[uid][rn] += 1

    sea_user_avg_rn_num = []
    for uid in sea_user_rn_dict:
        tmp = 0
        for rn in sea_user_rn_dict[uid]:
            tmp += 1
        sea_user_avg_rn_num.append(tmp)

    rec_user_avg_rn_num = []
    for uid in rec_user_rn_dict:
        tmp = 0
        for rn in rec_user_rn_dict[uid]:
            tmp += 1
        rec_user_avg_rn_num.append(tmp)

    overlap = []
    sea_click_num = []
    rec_click_num = []
    for uid in sea_user_nid_dict:
        if uid not in rec_user_nid_dict:
            continue
        and_num = 0
        sea_click_num.append(len(sea_user_nid_dict[uid]))
        rec_click_num.append(len(rec_user_nid_dict[uid]))
        all_num = len(list(set(sea_user_nid_dict[uid].keys() + rec_user_nid_dict[uid].keys())))
        for nid in sea_user_nid_dict[uid]:
            if nid in rec_user_nid_dict[uid]:
                and_num += 1
        overlap.append(1. * and_num / all_num)

    print 1. * sum(overlap) / len(overlap)
    print 1. * sum(sea_click_num) / len(sea_click_num)
    print 1. * sum(rec_click_num) / len(rec_click_num)
    print 1. * sum(sea_user_avg_rn_num) / len(sea_user_avg_rn_num), 1. * sum(rec_user_avg_rn_num) / len(rec_user_avg_rn_num)


if __name__ == '__main__':
    data_analysis()
    # get_all(['search', 'recommend'])
    # get_all(['search'])
    # get_all(['recommend'])
