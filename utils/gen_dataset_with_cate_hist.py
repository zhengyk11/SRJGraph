# -*- coding:utf-8 -*-
# @Organization: Alibaba
# @Author: Yukun Zheng
# @Email: zyk265182@alibaba-inc.com
# @Time: 2020/10/16 13:01

import os
import pandas as pd
import numpy as np
import random


def get_all(data_type_list):
    max_hist_len = 100
    uid_cate_hist = {}
    hist_len = []
    for f in ['back.tsv', 'train.tsv', 'dev.tsv', 'test.tsv']:
        lines = []
        all_data = []
        for data_type in data_type_list:
            print data_type, f
            dir = '../dataset/dataset_0929/{}/'.format(data_type)
            all_data += pd.read_csv(os.path.join(dir, f), header=None, sep='\t', index_col=False,
                                    dtype=str).values.tolist()

        all_data = sorted(all_data, key=lambda x: x[8])
        cnt = 0
        for arr in all_data:
            for i, x in enumerate(arr):
                if not isinstance(x, str):
                    arr[i] = ''
            rn, uid, query, nid, click, pay, add_cart, add_favorite, pv_time = arr[0:9]
            user_sex, user_age, user_power, user_tag = arr[9:13]
            query_mlr_score = arr[13]
            item_brand_id, item_seller_id, item_cate_id, item_cate_level1_id, py_price_list, r_price, item_tag = arr[14:21]
            cnt += 1
            if cnt % 200000 == 0:
                print cnt
            pv_time = int(pv_time)
            if uid not in uid_cate_hist:
                uid_cate_hist[uid] = {}
            if item_cate_level1_id not in uid_cate_hist[uid]:
                uid_cate_hist[uid][item_cate_level1_id] = []

            idx = 0
            tag = False
            hist_cate = uid_cate_hist[uid][item_cate_level1_id]
            for i, (hist_i, hist_pv_time, hist_q) in enumerate(hist_cate):
                idx = i
                if hist_pv_time < pv_time:
                    tag = True
                    break
            if tag:
                hist, hist_query = [], []
                for hist_i, _, hist_q in hist_cate[idx: idx + max_hist_len]:
                    hist.append(hist_i)
                    hist_query.append(hist_q)
                lines.append('\t'.join(arr) + '\t' + '\x03'.join(hist) + '\t' + '\x02'.join(hist_query) + '\n')
                hist_len.append(len(hist))
            else:
                lines.append('\t'.join(arr) + '\t\t\n')
                hist_len.append(0)
            if int(click) == 1:
                hist_cate = [[nid, pv_time, query]] + hist_cate
                if len(hist_cate) > 1 and pv_time < hist_cate[1][1]:
                    hist_cate = sorted(hist_cate, key=lambda x: -x[1])
                uid_cate_hist[uid][item_cate_level1_id] = hist_cate

        if len(data_type_list) == 1:
            out_dir = '../dataset/dataset_0929/{}_cate_hist'.format(data_type_list[0])
        else:
            out_dir = '../dataset/dataset_0929/all_cate_hist'

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        output = open(out_dir + '/{}_cate_hist.tsv'.format(f.replace('.tsv', '')), 'w')
        indices = range(0, len(lines))
        random.shuffle(indices)
        for ind in indices:
            output.write(lines[ind])
        output.close()
    print ' and '.join(data_type_list), 'done'
    print np.mean(hist_len)
    print


if __name__ == '__main__':
    get_all(['search', 'recommend'])
    get_all(['search'])
    get_all(['recommend'])
    # get_split('recommend')
    # get_split('search')
