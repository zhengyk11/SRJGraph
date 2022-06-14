# -*- coding:utf-8 -*-
# @Organization: Alibaba
# @Author: Yukun Zheng
# @Email: zyk265182@alibaba-inc.com
# @Time: 2020/9/18 13:37

import os
import cPickle


def gen_graph(data_paths, tag):
    user_cate_item_table = {}
    item_user_table = {}
    user_feat_table = {}
    item_feat_table = {}
    file_list = []
    for f in data_paths:
        # for root, dirs, files in os.walk(path):
        # for f in files:
        if not f.endswith('.tsv'):
            continue
        file_list.append(f)

    for f in file_list:
        print f
        for line in open(f):
            arr = line.split('\t')
            rn, user_id, query, nid, click, pay, add_cart, add_favorite, pv_time = arr[:9]
            user_sex, user_age, user_power, user_tag = arr[9:13]
            query_mlr_score = arr[13]
            item_brand_id, item_seller_id, item_cate_id, item_cate_level1_id, py_price_list, r_price, item_tag = arr[14:21]
            if query.strip() == '':
                query = []
            else:
                query = map(int, query.split('\x03'))

            user_id, nid, click, pv_time, user_sex, user_age, item_brand_id, item_seller_id, item_cate_id, item_cate_level1_id, r_price = \
                map(int, [user_id, nid, click, pv_time, user_sex, user_age, item_brand_id, item_seller_id, item_cate_id,
                          item_cate_level1_id, r_price])

            # if user_id not in user_feat_table:
            #     user_feat_table[user_id] = {'user': user_id, 'gender': user_sex, 'age': user_age}
            #
            # # item table
            # if nid not in item_feat_table:
            #     item_feat_table[nid] = {'item': nid, 'brand': item_brand_id, 'seller': item_seller_id,
            #                             'cate': item_cate_id, 'cate_level1': item_cate_level1_id, 'price': r_price}

            if click == 0:
                continue
            # user table
            item_cate_level1_id = 0
            if user_id not in user_cate_item_table:
                user_cate_item_table[user_id] = {}
            if item_cate_level1_id not in user_cate_item_table[user_id]:
                user_cate_item_table[user_id][item_cate_level1_id] = []
            user_cate_item_table[user_id][item_cate_level1_id].append(
                {'item': nid, 'time': pv_time, 'query': query})

            # item table
            if nid not in item_user_table:
                item_user_table[nid] = []
            item_user_table[nid].append({'user': user_id, 'time': pv_time, 'query': query})

    # output = open('../dataset/dataset_0929/user_cate_item_table_{}.txt'.format(tag), 'w')
    for user in user_cate_item_table:
        for cate in user_cate_item_table[user]:
            user_cate_item_table[user][cate] = sorted(user_cate_item_table[user][cate], key=lambda x:x['time'])
            # for item in user_cate_item_table[user][cate]:
            #     output.write('{}\t{}\t{}\t{}\t{}\n'.format(user, cate, item['item'], item['time'], ' '.join(map(str, item['query']))))
    # output.close()

    # output = open('../dataset/dataset_0929/item_user_table_{}.txt'.format(tag), 'w')
    for item in item_user_table:
        item_user_table[item] = sorted(item_user_table[item], key=lambda x:x['time'])
        # for user in item_user_table[item]:
        #     output.write('{}\t{}\t{}\t{}\t{}\n'.format(item, item_feat_table[item]['cate_level1'], user['user'], user['time'], ' '.join(map(str, user['query']))))
    # output.close()

    print 'done'
    print len(user_feat_table), len(item_feat_table)
    return user_cate_item_table, item_user_table, user_feat_table, item_feat_table


def cal_inout_degree(user_cate_item_table, item_user_table):
    do = []
    for user in user_cate_item_table:
        cnt = 0
        for cate in user_cate_item_table[user]:
            for item in user_cate_item_table[user][cate]:
                cnt += 1  # user_cate_item_table[user][cate][item]['click_cnt']
        do.append(cnt)
    print 1. * sum(do) / len(do)

    do = []
    for item in item_user_table:
        cnt = 0
        for user in item_user_table[item]:
            cnt += 1  # item_user_table[item][user]['click_cnt']
        do.append(cnt)
    print 1. * sum(do) / len(do)

def main():
    user_cate_item_table, item_user_table, user_feat_table, item_feat_table = gen_graph(
        ['../dataset/dataset_0929/search/back.tsv',
         '../dataset/dataset_0929/search/train.tsv',
         '../dataset/dataset_0929/search/dev.tsv',
         '../dataset/dataset_0929/search/test.tsv',
         '../dataset/dataset_0929/recommend/back.tsv',
         '../dataset/dataset_0929/recommend/train.tsv',
         '../dataset/dataset_0929/recommend/dev.tsv',
         '../dataset/dataset_0929/recommend/test.tsv'],
        'all')

    cal_inout_degree(user_cate_item_table, item_user_table)

    with open('../dataset/dataset_0929/user_cate_item_table_all_2.pkl', 'w') as _out:
        cPickle.dump(user_cate_item_table, _out, protocol=0)
    with open('../dataset/dataset_0929/item_user_table_all_2.pkl', 'w') as _out:
        cPickle.dump(item_user_table, _out, protocol=0)
    # with open('../dataset/dataset_0929/user_feat_table_all.pkl', 'w') as _out:
    #     cPickle.dump(user_feat_table, _out, protocol=0)
    # with open('../dataset/dataset_0929/item_feat_table_all.pkl', 'w') as _out:
    #     cPickle.dump(item_feat_table, _out, protocol=0)


    user_cate_item_table, item_user_table, _, _ = gen_graph(
        ['../dataset/dataset_0929/search/back.tsv',
         '../dataset/dataset_0929/search/train.tsv',
         '../dataset/dataset_0929/search/dev.tsv',
         '../dataset/dataset_0929/search/test.tsv'],
    'search')

    cal_inout_degree(user_cate_item_table, item_user_table)

    with open('../dataset/dataset_0929/user_cate_item_table_search_2.pkl', 'w') as _out:
        cPickle.dump(user_cate_item_table, _out, protocol=0)
    with open('../dataset/dataset_0929/item_user_table_search_2.pkl', 'w') as _out:
        cPickle.dump(item_user_table, _out, protocol=0)


    user_cate_item_table, item_user_table, _, _ = gen_graph(
        ['../dataset/dataset_0929/recommend/back.tsv',
         '../dataset/dataset_0929/recommend/train.tsv',
         '../dataset/dataset_0929/recommend/dev.tsv',
         '../dataset/dataset_0929/recommend/test.tsv'],
    'recommend')

    cal_inout_degree(user_cate_item_table, item_user_table)

    with open('../dataset/dataset_0929/user_cate_item_table_recommend_2.pkl', 'w') as _out:
        cPickle.dump(user_cate_item_table, _out, protocol=0)
    with open('../dataset/dataset_0929/item_user_table_recommend_2.pkl', 'w') as _out:
        cPickle.dump(item_user_table, _out, protocol=0)



if __name__ == '__main__':
    main()
