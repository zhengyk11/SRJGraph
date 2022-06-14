# -*- coding:utf-8 -*-
# @Organization: Alibaba
# @Author: Yukun Zheng
# @Email: zyk265182@alibaba-inc.com
# @Time: 2020/9/22 20:02

import copy
import json
import os
import logging
import random
import math
import sys
from bz2 import BZ2File
import time

import pandas as pd
import cPickle
import numpy as np
from binary_search import binarySearch

def get_pvtime(user_neighbor_items, i):
    return user_neighbor_items[i]['time']

class Dataset():
    def __init__(self, args, train_files=[], dev_files=[], test_files=[], need_neighbor=False):
        self.logger = logging.getLogger(args.model_name)
        self.args = args
        self.batch_size, self.max_q_len, self.max_hist_len = args.batch_size, args.max_q_len, args.max_hist_len
        self.user_neighbor_nums, self.item_neighbor_nums = copy.deepcopy(args.user_neighbor_nums), copy.deepcopy(args.item_neighbor_nums)
        self.train_files = train_files
        self.dev_files = dev_files
        self.test_files = test_files
        self.need_neighbor = need_neighbor

        self.user_cate_item_table, self.item_user_table = self.load_graph_and_feat_tables()
        self.logger.info('Loading graph and feature tables done.')
        self.train_set, self.dev_set, self.test_set = [], [], []
        if train_files:
            for f in train_files:
                self.train_set += self.load_data(f)
                # random.shuffle(self.train_set)
            self.logger.info('Train set size: {} samples.'.format(len(self.train_set)))
        if dev_files:
            for f in dev_files:
                self.dev_set += self.load_data(f)
            self.logger.info('Dev set size: {} samples.'.format(len(self.dev_set)))
        if test_files:
            for f in test_files:
                self.test_set += self.load_data(f)
            self.logger.info('Test set size: {} samples.'.format(len(self.test_set)))

    def load_graph_and_feat_tables(self):
        # user_feat_table      = cPickle.load(open(self.args.user_feat_table_path))
        # item_feat_table      = cPickle.load(open(self.args.item_feat_table_path))
        user_cate_item_table = cPickle.load(open(self.args.user_cate_item_table_path))
        item_user_table      = cPickle.load(open(self.args.item_user_table_path))
        return user_cate_item_table, item_user_table # , user_feat_table, item_feat_table

    def get_node_neighbors(self, node_id, node_type, cate_level1_id, pv_time, neighbor_num):
        node_id, cate_level1_id, pv_time, neighbor_num = map(int, [node_id, cate_level1_id, pv_time, neighbor_num])
        if node_type == 'user':
            user_id = node_id
            if (user_id in self.user_cate_item_table) and (cate_level1_id in self.user_cate_item_table[user_id]):
                user_neighbor_items_all = self.user_cate_item_table[user_id][cate_level1_id]
                # user_neighbor_items_all = sorted(self.user_cate_item_table[user_id][cate_level1_id], key=lambda x:x['time']) # already sorted when generating pickle
                user_neighbor_items_all = binarySearch(user_neighbor_items_all, 0, len(user_neighbor_items_all) - 1,
                                                       pv_time, get_pvtime)
                for x in user_neighbor_items_all:
                    assert x['time'] < pv_time
                if user_neighbor_items_all and neighbor_num > 0:
                    user_neighbor_items = np.random.choice(user_neighbor_items_all, neighbor_num).tolist()
                else:
                    return '\x03'.join(['0'] * neighbor_num), '\x02'.join([''] * neighbor_num)
            else:
                return '\x03'.join(['0'] * neighbor_num), '\x02'.join([''] * neighbor_num)
            items, queries = [], []
            for x in sorted(user_neighbor_items, key=lambda x: -x['time']):
                items.append(str(x['item']))
                x_query = map(str, x['query'])
                # x_query = map(str, x['query'][:self.max_q_len])
                # x_query = x_query + ['0'] * (self.max_q_len - len(x_query))
                queries.append('\x03'.join(x_query))
            return '\x03'.join(items), '\x02'.join(queries)
        elif node_type == 'item':
            item_id = node_id
            if item_id in self.item_user_table:
                item_neighbor_users_all = self.item_user_table[item_id]
                # item_neighbor_users_all = sorted(self.item_user_table[item_id], key=lambda x: x['time']) # already sorted when generating pickle
                item_neighbor_users_all = binarySearch(item_neighbor_users_all, 0, len(item_neighbor_users_all) - 1,
                                                       pv_time, get_pvtime)
                for x in item_neighbor_users_all:
                    assert x['time'] < pv_time
                if item_neighbor_users_all and neighbor_num > 0:
                    item_neighbor_users = np.random.choice(item_neighbor_users_all, neighbor_num).tolist()
                else:
                    return '\x03'.join(['0'] * neighbor_num), '\x02'.join([''] * neighbor_num)
            else:
                return '\x03'.join(['0'] * neighbor_num), '\x02'.join([''] * neighbor_num)
            users, queries = [], []
            for x in sorted(item_neighbor_users, key=lambda x: -x['time']):
                users.append(str(x['user']))
                x_query = map(str, x['query'])
                # x_query = map(str, x['query'][:self.max_q_len])
                # x_query = x_query + ['0'] * (self.max_q_len - len(x_query))
                queries.append('\x03'.join(x_query))
            return '\x03'.join(users), '\x02'.join(queries)
        else:
            raise BaseException('get_user_neighbor_items')

    def get_all_neighbors(self, node_id, node_type, cate_level1_id, pv_time, neighbor_nums):
        if len(neighbor_nums) == 0:
            return []
        neighbors, queries = self.get_node_neighbors(node_id, node_type, cate_level1_id, pv_time, neighbor_nums[0])
        src, tmp_src, tmp_query, outs, outs_query = neighbors, [], [], [neighbors], [queries]
        new_node_type = copy.deepcopy(node_type)
        for i in range(1, 4): # len(neighbor_nums)):
            new_node_type = 'item' if new_node_type == 'user' else 'user'
            for node in src.split('\x03'):
                node = 0 if node == '' else int(node)
                if node == 0 and i < len(neighbor_nums):
                    neighbors, queries = '\x03'.join(['0'] * neighbor_nums[i]), '\x02'.join([''] * neighbor_nums[i])
                elif node > 0 and i < len(neighbor_nums):
                    neighbors, queries = self.get_node_neighbors(node, new_node_type, cate_level1_id, pv_time, neighbor_nums[i])
                else:
                    neighbors, queries = '', ''
                tmp_query.append(queries)
                tmp_src.append(neighbors)
            outs.append('\x03'.join(tmp_src))
            outs_query.append('\x02'.join(tmp_query))
            src = '\x03'.join(tmp_src)
            tmp_src, tmp_query = [], []
        return '\t'.join(outs) + '\t' + '\t'.join(outs_query)

    def load_data(self, data_path):
        def split_list(x, max_len):
            if len(x.strip()) == 0:
                return [0] * max_len, 0
            x_split = x.split('\x03')[:max_len]
            return map(int, x_split) + [0] * (max_len - len(x_split)), len(x_split)

        data = []

        # lines = []
        cnt = -1
        for line in open(data_path):
        #     lines.append(line)
        # # random.shuffle(lines)
        # for line in lines:
            cnt += 1
            if cnt % 5000 == 0:
                print cnt

            if cnt % line_split != line_num:
                continue
            line = line.strip('\n')
            arr = line.split('\t')
            rn, user_id, query, nid, click, pay, add_cart, add_favorite, pv_time = arr[0:9]
            user_sex, user_age, user_power, user_tag = arr[9:13]
            query_mlr_score = arr[13]
            item_brand_id, item_seller_id, item_cate_id, item_cate_level1_id, py_price_list, r_price, item_tag, hist = arr[14:22]

            user_id, nid, click, pv_time, user_sex, user_age, item_brand_id, item_seller_id, item_cate_id, item_cate_level1_id, r_price = \
                map(int, [user_id, nid, click, pv_time, user_sex, user_age, item_brand_id, item_seller_id, item_cate_id,
                          item_cate_level1_id, r_price])

            # if not str(pv_time).startswith(ds):
            #     # print pv_time
            #     continue

            # query_split, query_len = split_list(query, self.max_q_len)
            # hist_split, hist_len = split_list(hist, self.max_hist_len)
            # sample = [rn+'__'+str(pv_time), user_id, nid, click, query_split, query_len]
            # sample += [hist_split, hist_len]
            if self.need_neighbor:
                line += '\t' + self.get_all_neighbors(user_id, 'user', 0, pv_time, self.user_neighbor_nums)
                line += '\t' + self.get_all_neighbors(nid, 'item', 0, pv_time, self.item_neighbor_nums)

            # columns = ['info', 'user_id', 'item_id', 'click', 'query', 'query_len', 'hist', 'hist_len',
            #            'hist_seller', 'hist_cate', 'hist_cate_level1', 'hist_price']
            output.write(line + '\n')
            data.append(1)
        return data

if __name__ == '__main__':

    data_type = sys.argv[1]
    data_split = sys.argv[2]
    line_split = int(sys.argv[3])
    line_num = int(sys.argv[4])

    # data_type = 'all'
    # data_split = 'dev'
    # line_split = 1
    # line_num = 0

    class config():
        def __init__(self):
            self.batch_size = 1
            self.max_q_len = 10
            self.max_hist_len = 25
            self.user_neighbor_nums = [15, 1, 15]
            self.item_neighbor_nums = [1, 15, 1]
            
            # self.user_neighbor_nums = [10, 1, 10]
            # self.item_neighbor_nums = [1, 10, 1]
            
            # self.user_neighbor_nums = [10, 2, 10]
            # self.item_neighbor_nums = [2, 10, 2]
            
            # self.user_neighbor_nums = [10, 3, 10]
            # self.item_neighbor_nums = [3, 10, 3]
            
            # self.user_neighbor_nums = [25, 1, 10]
            # self.item_neighbor_nums = [1, 25, 1]
            
            self.log_path = None
            # self.user_feat_table_path = '../dataset/dataset_0929/user_feat_table_all.pkl'
            # self.item_feat_table_path = '../dataset/dataset_0929/item_feat_table_all.pkl'
            self.user_cate_item_table_path = '../dataset/dataset_0929/user_cate_item_table_{}_2.pkl'.format(data_type)
            self.item_user_table_path = '../dataset/dataset_0929/item_user_table_{}_2.pkl'.format(data_type)
            self.model_name = 'Dataset'

    args = config()
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    out_dir = '../dataset/dataset_0929/{}_hist_graph_parts_{}_{}'.format(data_type, '-'.join(map(str, args.user_neighbor_nums)), '-'.join(map(str, args.item_neighbor_nums)))
    time.sleep(line_num)
    if line_num == 0 and not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # data_split = 'dev'
    # ds = '20200822'
    assert data_type in ['all', 'search', 'recommend']
    assert data_split in ['train', 'dev', 'test']
    assert line_split in range(52)

    if line_num in range(line_split): # map(str, range(20200820, 20200832) + range(20200901, 20200913))

        output = open(out_dir + '/{}_hist_graph_{}.tsv'.format(data_split, line_num), 'w')

        if data_split == 'train':
            d = Dataset(args,
                        train_files=['../dataset/dataset_0929/{}_hist/train_hist.tsv'.format(data_type)],
                        need_neighbor=True)
        elif data_split == 'dev':
            d = Dataset(args,
                        dev_files=['../dataset/dataset_0929/{}_hist/dev_hist.tsv'.format(data_type)],
                        need_neighbor=True)
        else:
            d = Dataset(args,
                        test_files=['../dataset/dataset_0929/{}_hist/test_hist.tsv'.format(data_type)],
                        need_neighbor=True)

        output.close()
    else:
        output = open(out_dir + '/{}_hist_graph.tsv'.format(data_split), 'w')
        for root, dirs, files in os.walk(out_dir):
            files = filter(lambda x:x.endswith('.tsv') and x.startswith('{}_hist_graph_'.format(data_split)), files)
            assert len(files) == line_split
            for f in files:
                for line in open(os.path.join(root, f)):
                    output.write(line)
        output.close()

