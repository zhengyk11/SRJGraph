# coding=utf-8
import cPickle
import copy
import json
import logging
from bz2 import BZ2File
import numpy as np


def get_pvtime(user_neighbor_items, i):
    return user_neighbor_items[i]['time']

def split_list(x, max_len, seg='\x03'):
    if len(x.strip()) == 0:
        return [0] * max_len, 0
    x_split = x.split(seg)[:max_len]
    return map(int, x_split) + [0] * (max_len - len(x_split)), len(x_split)

def split_list_query(x, first_len, second_len, first_seg='\x02', second_seg='\x03'):
    out_query = []
    x_split = x.split(first_seg)[:first_len]
    x_split = x_split + [''] * (first_len - len(x_split))
    for _x in x_split:
        if len(_x.strip()) == 0:
            _x = [0] * second_len
        else:
            _x = _x.split(second_seg)[:second_len]
            _x = map(int, _x) + [0] * (second_len - len(_x))
        out_query.append(_x)
    return out_query, first_len * second_len


class Dataset():
    def __init__(self, args, train_files=[], dev_files=[], test_files=[], features={}, need_neighbor=False):
        self.logger = logging.getLogger(args.model_name)
        self.args = args
        self.batch_size, self.max_q_len, self.max_hist_len = args.batch_size, args.max_q_len, args.max_hist_len
        self.user_neighbor_nums, self.item_neighbor_nums = copy.deepcopy(args.user_neighbor_nums), copy.deepcopy(args.item_neighbor_nums)
        self.train_files = train_files
        self.dev_files = dev_files
        self.test_files = test_files
        self.features = features
        self.need_neighbor = need_neighbor
        self.user_feat_table, self.item_feat_table = self.load_feat_tables()
        self.logger.info('Loading graph and feature tables done.')

    def load_feat_tables(self):
        user_feat_table = cPickle.load(open(self.args.user_feat_table_path))
        item_feat_table = cPickle.load(open(self.args.item_feat_table_path))
        return user_feat_table, item_feat_table

    def get_mini_batch(self, mode):
        if mode == 'train':
            data_files = self.train_files
            batch_size = self.batch_size
        elif mode == 'dev':
            data_files = self.dev_files
            batch_size = self.batch_size * 8
        elif mode == 'test':
            data_files = self.test_files
            batch_size = self.batch_size * 8
        else:
            raise Exception('Invalid mode when getting data samples:', mode)

        # for f in data_files:
        #     if self.need_neighbor:
        #         assert f.endswith('.bz2')
        #     if f.endswith('.tsv'):
        #         assert not self.need_neighbor

        # columns = ['info', 'user_id', 'item_id', 'click', 'query', 'user_gender', 'user_age', 'item_brand',
        #            'item_seller', 'item_cate', 'item_cate_level1', 'item_price', 'hist', 'hist_brand', 'hist_seller',
        #            'hist_cate', 'hist_cate_level1', 'hist_price']
        # neighbor_colunms = ['ui', 'uiu', 'uiui', 'uiuiu', 'ui_query', 'uiu_query', 'uiui_query', 'uiuiu_query',
        #                     'ui_brand', 'ui_seller', 'ui_cate', 'ui_cate_level1', 'ui_price',
        #                     'uiu_gender', 'uiu_age',
        #                     'uiui_brand', 'uiui_seller', 'uiui_cate', 'uiui_cate_level1', 'uiui_price',
        #                     'uiuiu_gender', 'uiuiu_age',
        #                     'iu', 'iui', 'iuiu', 'iuiui', 'iu_query', 'iui_query', 'iuiu_query', 'iuiui_query',
        #                     'iu_gender', 'iu_age',
        #                     'iui_brand', 'iui_seller', 'iui_cate', 'iui_cate_level1', 'iui_price',
        #                     'iuiu_gender', 'iuiu_age',
        #                     'iuiui_brand', 'iuiui_seller', 'iuiui_cate', 'iuiui_cate_level1', 'iuiui_price']
        columns = ['info', 'task', 'user_id', 'item_id', 'click', 'query', 'user_gender', 'user_age', 'item_brand',
                   'item_seller', 'item_cate', 'item_price', 'hist', 'hist_query', 'hist_brand', 'hist_seller',
                   'hist_cate', 'hist_price']
        neighbor_colunms = ['ui', 'uiu', 'uiui', 'uiuiu', 'ui_query', 'uiu_query', 'uiui_query', 'uiuiu_query',
                            'ui_brand', 'ui_seller', 'ui_cate', 'ui_price',
                            'uiu_gender', 'uiu_age',
                            'uiui_brand', 'uiui_seller', 'uiui_cate', 'uiui_price',
                            'uiuiu_gender', 'uiuiu_age',
                            'iu', 'iui', 'iuiu', 'iuiui', 'iu_query', 'iui_query', 'iuiu_query', 'iuiui_query',
                            'iu_gender', 'iu_age',
                            'iui_brand', 'iui_seller', 'iui_cate', 'iui_price',
                            'iuiu_gender', 'iuiu_age',
                            'iuiui_brand', 'iuiui_seller', 'iuiui_cate', 'iuiui_price']
        if not self.need_neighbor:
            neighbor_colunms = []
        raw_features_search = dict(zip(columns + neighbor_colunms, [[] for _ in range(len(columns) + len(neighbor_colunms))]))
        raw_features_recommend = copy.deepcopy(raw_features_search)

        data = []
        for f in data_files:
            for line in open(f):
                # if f.endswith('.bz2'):
                #     inputs = json.loads(line)
                # else:
                arr = line.strip('\n').split('\t')
                rn, user_id, query, nid, click, pay, add_cart, add_favorite, pv_time = arr[0:9]
                user_sex, user_age, user_power, user_tag = arr[9:13]
                query_mlr_score = arr[13]
                item_brand_id, item_seller_id, item_cate_id, item_cate_level1_id, py_price_list, r_price, item_tag, hist, hist_query = arr[14:23]

                user_id, nid, click, pv_time, user_sex, user_age, item_brand_id, item_seller_id, item_cate_id, item_cate_level1_id, r_price = \
                    map(int, [user_id, nid, click, pv_time, user_sex, user_age, item_brand_id, item_seller_id,
                              item_cate_id,
                              item_cate_level1_id, r_price])

                query_split, query_len = split_list(query, self.max_q_len)
                inputs = [{'rn': rn, 'pv_time': pv_time}, user_id, nid, click, query_split, query_len]
                if len(self.features) == 0 or (len(self.features) > 0 and 'hist' in self.features):
                    hist_split, hist_len = split_list(hist, self.max_hist_len)
                    inputs += [hist_split, hist_len]
                else:
                    inputs += [[], 0]
                if len(self.features) == 0 or (len(self.features) > 0 and 'hist_query' in self.features):
                    hist_query, _ = split_list_query(hist_query, self.max_hist_len, self.max_q_len)
                    inputs.append(hist_query)
                else:
                    inputs.append([])

                if self.need_neighbor:
                    assert len(arr) == 39
                    ui, uiu, uiui, uiuiu = arr[23:27]
                    ui_query, uiu_query, uiui_query, uiuiu_query = arr[27:31]
                    iu, iui, iuiu, iuiui = arr[31:35]
                    iu_query, iui_query, iuiu_query, iuiui_query = arr[35:39]

                    if len(self.user_neighbor_nums) >= 1 and np.prod(self.user_neighbor_nums[:1]) > 0:
                        ui, _ = split_list(ui, np.prod(self.user_neighbor_nums[:1]))
                        ui_query, _ = split_list_query(ui_query, np.prod(self.user_neighbor_nums[:1]), self.max_q_len)
                    if len(self.user_neighbor_nums) >= 2 and np.prod(self.user_neighbor_nums[:2]) > 0:
                        uiu, _ = split_list(uiu, np.prod(self.user_neighbor_nums[:2]))
                        uiu_query, _ = split_list_query(uiu_query, np.prod(self.user_neighbor_nums[:2]), self.max_q_len)
                    if len(self.user_neighbor_nums) >= 3 and np.prod(self.user_neighbor_nums[:3]) > 0:
                        uiui, _ = split_list(uiui, np.prod(self.user_neighbor_nums[:3]))
                        uiui_query, _ = split_list_query(uiui_query, np.prod(self.user_neighbor_nums[:3]), self.max_q_len)
                    if len(self.user_neighbor_nums) >= 4 and np.prod(self.user_neighbor_nums[:4]) > 0:
                        uiuiu, _ = split_list(uiuiu, np.prod(self.user_neighbor_nums[:4]))
                        uiuiu_query, _ = split_list_query(uiuiu_query, np.prod(self.user_neighbor_nums[:4]), self.max_q_len)

                    if len(self.item_neighbor_nums) >= 1 and np.prod(self.item_neighbor_nums[:1]) > 0:
                        iu, _ = split_list(iu, np.prod(self.item_neighbor_nums[:1]))
                        iu_query, _ = split_list_query(iu_query, np.prod(self.item_neighbor_nums[:1]), self.max_q_len)
                    if len(self.item_neighbor_nums) >= 2 and np.prod(self.item_neighbor_nums[:2]) > 0:
                        iui, _ = split_list(iui, np.prod(self.item_neighbor_nums[:2]))
                        iui_query, _ = split_list_query(iui_query, np.prod(self.item_neighbor_nums[:2]), self.max_q_len)
                    if len(self.item_neighbor_nums) >= 3 and np.prod(self.item_neighbor_nums[:3]) > 0:
                        iuiu, _ = split_list(iuiu, np.prod(self.item_neighbor_nums[:3]))
                        iuiu_query, _ = split_list_query(iuiu_query, np.prod(self.item_neighbor_nums[:3]), self.max_q_len)
                    if len(self.item_neighbor_nums) >= 4 and np.prod(self.item_neighbor_nums[:4]) > 0:
                        iuiui, _ = split_list(iuiui, np.prod(self.item_neighbor_nums[:4]))
                        iuiui_query, _ = split_list_query(iuiui_query, np.prod(self.item_neighbor_nums[:4]), self.max_q_len)

                    inputs += [ui, uiu, uiui, uiuiu, ui_query, uiu_query, uiui_query, uiuiu_query, iu, iui, iuiu, iuiui, iu_query, iui_query, iuiu_query, iuiui_query]
                else:
                    inputs += [[]] * 16
                inputs += [user_sex, user_age, item_brand_id, item_seller_id, item_cate_id, item_cate_level1_id, r_price]
                inputs += [1 if sum(query_split) else 0]
                data.append(inputs)

                if len(data) >= self.batch_size * 256:
                    for inputs in data:
                        if sum(inputs[4]) > 0: # query_split
                            raw_features_search = self.update_raw_features(raw_features_search, inputs)
                        else:
                            raw_features_recommend = self.update_raw_features(raw_features_recommend, inputs)

                        if len(raw_features_recommend['info']) >= batch_size:
                            outs = self._gen_outs(raw_features_recommend, columns)
                            if self.need_neighbor:
                                neighbor_outs = self._gen_outs(raw_features_recommend, neighbor_colunms)
                                outs.update(neighbor_outs)
                            for column in raw_features_recommend:
                                raw_features_recommend[column] = []
                            yield outs

                        if len(raw_features_search['info']) >= batch_size:
                            outs = self._gen_outs(raw_features_search, columns)
                            if self.need_neighbor:
                                neighbor_outs = self._gen_outs(raw_features_search, neighbor_colunms)
                                outs.update(neighbor_outs)
                            for column in raw_features_search:
                                raw_features_search[column] = []
                            yield outs
                    data = []

        if len(data):
            for inputs in data:
                if sum(inputs[4]) > 0:  # query_split
                    raw_features_search = self.update_raw_features(raw_features_search, inputs)
                else:
                    raw_features_recommend = self.update_raw_features(raw_features_recommend, inputs)

                if len(raw_features_recommend['info']) >= batch_size:
                    outs = self._gen_outs(raw_features_recommend, columns)
                    if self.need_neighbor:
                        neighbor_outs = self._gen_outs(raw_features_recommend, neighbor_colunms)
                        outs.update(neighbor_outs)
                    for column in raw_features_recommend:
                        raw_features_recommend[column] = []
                    yield outs

                if len(raw_features_search['info']) >= batch_size:
                    outs = self._gen_outs(raw_features_search, columns)
                    if self.need_neighbor:
                        neighbor_outs = self._gen_outs(raw_features_search, neighbor_colunms)
                        outs.update(neighbor_outs)
                    for column in raw_features_search:
                        raw_features_search[column] = []
                    yield outs
            data = []

        if len(raw_features_recommend['info']):
            outs = self._gen_outs(raw_features_recommend, columns)
            if self.need_neighbor:
                neighbor_outs = self._gen_outs(raw_features_recommend, neighbor_colunms)
                outs.update(neighbor_outs)
            for column in raw_features_recommend:
                raw_features_recommend[column] = []
            yield outs

        if len(raw_features_search['info']):
            outs = self._gen_outs(raw_features_search, columns)
            if self.need_neighbor:
                neighbor_outs = self._gen_outs(raw_features_search, neighbor_colunms)
                outs.update(neighbor_outs)
            for column in raw_features_search:
                raw_features_search[column] = []
            yield outs

    def update_raw_features(self, raw_features, inputs):
        assert len(inputs) == 33
        _user_id, _item_id = inputs[1], inputs[2]
        raw_features['info'].append(inputs[0])
        raw_features['task'].append(inputs[32])
        raw_features['user_id'].append(inputs[1])
        raw_features['item_id'].append(inputs[2])
        raw_features['click'].append(inputs[3])
        raw_features['query'].append(inputs[4])

        # query_len.append(inputs[5])
        # raw_features['user_gender'].append(self.user_feat_table[_user_id]['gender'])
        # raw_features['user_age'].append(self.user_feat_table[_user_id]['age'])
        # raw_features['item_brand'].append(self.item_feat_table[_item_id]['brand'])
        # raw_features['item_seller'].append(self.item_feat_table[_item_id]['seller'])
        # raw_features['item_cate'].append(self.item_feat_table[_item_id]['cate'])
        # # raw_features['item_cate_level1'].append(self.item_feat_table[_item_id]['cate_level1'])
        # raw_features['item_price'].append(self.item_feat_table[_item_id]['price'])

        raw_features['user_gender'].append(inputs[25])
        raw_features['user_age'].append(inputs[26])
        raw_features['item_brand'].append(inputs[27])
        raw_features['item_seller'].append(inputs[28])
        raw_features['item_cate'].append(inputs[29])
        # raw_features['item_cate_level1'].append(inputs[30])
        raw_features['item_price'].append(inputs[31])

        if len(self.features) == 0 or (len(self.features) > 0 and 'hist' in self.features):
            raw_features['hist'].append(inputs[6])
            # hist_len.append(inputs[7])
            raw_features['hist_brand'].append([self.item_feat_table[x]['brand'] if x else 0 for x in inputs[6]])
            raw_features['hist_seller'].append([self.item_feat_table[x]['seller'] if x else 0 for x in inputs[6]])
            raw_features['hist_cate'].append([self.item_feat_table[x]['cate'] if x else 0 for x in inputs[6]])
            # raw_features['hist_cate_level1'].append([self.item_feat_table[x]['cate_level1'] if x else 0 for x in inputs[6]])
            raw_features['hist_price'].append([self.item_feat_table[x]['price'] if x else 0 for x in inputs[6]])

        if len(self.features) == 0 or (len(self.features) > 0 and 'hist_query' in self.features):
            raw_features['hist_query'].append(inputs[8])

        if self.need_neighbor:
            if len(self.user_neighbor_nums) >= 1 and np.prod(self.user_neighbor_nums[:1]) > 0:
                raw_features['ui'].append(inputs[9])
                raw_features['ui_query'].append(inputs[13])
                raw_features['ui_brand'].append([self.item_feat_table[x]['brand'] if x else 0 for x in inputs[9]])
                raw_features['ui_seller'].append([self.item_feat_table[x]['seller'] if x else 0 for x in inputs[9]])
                raw_features['ui_cate'].append([self.item_feat_table[x]['cate'] if x else 0 for x in inputs[9]])
                # raw_features['ui_cate_level1'].append([self.item_feat_table[x]['cate_level1'] if x else 0 for x in inputs[9]])
                raw_features['ui_price'].append([self.item_feat_table[x]['price'] if x else 0 for x in inputs[9]])

            if len(self.user_neighbor_nums) >= 2 and np.prod(self.user_neighbor_nums[:2]) > 0:
                raw_features['uiu'].append(inputs[10])
                raw_features['uiu_query'].append(inputs[14])
                raw_features['uiu_gender'].append([self.user_feat_table[x]['gender'] if x else 0 for x in inputs[10]])
                raw_features['uiu_age'].append([self.user_feat_table[x]['age'] if x else 0 for x in inputs[10]])

            if len(self.user_neighbor_nums) >= 3 and np.prod(self.user_neighbor_nums[:3]) > 0:
                raw_features['uiui'].append(inputs[11])
                raw_features['uiui_query'].append(inputs[15])
                raw_features['uiui_brand'].append([self.item_feat_table[x]['brand'] if x else 0 for x in inputs[11]])
                raw_features['uiui_seller'].append([self.item_feat_table[x]['seller'] if x else 0 for x in inputs[11]])
                raw_features['uiui_cate'].append([self.item_feat_table[x]['cate'] if x else 0 for x in inputs[11]])
                # raw_features['uiui_cate_level1'].append([self.item_feat_table[x]['cate_level1'] if x else 0 for x in inputs[11]])
                raw_features['uiui_price'].append([self.item_feat_table[x]['price'] if x else 0 for x in inputs[11]])

            if len(self.user_neighbor_nums) >= 4 and np.prod(self.user_neighbor_nums[:4]) > 0:
                raw_features['uiuiu'].append(inputs[12])
                raw_features['uiuiu_query'].append(inputs[16])
                raw_features['uiuiu_gender'].append([self.user_feat_table[x]['gender'] if x else 0 for x in inputs[12]])
                raw_features['uiuiu_age'].append([self.user_feat_table[x]['age'] if x else 0 for x in inputs[12]])

            if len(self.item_neighbor_nums) >= 1 and np.prod(self.item_neighbor_nums[:1]) > 0:
                raw_features['iu'].append(inputs[17])
                raw_features['iu_query'].append(inputs[21])
                raw_features['iu_gender'].append([self.user_feat_table[x]['gender'] if x else 0 for x in inputs[17]])
                raw_features['iu_age'].append([self.user_feat_table[x]['age'] if x else 0 for x in inputs[17]])

            if len(self.item_neighbor_nums) >= 2 and np.prod(self.item_neighbor_nums[:2]) > 0:
                raw_features['iui'].append(inputs[18])
                raw_features['iui_query'].append(inputs[22])
                raw_features['iui_brand'].append([self.item_feat_table[x]['brand'] if x else 0 for x in inputs[18]])
                raw_features['iui_seller'].append([self.item_feat_table[x]['seller'] if x else 0 for x in inputs[18]])
                raw_features['iui_cate'].append([self.item_feat_table[x]['cate'] if x else 0 for x in inputs[18]])
                # raw_features['iui_cate_level1'].append([self.item_feat_table[x]['cate_level1'] if x else 0 for x in inputs[18]])
                raw_features['iui_price'].append([self.item_feat_table[x]['price'] if x else 0 for x in inputs[18]])

            if len(self.item_neighbor_nums) >= 3 and np.prod(self.item_neighbor_nums[:3]) > 0:
                raw_features['iuiu'].append(inputs[19])
                raw_features['iuiu_query'].append(inputs[23])
                raw_features['iuiu_gender'].append([self.user_feat_table[x]['gender'] if x else 0 for x in inputs[19]])
                raw_features['iuiu_age'].append([self.user_feat_table[x]['age'] if x else 0 for x in inputs[19]])

            if len(self.item_neighbor_nums) >= 4 and np.prod(self.item_neighbor_nums[:4]) > 0:
                raw_features['iuiui'].append(inputs[20])
                raw_features['iuiui_query'].append(inputs[24])
                raw_features['iuiui_brand'].append([self.item_feat_table[x]['brand'] if x else 0 for x in inputs[20]])
                raw_features['iuiui_seller'].append([self.item_feat_table[x]['seller'] if x else 0 for x in inputs[20]])
                raw_features['iuiui_cate'].append([self.item_feat_table[x]['cate'] if x else 0 for x in inputs[20]])
                # raw_features['iuiui_cate_level1'].append([self.item_feat_table[x]['cate_level1'] if x else 0 for x in inputs[20]])
                raw_features['iuiui_price'].append([self.item_feat_table[x]['price'] if x else 0 for x in inputs[20]])

        return raw_features

    def _gen_outs(self, raw_features, columns):
        current_batch_size = len(raw_features['info'])
        outs, new_columns = [], []
        for column in columns:
            if len(raw_features[column]) != current_batch_size:
                continue
            new_columns.append(column)
            if column in ['info']:
                outs.append(raw_features[column])
            elif '_query' in column:
                outs.append(np.reshape(np.array(raw_features[column], dtype=np.int64), (current_batch_size, -1)))
            else:
                outs.append(np.array(raw_features[column], dtype=np.int64))
        return dict(zip(new_columns, outs))


if __name__ == '__main__':
    data_type = 'all'

    class config():
        def __init__(self):
            self.batch_size = 5
            self.max_q_len = 10
            self.max_hist_len = 25
            self.user_neighbor_nums = [25, 1, 10, 1]
            self.item_neighbor_nums = [1, 25, 1, 10]
            self.log_path = None
            self.user_feat_table_path = '../dataset/dataset_0918/user_feat_table_all.pkl'
            self.item_feat_table_path = '../dataset/dataset_0918/item_feat_table_all.pkl'
            self.model_name = 'dataset'

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

    # d = Dataset(args,
    #             train_files=['../dataset/dataset_0918/all_cate_hist_graph_done/train_cate_hist_graph.tsv.bz2'],
    #             dev_files=['../dataset/dataset_0918/all_cate_hist_graph_done/dev_cate_hist_graph.tsv.bz2'],
    #             test_files=['../dataset/dataset_0918/all_cate_hist_graph_done/test_cate_hist_graph.tsv.bz2'],
    #             need_neighbor=True)
    #
    # cnt = 0
    # for x in d.get_mini_batch('dev'):
    #     for column, _x in x.items():
    #         print column + ':', _x
    #     print
    #     cnt += 1
    #     if cnt == 20:
    #         break


    # d = Dataset(args,
    #             train_files=['../dataset/dataset_0918/all_cate_hist/train_cate_hist.tsv'],
    #             dev_files=['../dataset/dataset_0918/all_cate_hist/dev_cate_hist.tsv'],
    #             test_files=['../dataset/dataset_0918/all_cate_hist/test_cate_hist.tsv'],
    #             need_neighbor=False)

    d = Dataset(args,
                # train_files=['../dataset/dataset_0918/all_cate_hist/train_cate_hist.tsv'],
                dev_files=['../dataset/dataset_0918/all_cate_hist_graph_parts/dev_cate_hist_graph_0.tsv'],
                # test_files=['../dataset/dataset_0918/all_cate_hist/test_cate_hist.tsv'],
                need_neighbor=True)

    cnt = 0
    for x in d.get_mini_batch('dev'):
        # for column, _x in x.items():
        #     print column + ':', _x
        # print
        cnt += 1
        if cnt % 5000 == 0:
            print cnt
