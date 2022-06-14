import os
import cPickle
from datetime import datetime
from datetime import timedelta

user_age_boundaries = [5, 10, 15] + [18, 22, 26, 30, 34, 38, 42, 46, 50] + [55, 60]
r_price_boundaries = [2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                      31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 45, 46, 48, 49, 50, 51, 55, 56, 58, 59, 60, 61,
                      63, 67, 68, 69, 70, 71, 75, 78, 79, 80, 85, 88, 89, 90, 98, 99, 100, 108, 112, 118, 126, 128, 131,
                      138, 139, 148, 149, 158, 159, 168, 175, 185, 198, 199, 225, 242, 268, 288, 300, 349, 389, 449,
                      519, 637, 799, 1060, 1480, 2180, 3850]

out_dir = '../dataset/dataset_0929'
in_dir = '../dataset/dataset_0929_origin'

def update_id_mapping_dict(id_dict, id):
    if len(id.strip()) == 0:
        return id_dict, 0
    if id not in id_dict:
        id_dict[id] = len(id_dict) + 1
    return id_dict, id_dict[id]


def get_raw_feat_mapping_id(raw_value, boundaries):
    assert len(boundaries) > 0
    if len(raw_value.strip()) == 0:
        return 0
    raw_value = float(raw_value)
    boundaries = sorted(boundaries)
    if raw_value > boundaries[-1]:
        return len(boundaries) + 1
    for i in range(len(boundaries) - 1):
        if boundaries[i] < raw_value <= boundaries[i + 1]:
            return i + 2
    return 1

user_id_map, item_id_map, query_term_map = {}, {}, {}
brand_id_map, seller_id_map, cate_id_map, cate_level1_id_map = {}, {}, {}, {}

ds_range = {'back': map(str, range(20200830, 20200832) + range(20200901, 20200913)),
            'train': map(str, range(20200913, 20200920)),
            'dev': map(str, range(20200920, 20200921)),
            'test': map(str, range(20200921, 20200923))}

for data_type in ['search', 'recommend']:
    for data_split in ['back', 'train', 'dev', 'test']:
        if not os.path.exists(out_dir + '/{}/'.format(data_type)):
            os.mkdir(out_dir + '/{}/'.format(data_type))
        out = open(out_dir + '/{}/'.format(data_type) + data_split + '.tsv', 'w')
        for root, dirs, files in os.walk(in_dir + '/{}/{}'.format(data_type, data_split)):
            for fname in sorted(files):
                if not fname.endswith('.tsv'):
                    continue
                print os.path.join(root, fname)
                # ds = fname.replace('.tsv', '').split('_')[-1]
                for line in open(os.path.join(root, fname)):
                    arr = line.strip('\n').split('\t')
                    rn, user_id, query, nid, click, pay, add_cart, add_favorite, pv_time, click_time = arr[0:10]
                    user_sex, user_age, user_power, user_tag = arr[10:14]
                    query_mlr_score = arr[14]
                    item_brand_id, item_seller_id, item_cate_id, item_cate_level1_id, py_price_list, r_price, item_tag = arr[15:22]
                    if pv_time[:8] not in ds_range[data_split]:
                        print 'pv_time:', pv_time
                        continue
                    # if click_time.strip() and click_time[:8] not in ds_range[data_split]:
                    #     print 'click_time:', click_time
                    #     continue
                    # if click_time.strip() and int(click_time) < int(pv_time):
                    #     print 'pv_time:', pv_time, ', click_time:', click_time
                    #     continue
                    if len(user_id.strip()) == 0 or len(nid.strip()) == 0:
                        continue
                    if len(query.strip()) == 0 and data_type == 'search':
                        continue
                    query_terms = query.lower().strip().split('\x03')
                    query_term_ids = []
                    for query_term in query_terms:
                        query_term = query_term.strip()
                        if len(query_term) < 1:
                            continue
                        query_term_map, new_query_term = update_id_mapping_dict(query_term_map, query_term)
                        query_term_ids.append(str(new_query_term))
                    if len(query_term_ids) == 0 and data_type == 'search':
                        continue

                    query_mlr_score = query_mlr_score.strip().split('\x03')
                    query_mlr_score_ids = []
                    for cate in query_mlr_score:
                        cate_id_map, new_cate = update_id_mapping_dict(cate_id_map, cate)
                        query_mlr_score_ids.append(str(new_cate))

                    user_id_map, new_user_id = update_id_mapping_dict(user_id_map, user_id)
                    item_id_map, new_nid = update_id_mapping_dict(item_id_map, nid)
                    brand_id_map, new_item_brand_id = update_id_mapping_dict(brand_id_map, item_brand_id)
                    seller_id_map, new_item_seller_id = update_id_mapping_dict(seller_id_map, item_seller_id)
                    cate_id_map, new_item_cate_id = update_id_mapping_dict(cate_id_map, item_cate_id)
                    cate_level1_id_map, new_item_cate_level1_id = update_id_mapping_dict(cate_level1_id_map, item_cate_level1_id)

                    new_user_age = get_raw_feat_mapping_id(user_age, user_age_boundaries)
                    new_r_price = get_raw_feat_mapping_id(r_price, r_price_boundaries)

                    new_user_id, new_user_age = map(str, [new_user_id, new_user_age])
                    new_nid, new_r_price = map(str, [new_nid, new_r_price])
                    new_item_brand_id, new_item_seller_id = map(str, [new_item_brand_id, new_item_seller_id])
                    new_item_cate_id, new_item_cate_level1_id = map(str, [new_item_cate_id, new_item_cate_level1_id])

                    new_line = [rn, new_user_id, '\x03'.join(query_term_ids), new_nid, click, pay, add_cart, add_favorite, pv_time,
                                user_sex, new_user_age, user_power, user_tag,
                                '\03'.join(query_mlr_score_ids),
                                new_item_brand_id, new_item_seller_id, new_item_cate_id, new_item_cate_level1_id, py_price_list, new_r_price, item_tag]

                    out.write('\t'.join(new_line) + '\n')
        out.close()

print len(user_id_map)+1, len(item_id_map)+1
print len(query_term_map)+1
print 3, len(user_age_boundaries) + 2
print len(brand_id_map)+1, len(seller_id_map)+1, len(cate_id_map)+1, len(cate_level1_id_map)+1, len(r_price_boundaries) + 2

'''
82888 9621326
98404
3 16
169267 722971 737 13 101
'''


'''
20800 4755765
52035
3 16
128557 485032 670 13 101
'''


with open(out_dir + '/user_item_id_file.cpkl', 'w') as _out:
    cPickle.dump(user_id_map, _out, protocol=0)
with open(out_dir + '/query_term_id_file.cpkl', 'w') as _out:
    cPickle.dump(query_term_map, _out, protocol=0)
