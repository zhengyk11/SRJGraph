import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
sys.path.append('..')

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import argparse
import logging
from inputs.dataset import Dataset
from models import *
import tensorflow as tf
from tensorflow.python.keras import backend as K
import random
import numpy as np

random.seed(1024)
np.random.seed(1024)
tf.set_random_seed(1024)

def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('SGM')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')
    # parser.add_argument('--gpu', type=str, default='',
    #                     help='specify gpu device')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', default='sgm_v2',
                                help='choose the algorithm to use')
    model_settings.add_argument('--model_name', default='search_graph_model',
                                help='choose the algorithm to use')
    model_settings.add_argument('--user_embed_size', type=int, default=16,
                                help='size of the embeddings')
    model_settings.add_argument('--item_embed_size', type=int, default=16,
                                help='size of the embeddings')
    model_settings.add_argument('--query_embed_size', type=int, default=16,
                                help='size of the embeddings')
    model_settings.add_argument('--user_feat_embed_size', type=int, default=4,
                                help='size of the embeddings')
    model_settings.add_argument('--item_feat_embed_size', type=int, default=8,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', nargs='+', type=int, default=[256, 64],
                                help='size of LSTM hidden units')

    model_settings.add_argument('--user_vocab_size', type=int, default=82888)
    model_settings.add_argument('--item_vocab_size', type=int, default=9621326)
    model_settings.add_argument('--query_vocab_size', type=int, default=98404)
    model_settings.add_argument('--user_gender_vocab_size', type=int, default=3)
    model_settings.add_argument('--user_age_vocab_size', type=int, default=16)
    model_settings.add_argument('--item_brand_vocab_size', type=int, default=169267)
    model_settings.add_argument('--item_seller_vocab_size', type=int, default=722971)
    model_settings.add_argument('--item_cate_vocab_size', type=int, default=737)
    model_settings.add_argument('--item_cate_level1_vocab_size', type=int, default=13)
    model_settings.add_argument('--item_price_vocab_size', type=int, default=101)

    model_settings.add_argument('--max_q_len', type=int, default=10,
                                help='max term number of query')
    model_settings.add_argument('--max_hist_len', type=int, default=25,
                                help='max number of docs in a session')

    model_settings.add_argument('--user_neighbor_nums', nargs='+', type=int, default=[25, 1, 10],
                                help='define how many neighbors are sampled at the first and second orders')
    model_settings.add_argument('--item_neighbor_nums', nargs='+', type=int, default=[1, 25, 1],
                                help='define how many neighbors are sampled at the first and second orders')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--opt', default='adagrad',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.01,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--momentum', type=float, default=0.99,
                                help='momentum')
    train_settings.add_argument('--dropout_rate', type=float, default=0.0,
                                help='dropout rate')
    train_settings.add_argument('--batch_size', type=int, default=128,
                                help='batch size')
    train_settings.add_argument('--train_epoch', type=int, default=1,
                                help='number of training epochs')
    train_settings.add_argument('--eval_step', type=int, default=20000,
                               help='the frequency of evaluating on the dev set when training')
    train_settings.add_argument('--save_step', type=int, default=9999999999999999,
                               help='the frequency of saving model')
    train_settings.add_argument('--patience', type=int, default=5,
                               help='lr decay when more than the patience times of evaluation\' loss don\'t decrease')
    train_settings.add_argument('--lr_decay', type=float, default=0.8,
                               help='lr decay')
    train_settings.add_argument('--global_step', type=int, default=0,
                                help='global step')
    train_settings.add_argument('--load_model_path',
                               help='load model path')
    train_settings.add_argument('--embed_init', default='glorot_uniform',
                                help='embeddings initializer')
    train_settings.add_argument('--tb_port', type=int, default='8980',
                                help='tensorboard port')
    train_settings.add_argument('--cotrain', type=bool, default=False,
                                help='cotrain')

    path_settings = parser.add_argument_group('path settings')

    path_settings.add_argument('--train_files', nargs='+',
                               default=['./dataset/dataset_0929/all_cate_hist_graph/train_cate_hist_graph.tsv'],
                               help='list of files that contain the preprocessed train data')
    path_settings.add_argument('--dev_files', nargs='+',
                               default=['./dataset/dataset_0929/all_cate_hist_graph/dev_cate_hist_graph.tsv'],
                               help='list of files that contain the preprocessed dev data')
    path_settings.add_argument('--test_files', nargs='+',
                               default=['./dataset/dataset_0929/all_cate_hist_graph/test_cate_hist_graph.tsv'],
                               help='list of files that contain the preprocessed test data')

    # path_settings.add_argument('--train_files', nargs='+',
    #                            default=['./dataset/dataset_0929/all_cate_hist_graph_done/train_cate_hist_graph.tsv.bz2'],
    #                            help='list of files that contain the preprocessed train data')
    # path_settings.add_argument('--dev_files', nargs='+',
    #                            default=['./dataset/dataset_0929/all_cate_hist_graph_done/dev_cate_hist_graph.tsv.bz2'],
    #                            help='list of files that contain the preprocessed dev data')
    # path_settings.add_argument('--test_files', nargs='+',
    #                            default=['./dataset/dataset_0929/all_cate_hist_graph_done/test_cate_hist_graph.tsv.bz2'],
    #                            help='list of files that contain the preprocessed test data')

    path_settings.add_argument('--user_feat_table_path',
                               default='./dataset/dataset_0929/user_feat_table_all.pkl')
    path_settings.add_argument('--item_feat_table_path',
                               default='./dataset/dataset_0929/item_feat_table_all.pkl')
    # path_settings.add_argument('--user_cate_item_table_path',
    #                            default='./dataset/dataset_0929/user_cate_item_table_all.pkl')
    # path_settings.add_argument('--item_user_table_path',
    #                            default='./dataset/dataset_0929/item_user_table_all.pkl')

    path_settings.add_argument('--ckpt_dir', default='./logs/ckpt/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='./logs/result/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='./logs/summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')


    return parser.parse_args()

def prepare(args):
    """
    prepare the data
    """
    pass



def train(args, algo):
    """
    trains the search graph model
    """
    logger = logging.getLogger(args.model_name)
    logger.info('Checking the data files...')
    for data_path in args.train_files + args.dev_files + args.test_files:
        assert os.path.exists(data_path), '{} does not exist.'.format(data_path)
    logger.info('Load dataset...')
    logger.info('Initialize the model...')
    model = algo(args)
    logger.info('Initialize global_step: {}'.format(model.global_step))
    model.train()
    logger.info('Done with model training!')


def evaluate(args, algo):
    """
    evaluate the trained model on dev files
    """
    pass


def predict(args, algo):
    """
    predicte for test files
    """
    pass



def run():
    args = parse_args()
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

    logger.info('Running with args : {}'.format(args))

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    logger.info('Checking the files and directories...')
    for dir_path in [args.ckpt_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    if args.algo == 'sgm':
        algo = SGM
    elif args.algo == 'sgm_v2':
        algo = SGM_v2
    elif args.algo == 'sgm_no_query':
        algo = SGM_no_query
    elif args.algo == 'sgm_no_intent':
        algo = SGM_no_intent
    elif args.algo == 'sgm_only_node':
        algo = SGM_only_node
    elif args.algo == 'sgm_no_upstream':
        algo = SGM_no_upstream
    elif args.algo =='din':
        algo = DIN
    elif args.algo == 'dnn_no_hist':
        algo = DNN_no_hist
    elif args.algo == 'dnn_hist':
        algo = DNN_hist
    elif args.algo == 'xdeepfm':
        algo = xDeepFM
    elif args.algo == 'nfm':
        algo = NFM
    elif args.algo == 'ngcf':
        algo = NGCF
    elif args.algo == 'gcn':
        algo = GCN
    else:
        raise Exception('Invalid algo:', args.algo)

    if args.prepare:
        prepare(args)
    elif args.train:
        train(args, algo)
    elif args.evaluate:
        evaluate(args, algo)
    elif args.predict:
        predict(args, algo)
    else:
        train(args, algo)
    logger.info('run done.')

if __name__ == '__main__':
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    K.set_session(tf.Session(config=config))
    run()
